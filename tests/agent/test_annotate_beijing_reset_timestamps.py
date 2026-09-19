"""Annotate naive Beijing-time "reset at" timestamps in provider error summaries.

Z.AI / Zhipu usage-limit 429 bodies phrase the quota reset as "Your limit will
reset at 2026-09-04 19:01:25" — a *naive* Beijing-time (UTC+8) timestamp with no
zone marker. Rendered verbatim to a user in another timezone it misleads by the
zone offset (6h for Europe/Warsaw in summer). These tests lock the contract:

* when the Beijing interpretation lands in the sanity window (strictly future,
  ≤ 6h ahead), the timestamp gains "(Beijing time, UTC+8; local: …)";
* outside the window (a UTC or already-local timestamp would land there), the
  text passes through unchanged — the guard makes a wrong annotation
  impossible, so a provider switch to UTC/local silently disables the feature;
* no "reset at" phrase, no annotation;
* already-annotated text is never annotated twice (idempotence);
* the annotation survives the JSON-body and httpx-fallback summary paths, not
  just the direct-message path.

Timestamps are computed relative to ``datetime.now()`` so the tests never go
stale (behavior contracts, not frozen snapshots).
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from run_agent import AIAgent

_BEIJING = ZoneInfo("Asia/Shanghai")


def _beijing_ts_ahead(hours: float) -> str:
    """A naive Beijing-time timestamp ``hours`` ahead of now, as Z.AI sends it."""
    ts = datetime.now(_BEIJING) + timedelta(hours=hours)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _zai_body(hours: float) -> str:
    return (
        "Usage limit reached for 5 hour. "
        f"Your limit will reset at {_beijing_ts_ahead(hours)}"
    )


def test_reset_timestamp_in_window_is_annotated_with_local_time():
    text = _zai_body(2)

    result = AIAgent._annotate_naive_beijing_reset_timestamp(text)

    assert "(Beijing time, UTC+8; local:" in result
    # The original verbatim timestamp is preserved (annotation is additive).
    assert "reset at 20" in result or "reset at 1" in result or "reset at 0" in result


def test_reset_timestamp_outside_window_passes_through_unchanged():
    # A UTC or already-local timestamp interprets as ~6-8h AHEAD when read as
    # Beijing (outside the 6h window) or BEHIND now — both must pass through.
    for hours in (-2, 8):
        text = _zai_body(hours)

        assert AIAgent._annotate_naive_beijing_reset_timestamp(text) == text


def test_no_reset_at_phrase_is_untouched():
    text = "Usage limit reached for 5 hour. Try again later."

    assert AIAgent._annotate_naive_beijing_reset_timestamp(text) == text


def test_already_annotated_text_is_idempotent():
    text = AIAgent._annotate_naive_beijing_reset_timestamp(_zai_body(2))
    assert "(Beijing time, UTC+8; local:" in text  # sanity: first pass annotated

    assert AIAgent._annotate_naive_beijing_reset_timestamp(text) == text


def test_local_conversion_is_arithmetically_correct(monkeypatch):
    # Pin the zone the method resolves: the annotated local time must be the
    # SAME instant as the Beijing timestamp, expressed in the pinned zone —
    # regardless of the machine's container zone.
    import hermes_time

    warsaw = ZoneInfo("Europe/Warsaw")
    monkeypatch.setattr(hermes_time, "get_timezone", lambda: warsaw)

    ts = datetime.now(_BEIJING) + timedelta(hours=2)
    text = (
        "Usage limit reached for 5 hour. "
        f"Your limit will reset at {ts.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    result = AIAgent._annotate_naive_beijing_reset_timestamp(text)

    local_str = result.split("local: ")[-1].rstrip(")")
    annotated_local = datetime.strptime(local_str, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=warsaw
    )
    expected = ts.astimezone(warsaw)
    assert abs((annotated_local - expected).total_seconds()) < 5  # same instant


def test_json_body_error_message_is_annotated():
    err = Exception("")
    err.status_code = 429
    err.body = {"error": {"message": _zai_body(1), "type": "rate_limit_exceeded"}}

    summary = AIAgent._summarize_api_error(err)

    assert "HTTP 429" in summary
    assert "(Beijing time, UTC+8; local:" in summary


def test_httpx_fallback_payload_is_annotated():
    payload = '{"error": {"message": "%s"}}' % _zai_body(1)
    err = Exception("")
    err.status_code = 429
    err.body = {}  # empty — forces the httpx response.text fallback (#36109 path)
    err.response = SimpleNamespace(text=payload)

    summary = AIAgent._summarize_api_error(err)

    assert "HTTP 429" in summary
    assert "(Beijing time, UTC+8; local:" in summary
