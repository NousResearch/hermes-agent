import sys
from datetime import datetime, timezone, UTC
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from gateway import message_timestamps
from gateway.message_timestamps import (
    coerce_message_timestamp,
    format_message_timestamp,
    render_user_content_with_timestamp,
)
from hermes_time import safe_strftime


BERLIN = ZoneInfo("Europe/Berlin")


def _epoch(year, month, day, hour, minute, second):
    return datetime(year, month, day, hour, minute, second, tzinfo=BERLIN).timestamp()


@pytest.mark.parametrize("epoch", [1.0, 1_000_000_000.0])
def test_render_numeric_timestamp_preserves_instant_in_system_timezone(epoch):
    # Epoch 1 is still in 1969 west of UTC. Windows rejects a naive
    # astimezone() conversion there, although the Unix timestamp is positive.
    local = datetime.fromtimestamp(epoch, tz=UTC).astimezone()
    prefix = safe_strftime(local, "%a %Y-%m-%d %H:%M:%S %Z")

    assert render_user_content_with_timestamp("hello", epoch) == f"[{prefix}] hello"


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("epoch", [0.0, 1.123456, 86_399.0])
def test_early_naive_iso_preserves_system_local_instant(epoch):
    local = datetime.fromtimestamp(epoch, tz=UTC).astimezone()
    text = local.replace(tzinfo=None).isoformat()

    assert coerce_message_timestamp(text) == pytest.approx(epoch, rel=0, abs=1e-6)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("style", ["human", "iso"])
@pytest.mark.parametrize("epoch", [0.0, 1.0, 86_399.0])
def test_early_embedded_local_time_preserves_instant(style, epoch):
    local = datetime.fromtimestamp(epoch, tz=UTC).astimezone()
    stamp = (safe_strftime(local, "%a %Y-%m-%d %H:%M:%S") if style == "human"
             else local.replace(tzinfo=None).isoformat())
    content = f"[{stamp}] hello"

    assert render_user_content_with_timestamp(content, 1_000_000_000.0) == (
        render_user_content_with_timestamp("hello", epoch)
    )


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("style", ["human", "iso"])
@pytest.mark.parametrize("enabled", [False, True])
def test_early_embedded_local_time_replays_with_injection_on_or_off(style, enabled):
    from gateway.run import _build_gateway_agent_history

    local = datetime.fromtimestamp(1.0, tz=UTC).astimezone()
    stamp = (safe_strftime(local, "%a %Y-%m-%d %H:%M:%S") if style == "human"
             else local.replace(tzinfo=None).isoformat())
    content = f"[{stamp}] hello"
    history, _ = _build_gateway_agent_history(
        [{"role": "user", "content": content, "timestamp": 1_000_000_000.0},
         {"role": "assistant", "content": "hi"}],
        inject_timestamps=enabled,
    )

    expected = render_user_content_with_timestamp("hello", 1.0) if enabled else content
    assert history[0]["content"] == expected
    assert history[1]["content"] == "hi"


@pytest.mark.platforms("windows")
def test_early_embedded_local_time_renders_in_observed_context():
    from gateway.run import _build_gateway_agent_history

    local = datetime.fromtimestamp(1.0, tz=UTC).astimezone()
    content = f"[{local.replace(tzinfo=None).isoformat()}] hello"
    history, observed = _build_gateway_agent_history(
        [{"role": "user", "content": content, "observed": True}],
        inject_timestamps=True, channel_prompt="observed Telegram group context",
    )

    assert history == []
    assert observed == render_user_content_with_timestamp("hello", 1.0)


@pytest.mark.parametrize("fold", [0, 1])
@pytest.mark.parametrize("wall_time", [
    datetime(2026, 1, 15, 12), datetime(2026, 7, 15, 12),
    datetime(2026, 3, 8, 2, 30), datetime(2026, 11, 1, 1, 30),
])
def test_modern_local_conversion_keeps_existing_fold_and_gap_behavior(wall_time, fold):
    dt = wall_time.replace(fold=fold)

    assert message_timestamps._localize(dt, None) == dt.astimezone().timestamp()


@pytest.mark.parametrize("text", ["1969-06-01T12:00:00", "2026-07-15T12:00:00"])
def test_explicit_timezone_parsing_preserves_wall_time(text):
    expected = datetime.fromisoformat(text).replace(tzinfo=BERLIN).timestamp()

    assert coerce_message_timestamp(text, tz=BERLIN) == expected


@pytest.mark.platforms("windows")
@pytest.mark.skipif(sys.platform != "win32", reason="Windows' pre-epoch range limit")
def test_unsupported_local_iso_is_uninterpretable_instead_of_raising():
    assert coerce_message_timestamp("1969-06-01T12:00:00") is None


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("epoch", [float("nan"), float("inf"), -1.0])
@pytest.mark.skipif(sys.platform != "win32", reason="Windows' pre-epoch range limit")
def test_unrenderable_epoch_preserves_the_message_body(epoch):
    assert format_message_timestamp(epoch) == ""
    assert render_user_content_with_timestamp("hello", epoch) == "hello"


@pytest.mark.platforms("windows")
@pytest.mark.skipif(sys.platform != "win32", reason="Windows' pre-epoch range limit")
def test_unsupported_embedded_time_keeps_metadata_fallback():
    content = "[Sun 1969-06-01 12:00:00] hello"

    assert render_user_content_with_timestamp(content, 1_000_000_000.0) == (
        render_user_content_with_timestamp("hello", 1_000_000_000.0)
    )


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.skipif(sys.platform != "win32", reason="Windows' pre-epoch range limit")
def test_unsupported_embedded_time_does_not_abort_history(enabled):
    from gateway.run import _build_gateway_agent_history

    content = "[Sun 1969-06-01 12:00:00] hello"
    history, _ = _build_gateway_agent_history(
        [{"role": "user", "content": content}], inject_timestamps=enabled
    )

    assert history[0]["content"] == ("hello" if enabled else content)


@pytest.mark.platforms("windows")
def test_system_timezone_conversion_starts_from_aware_utc(monkeypatch):
    # Record real datetime calls: output alone also passes with the unsafe
    # conversion when the canonical runner sets TZ=UTC. A modern epoch keeps
    # the failure at this assertion rather than a host-dependent exception.
    datetime_spy = Mock(wraps=datetime)
    monkeypatch.setattr(message_timestamps, "datetime", datetime_spy)
    epoch = 1_000_000_000.0

    render_user_content_with_timestamp("hello", epoch)

    datetime_spy.fromtimestamp.assert_called_once_with(epoch, tz=UTC)


def test_render_user_content_deduplicates_existing_timestamp_and_preserves_embedded_time():
    db_processing_ts = _epoch(2026, 4, 27, 15, 55, 36)
    stored_content = (
        "[Mon 2026-04-27 15:54:44 CEST] "
        "[Example User] This should go on our todo list"
    )

    rendered = render_user_content_with_timestamp(
        stored_content,
        db_processing_ts,
        tz=BERLIN,
    )

    assert rendered == stored_content
    assert rendered.count("2026-04-27") == 1


# ---------------------------------------------------------------------------
# Opt-in gate: gateway.message_timestamps.enabled (default OFF)
# ---------------------------------------------------------------------------




def test_build_history_injects_only_when_enabled():
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "hello", "timestamp": _epoch(2026, 4, 28, 13, 40, 53)},
        {"role": "assistant", "content": "hi"},
    ]

    # Default (off): user content stays clean, no timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history)
    assert agent_history[0]["content"] == "hello"

    # Enabled: user content gets exactly one timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history, inject_timestamps=True)
    assert agent_history[0]["content"].startswith("[")
    assert agent_history[0]["content"].endswith("hello")
    # Assistant message is never timestamped.
    assert agent_history[1]["content"] == "hi"


def test_compaction_summary_replays_without_timestamp():
    from gateway.run import _build_gateway_agent_history

    summary = {
        "role": "user",
        "content": "[CONTEXT COMPACTION] prior turns",
        "timestamp": 1.0,
        "_compressed_summary": True,
    }
    history, _ = _build_gateway_agent_history([summary], inject_timestamps=True)
    assert history[0]["content"] == summary["content"]


def test_regular_user_message_still_replays_with_timestamp():
    from gateway.run import _build_gateway_agent_history

    message = {"role": "user", "content": "what is next?", "timestamp": 1_000_000_000.0}
    history, _ = _build_gateway_agent_history([message], inject_timestamps=True)
    assert history[0]["content"].endswith(message["content"])
    assert history[0]["content"] != message["content"]


def test_compactor_restatement_replays_unstamped_through_session_db(tmp_path):
    """Compaction sends its restated in-flight task unstamped; every reload (also after a re-persist of the
    replayed view) must replay those exact bytes."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from agent.context_compressor import ContextCompressor, _INFLIGHT_TASK_REPLAY_HEADER
    from gateway.config import GatewayConfig
    from gateway.run import _build_gateway_agent_history
    from gateway.session import SessionStore
    from hermes_state import SessionDB
    from hermes_time import get_timezone

    sent_at = _epoch(2026, 4, 28, 13, 40, 53)
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "Summarize project files", "timestamp": sent_at}]
    for i in range(40):
        messages += [
            {"role": "assistant", "content": f"step {i}", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "fixture " * 200},
        ]
    compressor = ContextCompressor(model="m", quiet_mode=True, config_context_length=100_000,
                                   protect_first_n=2, protect_last_n=2)
    compressor.tail_token_budget = 500
    summary = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="## Historical Task Snapshot\nx\n## Summary\ny"), finish_reason="stop")])
    with patch("agent.context_compressor.call_llm", return_value=summary):
        sent = [m for m in compressor.compress(messages, current_tokens=200_000, force=True) if m["role"] != "system"]
    sent.append({"role": "assistant", "content": "Done"})
    # A human pasting the header text is still a message someone sent: it keeps its stamp.
    pasted = f"{_INFLIGHT_TASK_REPLAY_HEADER}\nthis is a pasted log"
    sent += [{"role": "user", "content": pasted, "timestamp": sent_at}, {"role": "assistant", "content": "ok"}]

    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    try:
        db.create_session("room", source="matrix")
        db.archive_and_compact("room", sent)
        first, _ = _build_gateway_agent_history(store.load_transcript("room"), inject_timestamps=True)
        # A later in-session prune re-persists the replayed view; the next reload must still know the row.
        db.archive_and_compact("room", first)
        replay, _ = _build_gateway_agent_history(store.load_transcript("room"), inject_timestamps=True)
    finally:
        db.close()
        store.close_all_db_handles()

    def restatement(rows):
        return next((m["content"] for m in rows if m["role"] == "user"
                     and str(m["content"]).startswith(_INFLIGHT_TASK_REPLAY_HEADER)), None)

    assert restatement(sent) is not None
    assert restatement(first) == restatement(sent)
    assert restatement(replay) == restatement(sent)
    assert next(m for m in replay if m["role"] == "user" and str(m["content"]).endswith("this is a pasted log"))[
        "content"] == render_user_content_with_timestamp(pasted, sent_at, tz=get_timezone())


def test_compaction_summary_stays_unstamped_across_real_prune(tmp_path):
    """A proactive prune re-persists the replayed view; the summary carrier must keep its flag through it."""
    from agent.context_compressor import ContextCompressor
    from gateway.config import GatewayConfig
    from gateway.run import _build_gateway_agent_history
    from gateway.session import SessionStore
    from hermes_state import SessionDB

    summary = {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] retained summary",
               "_compressed_summary": True}
    rows = [summary, {"role": "assistant", "content": "Continue"}]
    for i in range(5):
        rows += [{"role": "assistant", "content": None, "tool_calls": [
                     {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "content": f"file {i} " + f"unique output {i} " * 5000}]
    rows += [{"role": "user", "content": "new task"}, {"role": "assistant", "content": "working"}]
    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    try:
        db.create_session("room", source="matrix")
        db.archive_and_compact("room", rows)
        first, _ = _build_gateway_agent_history(store.load_transcript("room"), inject_timestamps=True)
        compressor = ContextCompressor(model="m", quiet_mode=True, config_context_length=100_000, protect_first_n=1,
                                       protect_last_n=2, proactive_prune_tokens=1, proactive_prune_min_reclaim_tokens=1)
        compressor._session_db, compressor._session_id = db, "room"
        _, pruned = compressor.prune_tool_results_only(first, current_tokens=100_000)
        replay, _ = _build_gateway_agent_history(store.load_transcript("room"), inject_timestamps=True)
    finally:
        db.close()
        store.close_all_db_handles()
    assert pruned
    assert first[0]["content"] == replay[0]["content"] == summary["content"]
