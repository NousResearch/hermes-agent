"""A cron report longer than the chat cap is shortened only where it is sent.

The full text stays in the cron output file and in the durable delivery queue
(a restart must still be able to deliver the real report). The live adapter
receives the capped text.
"""

import cron.scheduler_delivery as sd


def _job():
    return {"id": "job-1", "name": "daily"}


def _caps(monkeypatch, lines=6, chars=900):
    monkeypatch.setattr(sd, "_cron_delivery_caps", lambda: (lines, chars))


def test_cap_keeps_a_short_report_verbatim(monkeypatch):
    _caps(monkeypatch)
    text = "line one\nline two"
    assert sd._cap_user_facing_delivery(_job(), text) == text


def test_cap_keeps_the_head_and_names_the_saved_output(monkeypatch):
    _caps(monkeypatch)
    lines = [f"finding {i}" for i in range(sd._DEFAULT_CRON_DELIVERY_MAX_LINES + 25)]
    capped = sd._cap_user_facing_delivery(_job(), "\n".join(lines))
    kept = [line for line in capped.splitlines() if line.startswith("finding ")]
    assert kept[0] == "finding 0"
    assert len(kept) == sd._DEFAULT_CRON_DELIVERY_MAX_LINES
    assert "cron/output/job-1/" in capped
    assert f"finding {sd._DEFAULT_CRON_DELIVERY_MAX_LINES}" not in capped


def test_cap_respects_the_character_budget_on_one_long_line(monkeypatch):
    _caps(monkeypatch)
    capped = sd._cap_user_facing_delivery(
        _job(), "x" * (sd._DEFAULT_CRON_DELIVERY_MAX_CHARS + 5000))
    body = capped.split("[trimmed:", 1)[0].rstrip()
    assert len(body) <= sd._DEFAULT_CRON_DELIVERY_MAX_CHARS
    assert "cron/output/job-1/" in capped


def test_a_zero_cap_disables_trimming(monkeypatch):
    _caps(monkeypatch, lines=0, chars=0)
    text = "\n".join(f"row {i}" for i in range(40))
    assert sd._cap_user_facing_delivery(_job(), text) == text


def test_live_lane_sends_the_capped_text_only(monkeypatch):
    _caps(monkeypatch)
    sent = []
    monkeypatch.setattr(
        sd, "_live_send_text",
        lambda t, text, thread, meta, **k: sent.append(text) or (True, False, "1"))
    monkeypatch.setattr(sd, "_live_route_metadata", lambda t: (None, {}, {}))
    monkeypatch.setattr(sd, "_seed_live_delivery_sessions", lambda t, message_id: None)

    class Target:
        job = _job()
        where = "telegram:-100"
        platform_name = "telegram"
        chat_id = "-100"
        thread_id = None
        origin = {}
        is_relay = False
        mirror_this_target = False
        in_channel_surface = False
        inchannel_continuable = False
        live_adapter_ready = True
        opened_thread_id = None
        origin_user_id = None
        mirror_text = ""

    long_text = "\n".join(f"row {i}" for i in range(sd._DEFAULT_CRON_DELIVERY_MAX_LINES + 10))
    assert sd._deliver_via_live_adapter(
        Target(), long_text, [], target_errors=[], delivery_errors=[], unverified_targets=[])
    assert sent and "cron/output/job-1/" in sent[0]
    assert "row 0" in sent[0]
    assert f"row {sd._DEFAULT_CRON_DELIVERY_MAX_LINES + 5}" not in sent[0]
