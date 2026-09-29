"""Optional backlog triage: off by default, shadow never acts, failures fail open."""
import pytest

from cron import backlog_triage
from cron import bot_chat_delivery as queue
from cron import scheduler_delivery as delivery


def test_off_by_default_never_calls_the_triage_service(monkeypatch):
    monkeypatch.delenv("HERMES_BACKLOG_TRIAGE", raising=False)
    called = []
    monkeypatch.setattr(backlog_triage, "classify", lambda text: called.append(text) or "suppress")
    assert backlog_triage.should_suppress("anything") is False
    assert called == []


@pytest.mark.parametrize("mode", ["off", "0", "false", "no"])
def test_explicit_off_modes_do_not_classify(mode, monkeypatch):
    monkeypatch.setenv("HERMES_BACKLOG_TRIAGE", mode)
    monkeypatch.setattr(backlog_triage, "classify", lambda text: "suppress")
    assert backlog_triage.should_suppress("anything") is False


def test_shadow_classifies_but_never_acts(monkeypatch):
    monkeypatch.setenv("HERMES_BACKLOG_TRIAGE", "shadow")
    monkeypatch.setattr(backlog_triage, "classify", lambda text: "suppress")
    assert backlog_triage.should_suppress("noise", source="test") is False


def test_on_acts_only_on_suppress(monkeypatch):
    monkeypatch.setenv("HERMES_BACKLOG_TRIAGE", "on")
    for verdict, expected in [("suppress", True), ("watch", False),
                              ("review", False), ("notify", False), ("page", False)]:
        monkeypatch.setattr(backlog_triage, "classify", lambda text, v=verdict: v)
        assert backlog_triage.should_suppress("x") is expected


def test_service_failure_fails_open(monkeypatch):
    monkeypatch.setenv("HERMES_BACKLOG_TRIAGE", "on")
    monkeypatch.setattr(backlog_triage, "classify", lambda text: None)
    assert backlog_triage.should_suppress("x") is False


def test_drain_settles_suppressed_records_and_leaves_others_queued(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_BACKLOG_TRIAGE", "on")
    monkeypatch.setattr(backlog_triage, "classify",
                        lambda text: "suppress" if "noise" in text else "review")
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda *a, **kw: None)
    queue.defer("a" * 64, {"id": "noisy"}, "routine noise", "", tmp_path)
    queue.defer("b" * 64, {"id": "real"}, "important output", "", tmp_path)
    queue.drain()
    assert queue.read_pending("a" * 64)["status"] == "suppressed"
    # Non-suppress verdicts proceed through the normal delivery path.
    assert queue.read_pending("b" * 64)["status"] != "queued"
