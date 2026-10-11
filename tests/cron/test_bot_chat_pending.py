"""Only never-started cron delivery may wait for a CLI owner's release."""
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from cron import bot_chat_delivery as queue
from cron import scheduler_delivery as delivery
from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB


@pytest.mark.parametrize("error", [None, subprocess.TimeoutExpired("hermes", 1)])
def test_cli_owner_deferral_and_attempt_fence(tmp_path, monkeypatch, error):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(session_id="chat", surface="cli", config={}, registry_home=tmp_path)
    assert refusal is None and lease is not None
    run = Mock(side_effect=error, return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    monkeypatch.setattr(delivery.shutil, "which", lambda _: "/bin/hermes")
    job = {"id": "job", "execution_id": "execution"}
    try:
        assert "queued" in delivery._deliver_to_bot_chat(job, "output", "")
        key = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]
        queue.drain()
        run.assert_not_called()
        lease.release()
        queue.drain()
        assert run.call_count == 1
        expected = "ambiguous" if error else "settled"
        assert queue.read_pending(key)["status"] == expected
        queue.drain()
        delivery._deliver_to_bot_chat(job, "output", "")
        # The failed turn is never retried; a timeout additionally drains ONE short
        # degraded-delivery marker (its own turn), and its timeout queues nothing more.
        assert run.call_count == (2 if error else 1)
        markers = [r for _, r in queue._records(queue._root()) if r.get("degraded")]
        assert len(markers) == (1 if error else 0)
    finally:
        lease.release()
        db.close()


def test_delivery_exception_retains_attempt_and_continues_siblings(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    blocked_parent = tmp_path / "blocked"
    blocked_home = blocked_parent / "recipient"
    blocked_home.mkdir(parents=True)
    for home in (blocked_home, tmp_path):
        db = SessionDB(db_path=home / "state.db")
        db.create_session(session_id="chat", source="cli")
        db.set_session_title("chat", "Bot Chat")
        db.close()
    queue.defer("b" * 64, {"id": "bad"}, "bad output", "", blocked_home)
    queue.defer("a" * 64, {"id": "good"}, "good output", "", tmp_path)
    calls = []
    original_is_dir = Path.is_dir
    armed = False

    def resolve_cli(name, *args, **kwargs):
        # Delivery resolves the CLI (the running install's ``hermes_cli``) right after discovery.
        nonlocal armed
        armed = True
        return object()

    def is_dir(self):
        if armed and self == blocked_home:
            raise PermissionError("target traversal denied after discovery")
        return original_is_dir(self)

    def run(argv, env, report_path, timeout):
        calls.append(env["HERMES_HOME"])
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(importlib.util, "find_spec", resolve_cli)
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    monkeypatch.setattr(Path, "is_dir", is_dir)
    queue.drain()
    assert queue.read_pending("b" * 64)["status"] == "ambiguous"
    assert "PermissionError" in queue.read_pending("b" * 64)["error"]
    assert queue.read_pending("a" * 64)["status"] == "settled"
    queue.drain()
    assert calls == [str(tmp_path)]


def test_pending_queue_uses_admission_order_and_keeps_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {"id": "job"}
    queue.defer("f" * 64, job, "older", "", tmp_path)
    queue.defer("a" * 64, job, "newer", "", tmp_path)
    seen = []

    def interrupted(job, content, profile, **kwargs):
        seen.append(content)
        raise KeyboardInterrupt

    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", interrupted)
    with pytest.raises(KeyboardInterrupt):
        queue.drain()
    assert seen == ["older"]
    assert queue.read_pending("f" * 64)["status"] == "claimed"
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p, **kw: seen.append(c))
    queue.drain()
    queue.drain()
    assert seen == ["older", "newer"]


def test_policy_change_settles_diagnostic_without_waiting_for_cli_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(session_id="chat", surface="cli", config={}, registry_home=tmp_path)
    assert refusal is None
    run = Mock()
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    job = {"id": "failure", "execution_id": "run"}
    try:
        assert "queued" in delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True)
        key = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]
        assert queue.read_pending(key)["for_failure"] is True
        with pytest.raises(ValueError, match="different payload"):
            queue.defer(key, job, "diagnostic", "", tmp_path, for_failure=False)
        (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: true}")
        queue.drain()
        assert queue.read_pending(key)["status"] == "suppressed"
        queue.drain()
        run.assert_not_called()
    finally:
        lease.release()
        db.close()


def test_live_receipt_outcome_survives_later_suppression(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="tui")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(session_id="chat", surface="desktop", config={}, registry_home=tmp_path,
        metadata={"bot_live_delivery_consumer": True, "live_session_id": "live"})
    assert refusal is None
    job = {"id": "failure", "execution_id": "run"}
    try:
        assert "queued" in delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True)
        (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: true}")
        assert "queued" in delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True)
        assert not job.get("_notification_all_targets_suppressed")
    finally:
        lease.release()
        db.close()

@pytest.mark.skipif(not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() == 0,
                    reason="needs POSIX file permissions for an unreadable receipt")
def test_unreadable_deferred_receipt_does_not_block_siblings(tmp_path, monkeypatch, caplog):
    """One permission-denied receipt beside healthy queued work must degrade to a logged skip,
    not abort the drain (same class as the live-owner mailbox wedge, #109820)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    queue.defer("a" * 64, {"id": "job"}, "healthy", "", tmp_path)
    bad = queue._root() / f"{'e' * 64}.json"
    bad.write_text('{"status": "queued"}', encoding="utf-8")
    bad.chmod(0)
    seen = []
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p, **kw: seen.append(c))
    with caplog.at_level("ERROR", logger=queue.logger.name):
        queue.drain()
        queue.drain()  # every scheduler tick drains; the same bad receipt must not re-log
    assert seen == ["healthy"]
    assert [r for r in caplog.records
            if "Unreadable deferred Bot Chat receipt" in r.message and "Permission denied" in r.message] and \
        sum("Unreadable deferred Bot Chat receipt" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("payload", ["42", '"oops"', "[1, 2, 3]"])
def test_non_dict_deferred_receipt_is_skipped_by_the_drain_and_fails_exact_id_reads_closed(
        tmp_path, monkeypatch, caplog, payload):
    """A receipt that parses but is not a JSON object is a bad file like any other: the drain
    and new admissions skip it (warned once, preserved as evidence) and an exact-id read of it
    never licenses an overwrite."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    queue.defer("a" * 64, {"id": "job"}, "healthy", "", tmp_path)
    bad = queue._root() / f"{'e' * 64}.json"
    bad.write_text(payload, encoding="utf-8")
    seen = []
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p, **kw: seen.append(c))
    with caplog.at_level("ERROR", logger=queue.logger.name):
        queue.drain()
        queue.drain()
        later = queue.defer("f" * 64, {"id": "job"}, "later", "", tmp_path)
    assert seen == ["healthy"] and later["status"] == "queued"
    assert sum("Unreadable deferred Bot Chat receipt" in r.message for r in caplog.records) == 1
    with pytest.raises(ValueError):
        queue.defer("e" * 64, {"id": "job"}, "same id", "", tmp_path)
    assert bad.read_text(encoding="utf-8") == payload


def _stub_tui_server(monkeypatch, home):
    """Fake in-process tui_gateway server: no heavy import, no real-home I/O."""
    import sys
    import threading
    import types
    fake = types.ModuleType("tui_gateway.server")
    fake._sessions = {}
    fake._sessions_lock = threading.RLock()
    fake._detached_ws_transport = object()
    fake._launch_home = lambda: home
    monkeypatch.setitem(sys.modules, "tui_gateway.server", fake)
    return fake


def _inject_visible_desktop_session(srv, home, session_key, *, sid="visible-sid", source="desktop",
                                    finalized=False, transport=None, agent=None,
                                    profile_home=None):
    """Lease-free resumed Desktop record in the (stubbed) tui_gateway server."""
    if transport is None:
        transport = object()
    record = {
        "session_key": session_key,
        "profile_home": str(profile_home if profile_home is not None else home),
        "transport": transport,
        "source": source,
        "agent": agent,
    }
    if finalized:
        record["_finalized"] = True
    srv._sessions[sid] = record
    return sid


def test_lease_free_visible_desktop_defers_and_drain_skips_until_close(tmp_path, monkeypatch):
    """Resumed Desktop Bot Chat holds no lease until its first turn; the CLI fallback must not
    steal it (regression for #129858: SESSION_NOT_OWNED fenced the human out mid-turn)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="desktop")
    db.set_session_title("chat", "Bot Chat")
    from tools import bot_live_delivery as live
    srv = _stub_tui_server(monkeypatch, tmp_path)
    sid = "visible-desktop-sid"
    try:
        _inject_visible_desktop_session(srv, tmp_path, "chat", sid=sid)
        owner = live.find_canonical_owner(tmp_path)
        assert owner is not None
        assert owner.get("surface") == "desktop"
        assert owner.get("session_id") == "chat"
        assert owner.get("live_session_id") == sid
        assert live.find_canonical_live_owner(tmp_path) is None

        run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
        monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
        job = {"id": "job", "execution_id": "execution"}
        assert "queued" in delivery._deliver_to_bot_chat(job, "output", "")
        key = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]
        run.assert_not_called()

        queue.drain()
        run.assert_not_called()
        assert queue.read_pending(key)["status"] == "queued"

        srv._sessions.pop(sid, None)
        assert live.find_canonical_owner(tmp_path) is None

        queue.drain()
        assert run.call_count == 1
        assert queue.read_pending(key)["status"] == "settled"
    finally:
        srv._sessions.pop(sid, None)
        db.close()


def test_visible_session_matching_rules(tmp_path, monkeypatch):
    """Only a live, same-profile, non-gateway visible session blocks the fallback."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="desktop")
    db.set_session_title("chat", "Bot Chat")
    from tools import bot_live_delivery as live
    srv = _stub_tui_server(monkeypatch, tmp_path)
    try:
        _inject_visible_desktop_session(
            srv, tmp_path, "chat", sid="other-profile",
            profile_home=str(tmp_path / "other"))
        assert live.find_canonical_owner(tmp_path) is None
        srv._sessions.pop("other-profile", None)

        _inject_visible_desktop_session(
            srv, tmp_path, "chat", sid="gateway-source", source="telegram")
        assert live.find_canonical_owner(tmp_path) is None
        srv._sessions.pop("gateway-source", None)

        class _Agent:
            session_id = "chat"

        _inject_visible_desktop_session(
            srv, tmp_path, "stale-key", sid="agent-tip", agent=_Agent())
        owner = live.find_canonical_owner(tmp_path)
        assert owner is not None
        assert owner.get("session_id") == "chat"
        assert owner.get("live_session_id") == "agent-tip"
        assert live.find_canonical_live_owner(tmp_path) is None
    finally:
        srv._sessions.clear()
        db.close()


def test_finalized_or_detached_visible_session_does_not_block_drain(tmp_path, monkeypatch):
    """A finalized record or a detached transport is not a visible owner: the drain may proceed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="desktop")
    db.set_session_title("chat", "Bot Chat")
    from tools import bot_live_delivery as live
    srv = _stub_tui_server(monkeypatch, tmp_path)
    try:
        _inject_visible_desktop_session(srv, tmp_path, "chat", sid="finalized", finalized=True)
        assert live.find_canonical_owner(tmp_path) is None
        srv._sessions.pop("finalized", None)

        _inject_visible_desktop_session(
            srv, tmp_path, "chat", sid="detached",
            transport=srv._detached_ws_transport)
        assert live.find_canonical_owner(tmp_path) is None
    finally:
        srv._sessions.clear()
        db.close()


def test_lease_free_visible_desktop_deferral_ceiling_fallback(tmp_path, monkeypatch):
    """If the user leaves the resumed chat open without typing, the deferral must end:
    past _DEFER_TTL_SECONDS or _MAX_DEFER_TICKS the drain delivers anyway (#129892)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="desktop")
    db.set_session_title("chat", "Bot Chat")
    srv = _stub_tui_server(monkeypatch, tmp_path)
    sid = "ceiling-desktop-sid"
    try:
        _inject_visible_desktop_session(srv, tmp_path, "chat", sid=sid)
        run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
        monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
        job = {"id": "job", "execution_id": "execution"}
        assert "queued" in delivery._deliver_to_bot_chat(job, "output", "")
        key = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]

        queue.drain()
        run.assert_not_called()
        pending = queue.read_pending(key)
        assert pending["status"] == "queued"
        assert pending["defer_count"] == 1

        # TTL path: an aged record falls back to delivery on the next drain.
        receipt = queue._root() / f"{key}.json"
        record = json.loads(receipt.read_text(encoding="utf-8-sig"))
        record["created_at"] = time.time() - queue._DEFER_TTL_SECONDS - 1
        receipt.write_text(json.dumps(record), encoding="utf-8")
        queue.drain()
        assert run.call_count == 1
        assert queue.read_pending(key)["status"] == "settled"

        # Tick-count path: a record at the tick ceiling delivers on the next drain.
        job2 = {"id": "job2", "execution_id": "execution2"}
        assert "queued" in delivery._deliver_to_bot_chat(job2, "output2", "")
        key2 = job2["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]
        receipt2 = queue._root() / f"{key2}.json"
        record2 = json.loads(receipt2.read_text(encoding="utf-8-sig"))
        record2["defer_count"] = queue._MAX_DEFER_TICKS - 1
        receipt2.write_text(json.dumps(record2), encoding="utf-8")
        queue.drain()
        assert run.call_count == 2
        assert queue.read_pending(key2)["status"] == "settled"
    finally:
        srv._sessions.pop(sid, None)
        db.close()
