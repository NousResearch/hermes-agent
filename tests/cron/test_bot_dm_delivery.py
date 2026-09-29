"""Regression for #116210: message_agent used to hard-fail with reason=target_busy when a
local teammate's Bot Chat was open on another surface. Now the message is queued and the
cron ticker's drain (see cron/scheduler_tick.py) delivers it exactly once the target frees,
or logs loudly once its TTL elapses.
"""
from __future__ import annotations

import json
import sys

import pytest

from cron import bot_dm_delivery as queue
from tools import bot_mode_dm


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _busy_child(tmp_path, busy_flag, calls_file):
    """A fake 'hermes -p ops chat -Q --query-file <dm>' transport: refuses with the
    live-owner code while ``busy_flag`` exists, otherwise succeeds. Every invocation
    appends one line to ``calls_file`` so a test can assert exactly how many turns ran."""
    child = tmp_path / "child.py"
    child.write_text(
        "import sys\n"
        f"open({str(calls_file)!r}, 'a').write('x')\n"
        f"if __import__('os').path.exists({str(busy_flag)!r}):\n"
        "    print('hermes-refusal-reason: SESSION_NOT_OWNED', file=sys.stderr)\n"
        "    print('Session abc already has a live owner (desktop, pid 1).', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "print('delivered reply')\n",
        encoding="utf-8",
    )
    return child


def test_busy_dm_is_queued_then_delivered_exactly_once_when_target_frees(tmp_path, home):
    busy_flag = tmp_path / "busy"
    calls_file = tmp_path / "calls"
    busy_flag.touch()
    child = _busy_child(tmp_path, busy_flag, calls_file)
    argv = [sys.executable, str(child), "-p", "ops"]

    dm = tmp_path / "message.txt"
    dm.write_text("hi @ops", encoding="utf-8")
    delivery_id = bot_mode_dm._dm_delivery_id(dm)

    # 1) Busy: queued, not a hard failure, and the sender is told so (not reason=target_busy).
    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        rc = bot_mode_dm._run_delivery(argv, str(dm), stdin_file=False)
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["status"] == "queued_busy"
    assert payload["delivery_id"] == delivery_id
    assert "error" not in payload and "reason" not in payload
    assert not dm.exists()  # the runner's own temp file is always cleaned up

    record = queue.read_pending(delivery_id)
    assert record["status"] == "queued"
    assert record["content"] == "hi @ops"
    assert record["label"] == "ops"
    assert calls_file.read_text() == "x"  # the busy attempt still ran once

    # 2) Still busy: a drain tick leaves it queued, no second delivery attempt is missed.
    queue.drain()
    assert queue.read_pending(delivery_id)["status"] == "queued"
    assert calls_file.read_text() == "xx"

    # 3) Frees: the NEXT drain tick delivers it — this is the actual Bot Chat turn, so it
    # lands in the target's session history exactly like any other message_agent delivery.
    busy_flag.unlink()
    queue.drain()
    assert queue.read_pending(delivery_id)["status"] == "delivered"
    assert calls_file.read_text() == "xxx"

    # 4) One delivery, no duplicates: further ticks never touch a settled record.
    queue.drain()
    queue.drain()
    assert calls_file.read_text() == "xxx"


def test_ttl_expiry_is_logged_loudly_and_never_retried(tmp_path, home, caplog):
    argv = [sys.executable, "-c", "raise SystemExit('must never run')"]
    now = 1_000_000.0
    record = queue.enqueue_busy_dm(delivery_id="e" * 64, argv=argv, content="stale message",
                                    label="ops", ttl_seconds=60, now=now)
    assert record["status"] == "queued"

    with caplog.at_level("ERROR", logger=queue.logger.name):
        queue.drain(now=now + 61)  # past the 60s TTL

    assert queue.read_pending("e" * 64)["status"] == "expired"
    assert any("expired undelivered" in r.message and "ops" in r.message for r in caplog.records)

    # Expiry is terminal: a later tick must not resurrect or re-log it.
    caplog.clear()
    queue.drain(now=now + 1000)
    assert queue.read_pending("e" * 64)["status"] == "expired"
    assert not caplog.records


def test_enqueue_is_idempotent_on_the_same_delivery_id(tmp_path, home):
    first = queue.enqueue_busy_dm(delivery_id="a" * 64, argv=["hermes"], content="hi", label="ops")
    second = queue.enqueue_busy_dm(delivery_id="a" * 64, argv=["hermes"], content="hi", label="ops")
    assert first == second
    with pytest.raises(ValueError, match="different DM"):
        queue.enqueue_busy_dm(delivery_id="a" * 64, argv=["hermes"], content="different", label="ops")


def test_drain_replay_holds_the_profile_turn_lock(tmp_path, home, monkeypatch):
    """The replay's _run_local_turn call runs under the SAME per-profile turn lock a
    live delivery gets (#93091 — see tools/bot_relay.py::acquire_turn_lock): a
    concurrent live delivery into the same target profile cannot race it. Mirrors the
    proof in tests/tools/test_bot_turn_lock.py::test_run_delivery_holds_profile_lock_
    during_turn — argv0 must be literally 'hermes' for _delivery_lock to engage
    instead of short-circuiting to a no-op (the other tests in this file use a bare
    python script and never exercise the real flock)."""
    from tools.bot_relay import TurnBusyError, acquire_turn_lock

    argv = ["hermes", "-p", "ops", "chat"]
    delivery_id = "f" * 64
    queue.enqueue_busy_dm(delivery_id=delivery_id, argv=argv, content="hi", label="ops")
    observed = {}

    def _fake_run(call_argv, **kwargs):
        # While the replay's turn runs, a second acquire on the same profile must
        # fail — proving the replay genuinely holds the lock, not a no-op.
        with pytest.raises(TurnBusyError):
            with acquire_turn_lock(home, "ops", timeout_seconds=0.15):
                pass  # pragma: no cover — must not acquire concurrently
        observed["argv"] = call_argv

        class _P:
            returncode = 0
            stdout = "delivered"
            stderr = ""

        return _P()

    monkeypatch.setattr(bot_mode_dm.subprocess, "run", _fake_run)
    queue.drain()

    assert observed["argv"][:3] == ["hermes", "-p", "ops"]
    assert queue.read_pending(delivery_id)["status"] == "delivered"
    # ...and after the replay, the lock is free again for the next delivery.
    with acquire_turn_lock(home, "ops", timeout_seconds=0.5):
        pass


def test_unreadable_queued_dm_does_not_block_siblings(tmp_path, home, caplog):
    """One corrupt receipt beside healthy queued work degrades to a logged skip, never an
    aborted drain (same rule as cron/bot_chat_delivery.py and tools/bot_live_delivery.py)."""
    good_argv = [sys.executable, "-c", "print('ok')"]
    queue.enqueue_busy_dm(delivery_id="a" * 64, argv=good_argv, content="healthy", label="ops")
    bad = queue._root() / f"{'e' * 64}.json"
    bad.write_text("not json", encoding="utf-8")

    with caplog.at_level("ERROR", logger=queue.logger.name):
        queue.drain()

    assert queue.read_pending("a" * 64)["status"] == "delivered"
    assert any("Unreadable queued Bot DM" in r.message for r in caplog.records)
