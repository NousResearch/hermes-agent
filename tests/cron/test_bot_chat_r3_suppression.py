"""Local suppression is terminal evidence, not a nonexistent authority receipt."""
import json

import pytest
from cron import bot_chat_legacy as legacy, delivery_outcome, delivery_queue as queue
from cron import executions, jobs, scheduler as sched, scheduler_delivery as delivery
from tools import bot_live_delivery as live


@pytest.fixture
def store(tmp_path, monkeypatch):
    from gateway import warning_notifications
    from hermes_cli import config_effective, profiles

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "cron/deliveries.db")
    monkeypatch.setattr(sched, "load_config", dict)
    monkeypatch.setattr(sched, "run_job", lambda job, **kw: (True, "raw", "body", None))
    peer = tmp_path / "profiles/peer"
    peer.mkdir(parents=True)
    for home in (tmp_path, peer):
        (home / "state.db").touch()
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: peer)
    monkeypatch.setattr(config_effective, "load_user_config_effective",
                        lambda path: {"allow": path.parent == peer})
    monkeypatch.setattr(warning_notifications, "warning_notifications_enabled",
                        lambda platform, config: config.get("allow", True))
    monkeypatch.setattr(live, "find_canonical_live_owner", lambda home: {"session_id": "bot"})
    return tmp_path, peer


@pytest.mark.parametrize("keep", [0, 1000])
@pytest.mark.parametrize("completion", ["settled", "failed", "suppressed", "immediate"])
def test_mixed_local_suppression_survives_handoff_and_tombstone(store, monkeypatch, keep, completion):
    home, peer = store
    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="bot-chat,bot-chat:peer")
    sched.run_one_job(job)
    execution = executions.latest_execution(job["id"])["id"]
    # Exercise diagnostic routing on the real queued run/ledger, not fake bookkeeping.
    with queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET for_failure=1 WHERE execution_id=?", (execution,))
    original_job = dict(job, execution_id=execution)
    own_key = delivery._bot_chat_delivery_key(original_job, home, home)
    peer_key = delivery._bot_chat_delivery_key(original_job, home, peer)
    calls = []

    def admit(destination, owner, message, **kwargs):
        assert destination == peer, "suppressed own target must not reach transport"
        calls.append(kwargs["delivery_id"])
        root = live._root(destination)
        root.mkdir(parents=True, exist_ok=True)
        receipt = dict(id=peer_key, delivery_id=peer_key, message=message,
                       notification_category="diagnostic",
                       status="settled" if completion == "immediate" else "queued")
        legacy.atomic_json_write(root / (peer_key + ".json"), receipt)
        return receipt

    monkeypatch.setattr(live, "deliver_to_live_owner", admit)
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", keep)
    assert sched.drain_delivery_queue({}, None) == 1
    row = queue.get_status(execution)
    reference = json.loads(row["bot_chat_receipt"])
    own = next(t for t in reference["targets"] if t["delivery_id"] == own_key)
    assert own.get("local_disposition") == "suppressed", "local suppression was dropped before parking/finish"
    assert live.read_delivery_result(home, own_key) is None
    assert legacy.read_pending(own_key)["status"] == "suppressed"
    assert row.get("job_json", "{}") == "{}" and row.get("content", "") == ""
    # Neither current policy nor the bounded legacy JSON receipt is needed later.
    legacy._receipt_path(legacy._root(), own_key).unlink()
    from gateway import warning_notifications
    monkeypatch.setattr(warning_notifications, "warning_notifications_enabled", lambda *a: True)
    if completion != "immediate":
        assert row["status"] == "pending" and row["next_attempt_at"] is None
        assert delivery_outcome.settled_outcome(execution) is None
        assert queue.claim_next() is None
        legacy.atomic_json_write(live._root(peer) / (peer_key + ".json"),
            dict(id=peer_key, delivery_id=peer_key, status=completion,
                 error="peer failed" if completion == "failed" else None))
    expected = "failed" if completion == "failed" else "suppressed" if completion == "suppressed" else "delivered"
    assert sched.drain_delivery_queue({}, None) == 0
    assert delivery_outcome.settled_outcome(execution) == expected
    assert executions.get_execution(execution)["delivery_outcome"] == expected
    assert jobs.get_job(job["id"])["last_status"] == ("delivery_failed" if expected == "failed" else "ok")
    terminal = queue.get_status(execution)
    assert terminal["status"] == expected
    assert next(t for t in json.loads(terminal["bot_chat_receipt"])["targets"]
                if t["delivery_id"] == own_key)["local_disposition"] == "suppressed"
    assert calls == [peer_key]
    assert live.read_delivery_result(home, own_key) is None
