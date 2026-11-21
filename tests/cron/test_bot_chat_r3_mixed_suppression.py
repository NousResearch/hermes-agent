"""B4: locally suppressed targets need no live-authority receipt.

Exercise the real scheduler/queue/outcome path; only inference transport and
configuration discovery are replaced. No fabricated live suppression receipt.
"""
import json

import pytest

from cron import bot_chat_legacy as legacy, delivery_outcome, delivery_queue as queue
from cron import executions, jobs, scheduler, scheduler_delivery as delivery
from tools import bot_live_delivery as live


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    from gateway import warning_notifications
    from hermes_cli import config_effective, profiles as profile_api

    peer = tmp_path / "profiles/peer"
    peer.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "cron/deliveries.db")
    monkeypatch.setattr(scheduler, "load_config", dict)
    monkeypatch.setattr(scheduler, "run_job", lambda *a, **kw: (True, "raw", "body", None))
    monkeypatch.setattr(profile_api, "profile_exists", lambda name: name == "peer")
    monkeypatch.setattr(profile_api, "get_profile_dir", lambda name: peer)
    monkeypatch.setattr(config_effective, "load_user_config_effective",
                        lambda path: {"allow": path.parent == peer})
    monkeypatch.setattr(warning_notifications, "warning_notifications_enabled",
                        lambda platform, config: config.get("allow", True))
    monkeypatch.setattr(live, "find_canonical_live_owner", lambda home: {"session_id": "bot"})
    for home in (tmp_path, peer):
        (home / "state.db").touch()
    return tmp_path, peer


@pytest.mark.parametrize("keep", [0, 1000])
@pytest.mark.parametrize("completion,expected", [
    ("settled", "delivered"), ("failed", "failed"), ("suppressed", "suppressed"),
])
def test_suppressed_peer_does_not_block_original_run(profiles, monkeypatch, keep, completion, expected):
    home, peer = profiles
    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="bot-chat,bot-chat:peer")
    scheduler.run_one_job(job)
    execution = executions.latest_execution(job["id"])["id"]
    with queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET for_failure=1 WHERE execution_id=?", (execution,))
    identity = dict(job, execution_id=execution)
    own_key = delivery._bot_chat_delivery_key(identity, home, home)
    peer_key = delivery._bot_chat_delivery_key(identity, home, peer)
    admitted = []

    def transport(destination, owner, message, **kwargs):
        assert destination == peer
        admitted.append(kwargs["delivery_id"])
        root = live._root(destination)
        root.mkdir(parents=True, exist_ok=True)
        receipt = dict(id=peer_key, delivery_id=peer_key, message=message,
                       notification_category="diagnostic", status="queued")
        legacy.atomic_json_write(root / (peer_key + ".json"), receipt)
        return receipt

    monkeypatch.setattr(live, "deliver_to_live_owner", transport)
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", keep)
    assert scheduler.drain_delivery_queue({}, None) == 1
    pending = queue.get_status(execution)
    assert pending["status"] == "pending" and pending["next_attempt_at"] is None
    assert delivery_outcome.settled_outcome(execution) is None
    assert queue.claim_next() is None
    assert live.read_delivery_result(home, own_key) is None
    reference = json.loads(pending["bot_chat_receipt"])
    suppressed = next(t for t in reference["targets"] if t["delivery_id"] == own_key)
    assert suppressed.get("local_disposition") == "suppressed"
    # Delete bounded predecessor evidence and change policy: neither is authority
    # for this already-dispatched target once its disposition is persisted.
    legacy._receipt_path(legacy._root(home), own_key).unlink()
    from gateway import warning_notifications
    monkeypatch.setattr(warning_notifications, "warning_notifications_enabled", lambda *a: True)
    legacy.atomic_json_write(live._root(peer) / (peer_key + ".json"),
                             dict(id=peer_key, delivery_id=peer_key, status=completion,
                                  error="peer failed" if completion == "failed" else None))
    assert scheduler.drain_delivery_queue({}, None) == 0
    assert delivery_outcome.settled_outcome(execution) == expected
    assert executions.get_execution(execution)["delivery_outcome"] == expected
    assert jobs.get_job(job["id"])["last_status"] == ("delivery_failed" if expected == "failed" else "ok")
    terminal = queue.get_status(execution)
    assert terminal["status"] == expected
    assert terminal.get("job_json", "{}") == "{}" and terminal.get("content", "") == ""
    assert next(t for t in json.loads(terminal["bot_chat_receipt"])["targets"]
                if t["delivery_id"] == own_key)["local_disposition"] == "suppressed"
    assert admitted == [peer_key]
    assert live.read_delivery_result(home, own_key) is None
    if keep == 0:
        with queue._transaction() as conn:
            assert conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM delivery_tombstones").fetchone()[0] == 1
