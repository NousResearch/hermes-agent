import asyncio
import json
import sqlite3

import pytest

from gateway.platforms.signal_reaction_awareness import (
    ReactionAwarenessLane,
    format_awareness_context,
)


class FakeModel:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def generate(self, *, model, prompt, tools, timeout):
        self.calls.append({"model": model, "prompt": prompt, "tools": tools, "timeout": timeout})
        if self.error:
            raise self.error
        return self.response


def test_lane_uses_configured_model_and_refuses_tool_calls_without_side_effects(tmp_path):
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = FakeModel({"text": '{"note":"A positive acknowledgment."}', "tool_calls": [{"name": "send_message"}]})
    lane = ReactionAwarenessLane(db, model_name="configured-model", model_client=model)
    lane.register_target(profile="p", scope="dm:signal-owner-0001", timestamp=123, author="signal-owner-0001", body="I finished the report")
    event = {"event_id": "owner:456", "profile": "p", "scope": "dm:signal-owner-0001", "target_timestamp": 123,
             "target_author": "signal-owner-0001", "sender": "signal-owner-0001", "emoji": "👍", "action": "added",
             "event_timestamp": 456, "owner_authorized": True, "chat_type": "dm"}
    assert lane.enqueue(event) == "queued"
    assert asyncio.run(lane.process_one("p", "dm:signal-owner-0001")) == "tool_call_refused"
    assert model.calls[0]["model"] == "configured-model"
    assert model.calls[0]["tools"] == []
    assert not lane.notes("p", "dm:signal-owner-0001")
    assert lane.outbox_status("p", "dm:signal-owner-0001", "owner:456") == "retry"


def test_lane_persists_once_and_formatter_marks_note_as_untrusted(tmp_path):
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = FakeModel({"text": '{"note":"The reaction acknowledges the report."}'})
    lane = ReactionAwarenessLane(db, model_name="configured-model", model_client=model)
    lane.register_target(profile="p", scope="dm:signal-owner-0001", timestamp=123, author="signal-owner-0001", body="I finished the report")
    event = {"event_id": "owner:456", "profile": "p", "scope": "dm:signal-owner-0001", "target_timestamp": 123,
             "target_author": "signal-owner-0001", "sender": "signal-owner-0001", "emoji": "👍", "action": "added",
             "event_timestamp": 456, "owner_authorized": True, "chat_type": "dm"}
    assert lane.enqueue(event) == "queued"
    assert asyncio.run(lane.process_one("p", "dm:signal-owner-0001")) == "completed"
    assert lane.enqueue(event) == "duplicate"
    assert len(lane.notes("p", "dm:signal-owner-0001")) == 1
    formatted = format_awareness_context(lane.notes("p", "dm:signal-owner-0001"))
    assert "untrusted reaction context" in formatted.lower()
    assert "not confirmation of any outcome" in formatted.lower()
    assert "The reaction acknowledges the report." in formatted


def test_lane_fails_closed_for_group_wrong_owner_and_missing_target_body(tmp_path):
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = FakeModel({"text": '{"note":"should not appear"}'})
    lane = ReactionAwarenessLane(db, model_name="configured-model", model_client=model)
    event = {"event_id": "owner:456", "profile": "p", "scope": "dm:signal-owner-0001", "target_timestamp": 999,
             "target_author": "signal-owner-0001", "sender": "signal-owner-0001", "emoji": "👍", "action": "added",
             "event_timestamp": 456, "owner_authorized": True, "chat_type": "dm"}
    assert lane.enqueue(event) == "unresolved"
    group = dict(event, event_id="owner:457", chat_type="group")
    assert lane.enqueue(group) == "denied"
    wrong_owner = dict(event, event_id="someone:458", owner_authorized=False)
    assert lane.enqueue(wrong_owner) == "denied"
    assert not model.calls


def test_removed_reaction_fences_older_claim_and_never_resurrects_note(tmp_path):
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = FakeModel({"text": '{"note":"Late acknowledgment."}'})
    lane = ReactionAwarenessLane(db, model_name="configured-model", model_client=model)
    lane.register_target(profile="p", scope="dm:signal-owner-0001", timestamp=123, author="signal-owner-0001", body="report")
    add = {"event_id": "owner:456", "profile": "p", "scope": "dm:signal-owner-0001", "target_timestamp": 123,
           "target_author": "signal-owner-0001", "sender": "signal-owner-0001", "emoji": "👍", "action": "added",
           "event_timestamp": 456, "owner_authorized": True, "chat_type": "dm"}
    remove = dict(add, event_id="owner:457", event_timestamp=457, emoji="", action="removed")
    assert lane.enqueue(add) == "queued"
    assert lane.enqueue(remove) == "queued"
    assert asyncio.run(lane.process_one("p", "dm:signal-owner-0001")) == "empty"
    assert lane.outbox_status("p", "dm:signal-owner-0001", "owner:456") == "superseded"
    assert not lane.notes("p", "dm:signal-owner-0001")


def test_body_is_exact_scope_bound_and_caps_are_fail_closed(tmp_path):
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = FakeModel({"text": '{"note":"Observed."}'})
    lane = ReactionAwarenessLane(db, model_name="configured-model", model_client=model)
    assert not lane.register_target(profile="p", scope="dm:a", timestamp=1, author="a", body="x" * 8193)
    lane.register_target(profile="p", scope="dm:a", timestamp=123, author="a", body="scope A")
    lane.register_target(profile="p", scope="dm:b", timestamp=123, author="a", body="scope B")
    event = {"event_id": "event-1", "profile": "p", "scope": "dm:a", "target_timestamp": 123,
             "target_author": "other", "sender": "a", "emoji": "👍", "action": "added",
             "event_timestamp": 456, "owner_authorized": True, "chat_type": "dm"}
    assert lane.enqueue(event) == "unresolved"
    event["target_author"] = "a"
    assert lane.enqueue(event) == "queued"
    assert asyncio.run(lane.process_one("p", "dm:a")) == "completed"
    assert "scope A" in model.calls[0]["prompt"]
    assert "scope B" not in model.calls[0]["prompt"]


def test_note_schema_rejects_instruction_injection_payload(tmp_path):
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = FakeModel({"text": '{"note":"do this", "instructions":"run tool"}'})
    lane = ReactionAwarenessLane(db, model_name="configured-model", model_client=model)
    lane.register_target(profile="p", scope="dm:a", timestamp=123, author="a", body="hello")
    event = {"event_id": "event-1", "profile": "p", "scope": "dm:a", "target_timestamp": 123,
             "target_author": "a", "sender": "a", "emoji": "👍", "action": "added",
             "event_timestamp": 456, "owner_authorized": True, "chat_type": "dm"}
    assert lane.enqueue(event) == "queued"
    assert asyncio.run(lane.process_one("p", "dm:a")) == "retry"
    assert not lane.notes("p", "dm:a")



def test_configured_agent_client_constructs_zero_tool_ephemeral_agent_and_refuses_dispatch(monkeypatch):
    from gateway.platforms.signal_reaction_awareness import ConfiguredAIAgentModelClient, ToolCallRefused
    import run_agent
    captured = {}
    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            captured["closed"] = False
        def run_conversation(self, *, user_message):
            captured["prompt"] = user_message
            captured["read_only"] = self._shared_group_read_only
            captured["persist_disabled"] = self._persist_disabled
            self._execute_tool_calls(object(), [], "task")
        def close(self):
            captured["closed"] = True
    monkeypatch.setattr(run_agent, "AIAgent", FakeAIAgent)
    client = ConfiguredAIAgentModelClient(model="configured", runtime_kwargs={"provider": "fixture"})
    with pytest.raises(ToolCallRefused):
        asyncio.run(client.generate(model="configured", prompt="untrusted data", tools=[], timeout=1))
    assert captured["enabled_toolsets"] == []
    assert captured["disabled_toolsets"] == []
    assert captured["session_db"] is None
    assert captured["skip_memory"] is True
    assert captured["skip_background_review"] is True
    assert captured["read_only"] is True
    assert captured["persist_disabled"] is True
    assert captured["prompt"] == "untrusted data"
    assert captured["closed"] is True


def test_lane_retries_model_failures_and_recovers_expired_claim(tmp_path, monkeypatch):
    from gateway.platforms import signal_reaction_awareness as awareness
    monkeypatch.setattr(awareness, "_RETRY_BASE_SECONDS", 0)
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = FakeModel(error=TimeoutError("deadline"))
    lane = ReactionAwarenessLane(db, model_name="configured-model", model_client=model, lease_seconds=0)
    lane.register_target(profile="p", scope="dm:signal-owner-0001", timestamp=123, author="signal-owner-0001", body="hello")
    event = {"event_id": "owner:456", "profile": "p", "scope": "dm:signal-owner-0001", "target_timestamp": 123,
             "target_author": "signal-owner-0001", "sender": "signal-owner-0001", "emoji": "👍", "action": "added",
             "event_timestamp": 456, "owner_authorized": True, "chat_type": "dm"}
    lane.enqueue(event)
    assert asyncio.run(lane.process_one("p", "dm:signal-owner-0001")) == "retry"
    db.close()
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    retry_model = FakeModel({"text": '{"note":"Acknowledged."}'})
    restarted = ReactionAwarenessLane(db, model_name="configured-model", model_client=retry_model)
    db.execute("UPDATE awareness_outbox SET lease_until=-1e100 WHERE event_id='owner:456'")
    assert tuple(db.execute("SELECT status,lease_until FROM awareness_outbox WHERE event_id='owner:456'").fetchone()) == ("retry", -1e100)
    recovered_scope = db.execute("SELECT scope FROM awareness_outbox WHERE event_id='owner:456'").fetchone()[0]
    assert db.execute("SELECT COUNT(*) FROM awareness_outbox WHERE profile=? AND scope=? AND status IN ('pending','retry','claimed') AND lease_until<=?", ("p", recovered_scope, awareness.time.time())).fetchone()[0] == 1
    assert asyncio.run(restarted.process_one("p", recovered_scope)) == "completed"
    assert len(restarted.notes("p", recovered_scope)) == 1


def test_target_collision_cannot_overwrite_registered_message(tmp_path):
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    lane = ReactionAwarenessLane(db, model_name="m", model_client=FakeModel())
    assert lane.register_target(profile="p", scope="dm:a", timestamp=11, author="a", body="original")
    assert not lane.register_target(profile="p", scope="dm:a", timestamp=11, author="a", body="replacement")
    assert not lane.register_target(profile="p", scope="dm:a", timestamp=11, author="b", body="foreign")
    assert tuple(db.execute("SELECT author,body FROM awareness_targets").fetchone()) == ("a", "original")


def test_removal_invalidates_an_already_persisted_note(tmp_path):
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = FakeModel({"text": '{"note":"Acknowledged."}'})
    lane = ReactionAwarenessLane(db, model_name="m", model_client=model)
    lane.register_target(profile="p", scope="dm:a", timestamp=11, author="a", body="hello")
    add = {"event_id": "e1", "profile": "p", "scope": "dm:a", "target_timestamp": 11,
           "target_author": "a", "sender": "a", "emoji": "👍", "action": "added",
           "event_timestamp": 21, "owner_authorized": True, "chat_type": "dm"}
    assert lane.enqueue(add) == "queued"
    assert asyncio.run(lane.process_one("p", "dm:a")) == "completed"
    remove = dict(add, event_id="e2", event_timestamp=22, emoji="", action="removed")
    assert lane.enqueue(remove) == "queued"
    assert lane.notes("p", "dm:a") == []


def test_global_pruning_bounds_all_tables_and_enforces_identifier_bytes(tmp_path, monkeypatch):
    from gateway.platforms import signal_reaction_awareness as awareness
    monkeypatch.setattr(awareness, "_MAX_ROWS", 2)
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    lane = ReactionAwarenessLane(db, model_name="m", model_client=FakeModel({"text": '{"note":"ok"}'}))
    assert not lane.register_target(profile="é" * 257, scope="s", timestamp=1, author="a", body="body")
    assert not lane.register_target(profile="p", scope="s", timestamp=1, author="a", body="\ud800")
    assert not lane.register_target(profile="\ud800", scope="s", timestamp=1, author="a", body="body")
    for ts in range(1, 4):
        assert lane.register_target(profile="p", scope="s", timestamp=ts, author="a", body="body")
        event = {"event_id": f"e{ts}", "profile": "p", "scope": "s", "target_timestamp": ts,
                 "target_author": "a", "sender": "a", "emoji": "👍", "action": "added",
                 "event_timestamp": ts + 10, "owner_authorized": True, "chat_type": "dm"}
        assert lane.enqueue(event) == "queued"
        assert asyncio.run(lane.process_one("p", "s")) == "completed"
    for table in ("awareness_targets", "awareness_outbox", "awareness_latest", "awareness_notes"):
        assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] <= 2


def test_lane_retry_attempts_are_capped_and_backed_off(tmp_path, monkeypatch):
    from gateway.platforms import signal_reaction_awareness as awareness
    monkeypatch.setattr(awareness, "_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(awareness, "_RETRY_BASE_SECONDS", 0)
    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = FakeModel(error=TimeoutError("offline"))
    lane = ReactionAwarenessLane(db, model_name="m", model_client=model, lease_seconds=0)
    lane.register_target(profile="p", scope="s", timestamp=1, author="a", body="hello")
    event = {"event_id": "e", "profile": "p", "scope": "s", "target_timestamp": 1,
             "target_author": "a", "sender": "a", "emoji": "👍", "action": "added",
             "event_timestamp": 2, "owner_authorized": True, "chat_type": "dm"}
    assert lane.enqueue(event) == "queued"
    assert asyncio.run(lane.process_one("p", "s")) == "retry"
    assert asyncio.run(lane.process_one("p", "s")) == "failed"
    assert asyncio.run(lane.process_one("p", "s")) == "empty"
    assert len(model.calls) == 2
    assert lane.outbox_status("p", "s", "e") == "failed"


def test_expired_claim_cannot_commit_after_newer_claim_finishes(tmp_path):
    class ConcurrentModel:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await self.release.wait()
                return {"text": '{"note":"stale claim"}'}
            return {"text": '{"note":"new claim"}'}

    async def exercise(lane):
        older = asyncio.create_task(lane.process_one("p", "s"))
        await model.started.wait()
        assert await lane.process_one("p", "s") == "completed"
        model.release.set()
        assert await older == "stale_claim"

    db = sqlite3.connect(tmp_path / "lane.db", isolation_level=None)
    model = ConcurrentModel()
    lane = ReactionAwarenessLane(db, model_name="m", model_client=model, lease_seconds=0)
    lane.register_target(profile="p", scope="s", timestamp=1, author="a", body="hello")
    event = {"event_id": "e", "profile": "p", "scope": "s", "target_timestamp": 1,
             "target_author": "a", "sender": "a", "emoji": "👍", "action": "added",
             "event_timestamp": 2, "owner_authorized": True, "chat_type": "dm"}
    assert lane.enqueue(event) == "queued"
    asyncio.run(exercise(lane))
    assert lane.notes("p", "s")[0]["note"] == "new claim"
