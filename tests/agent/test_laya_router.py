import json
from types import SimpleNamespace

from agent import laya_router


def test_build_state_matches_jev_contract():
    agent = SimpleNamespace(provider="openai-codex", model="gpt-5.6-luna", platform="cli")
    state = laya_router.build_state(
        agent,
        [{"type": "text", "text": "Analyze this"}, {"type": "image_url", "url": "x"}],
    )
    assert state["has_multimodal_content"] is True
    assert set(state["route_policies"]) == set(laya_router.ROUTE_OPTIONS)
    assert len(state["input_sha256"]) == 64


def test_request_uses_same_typed_choice_schema(monkeypatch):
    captured = {}

    class FakeRouter:
        def predict(self, state, questions):
            captured["state"] = state
            captured["questions"] = questions
            return {"model": "laya-rl-agent", "answers": {"task_route": {"choice": "coding"}}}

    monkeypatch.setattr(laya_router, "_router", lambda settings: FakeRouter())
    result = laya_router._request({"user_message": "fix tests"}, {})
    question = captured["questions"]["task_route"]
    assert result["answers"]["task_route"]["choice"] == "coding"
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(laya_router.ROUTE_OPTIONS)


def test_shadow_route_disabled_by_default(monkeypatch):
    monkeypatch.setattr(laya_router, "_enabled", lambda: False)
    called = []
    monkeypatch.setattr(laya_router.threading, "Thread", lambda **kwargs: called.append(kwargs))
    laya_router.shadow_route(SimpleNamespace(platform="cli"), "hello", "task-1")
    assert called == []


def test_shadow_route_skips_subagents(monkeypatch):
    monkeypatch.setattr(laya_router, "_enabled", lambda: True)
    called = []
    monkeypatch.setattr(laya_router.threading, "Thread", lambda **kwargs: called.append(kwargs))
    laya_router.shadow_route(SimpleNamespace(platform="subagent"), "hello", "task-1")
    assert called == []


def test_record_outcome_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(laya_router, "_enabled", lambda: True)
    monkeypatch.setattr(laya_router, "_settings", lambda: {"outcome_log_path": str(tmp_path / "outcomes.jsonl")})
    laya_router.record_outcome(
        task_id="task-1", session_id="session-1", platform="cli",
        result={"completed": True}, duration_ms=1234, retry_count=2, tool_error_count=1,
    )
    record = json.loads((tmp_path / "outcomes.jsonl").read_text().strip())
    assert record["task_id"] == "task-1"
    assert record["outcome"] == "completed"
    assert record["duration_ms"] == 1234
