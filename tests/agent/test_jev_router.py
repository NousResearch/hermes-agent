import json
from types import SimpleNamespace

from agent import jev_router


def test_build_state_marks_multimodal_and_includes_route_policy():
    agent = SimpleNamespace(provider="openai-codex", model="gpt-5.6-luna", platform="cli")
    state = jev_router.build_state(
        agent,
        [
            {"type": "text", "text": "Analyze this"},
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
        ],
    )

    assert state["has_multimodal_content"] is True
    assert state["route_policies"]["coding"]["fallback"] == "zai/glm-5.3"
    assert state["route_policies"]["multimodal"]["primary"] == "kimi-k3"


def test_request_uses_typed_choice_and_all_routes(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"answers":{"task_route":{"choice":"coding","confidence":0.9}}}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(jev_router.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(jev_router, "_api_key", lambda: "test-key")

    result = jev_router._request({"user_message": "fix tests"}, {"timeout_seconds": 2})
    payload = json.loads(captured["request"].data.decode())

    assert result["answers"]["task_route"]["choice"] == "coding"
    assert captured["timeout"] == 2.0
    assert payload["model"] == "jev-latest"
    question = payload["questions"]["task_route"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(jev_router.ROUTE_OPTIONS)
    assert captured["request"].headers["Authorization"] == "Bearer test-key"


def test_shadow_route_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(jev_router, "_enabled", lambda: False)
    called = []
    monkeypatch.setattr(jev_router.threading, "Thread", lambda **kwargs: called.append(kwargs))

    jev_router.shadow_route(SimpleNamespace(platform="cli"), "hello", "task-1")

    assert called == []


def test_shadow_route_skips_subagents(monkeypatch):
    monkeypatch.setattr(jev_router, "_enabled", lambda: True)
    called = []
    monkeypatch.setattr(jev_router.threading, "Thread", lambda **kwargs: called.append(kwargs))

    jev_router.shadow_route(SimpleNamespace(platform="subagent"), "hello", "task-1")

    assert called == []


def test_build_state_includes_non_secret_input_metadata():
    agent = SimpleNamespace(provider="openai-codex", model="gpt-5.6-luna", platform="cli")

    state = jev_router.build_state(agent, "hello world")

    assert state["input_chars"] == 11
    assert len(state["input_sha256"]) == 64
    assert "hello world" not in state["input_sha256"]
    assert state["has_multimodal_content"] is False


def test_write_outcome_records_task_result(tmp_path, monkeypatch):
    monkeypatch.setattr(jev_router, "_enabled", lambda: True)
    monkeypatch.setattr(
        jev_router,
        "_settings",
        lambda: {"outcome_log_path": str(tmp_path / "outcomes.jsonl")},
    )

    jev_router.record_outcome(
        task_id="task-1",
        session_id="session-1",
        platform="cli",
        result={"completed": True, "failed": False},
        duration_ms=1234,
        retry_count=2,
        tool_error_count=1,
    )

    record = json.loads((tmp_path / "outcomes.jsonl").read_text().strip())
    assert record["task_id"] == "task-1"
    assert record["outcome"] == "completed"
    assert record["duration_ms"] == 1234
    assert record["retry_count"] == 2
    assert record["tool_error_count"] == 1
