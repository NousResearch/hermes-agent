from __future__ import annotations

import model_tools


def test_one_shot_dispatch_blocks_unsupported_tools_before_handlers(monkeypatch) -> None:
    calls: list[str] = []

    def fake_dispatch(name, args, **kwargs):
        calls.append(name)
        return "ok"

    monkeypatch.setattr(model_tools.registry, "dispatch", fake_dispatch)
    with model_tools.becky_one_shot_dispatch_scope(
        {"mcp__todoist__create_task"}
    ):
        blocked = model_tools.handle_function_call("terminal", {"command": "echo"})
        first = model_tools.handle_function_call(
            "mcp__todoist__create_task", {"content": "Buy milk"}
        )
        second = model_tools.handle_function_call(
            "mcp__todoist__create_task", {"content": "Buy eggs"}
        )

    assert "one-shot" in blocked
    assert "one-shot" in first
    assert "one-shot" in second
    assert calls == []


def test_one_shot_dispatch_allows_one_reviewed_handler(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda name, args, **kwargs: calls.append(name) or "ok",
    )
    with model_tools.becky_one_shot_dispatch_scope(
        {"mcp__todoist__create_task"}
    ):
        assert model_tools.handle_function_call(
            "mcp__todoist__create_task", {"content": "Buy milk"}
        ) == "ok"
        assert "one-shot" in model_tools.handle_function_call(
            "mcp__todoist__create_task", {"content": "Buy eggs"}
        )
    assert calls == ["mcp__todoist__create_task"]
