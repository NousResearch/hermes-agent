from types import SimpleNamespace

import agent.title_generator as title_generator


def test_derive_title_strips_leading_workspace_context():
    message = (
        '<workspace_context cwd="/private/project" branch="secret" />\n'
        "Fix login button on mobile"
    )

    assert title_generator.derive_title(message) == "Fix login button on mobile"


def test_generate_title_sends_only_request_after_workspace_context(monkeypatch):
    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"title": "Fix mobile login button"}'
                    )
                )
            ]
        )

    monkeypatch.setattr(title_generator, "_auto_title_enabled", lambda: True)
    monkeypatch.setattr(title_generator, "_title_language", lambda: "")
    monkeypatch.setattr(title_generator, "call_llm", fake_call_llm)

    message = (
        '<workspace_context cwd="/private/project" branch="secret" />\n'
        "Fix login button on mobile"
    )

    assert title_generator.generate_title(message) == "Fix mobile login button"
    assert captured["messages"][1]["content"] == "Fix login button on mobile"
    assert "workspace_context" not in captured["messages"][1]["content"]
    assert "/private/project" not in captured["messages"][1]["content"]


def test_workspace_context_is_removed_before_skill_summarization(monkeypatch):
    from agent import skill_commands

    seen = {}

    def fake_describe(message):
        seen["message"] = message
        return "/work — fix the title leak"

    monkeypatch.setattr(skill_commands, "describe_skill_invocation", fake_describe)
    message = '<workspace_context cwd="/private/project" />\n/skill expanded scaffold'

    assert title_generator.derive_title(message) == "/work — fix the title leak"
    assert seen["message"] == "/skill expanded scaffold"


def test_workspace_context_before_machine_marker_is_not_titleable():
    message = (
        '<workspace_context cwd="/private/project" />\n'
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted"
    )

    assert title_generator.is_titleable_user_message(message) is False


def test_workspace_context_only_is_not_titleable():
    assert (
        title_generator.is_titleable_user_message(
            '<workspace_context cwd="/private/project" />'
        )
        is False
    )
