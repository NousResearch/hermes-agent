"""Manual title repair has the same provider-output fence as live titling."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_cli.sessions_cmd import _cmd_retitle_skills


def test_retitle_skills_fences_provider_title_before_print_and_persist(monkeypatch, capsys):
    db = MagicMock()
    db.list_skill_scaffolded_sessions.return_value = [
        {"id": "session-1", "content": "expanded skill", "title": "Old title"}
    ]
    monkeypatch.setattr("agent.skill_commands.describe_skill_invocation", lambda _text: "fix the title")
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='{"title":"Public<memory-context>PRIVATE_RETITLE</memory-context> title"}'
    ))])
    monkeypatch.setattr("agent.title_generator.call_llm", lambda **_kwargs: response)
    _cmd_retitle_skills(db, SimpleNamespace(limit=1, apply=True))
    assert "PRIVATE_RETITLE" not in capsys.readouterr().out
    db.set_session_title.assert_called_once_with("session-1", "Public title")
