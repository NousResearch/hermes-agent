"""A TUI/Desktop agent built on a credential-resolution fallback (primary AuthError -> fallback_providers
entry) must carry the same one-shot switch notice the messaging gateway surfaces (#74349), and the
private notice key must never reach the AIAgent constructor. Drives the real ``_make_agent``."""
from unittest.mock import patch

from hermes_cli.auth import AuthError


class _StubAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _build(monkeypatch, tmp_path, *, primary_fails: bool):
    from tui_gateway import server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    (tmp_path / "config.yaml").write_text("model:\n  default: gpt-5.6-sol\n  provider: openai-codex\n")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_fallback_model",
                        lambda: [{"provider": "anthropic", "model": "claude-sonnet-5", "api_key": "fb"}])

    def _resolve(**kwargs):
        if primary_fails and kwargs.get("requested") != "anthropic":
            raise AuthError("expired")
        return {"api_key": "k", "base_url": "https://example.invalid/v1",
                "provider": kwargs.get("requested") or "openai-codex", "api_mode": "chat_completions"}

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("run_agent.AIAgent", _StubAgent):
        return server._make_agent("sid", "key", context_cwd_is_launch_artifact=False)


def test_fallback_build_attaches_the_switch_notice_outside_agent_kwargs(monkeypatch, tmp_path):
    agent = _build(monkeypatch, tmp_path, primary_fails=True)
    assert (agent.kwargs["provider"], agent.kwargs["model"]) == ("anthropic", "claude-sonnet-5")
    assert "_fallback_notice" not in agent.kwargs
    notice = agent._pending_fallback_notice
    assert "openai-codex/gpt-5.6-sol" in notice and "anthropic/claude-sonnet-5" in notice


def test_primary_build_carries_no_notice(monkeypatch, tmp_path):
    agent = _build(monkeypatch, tmp_path, primary_fails=False)
    assert agent.kwargs["provider"] == "openai-codex"
    assert not hasattr(agent, "_pending_fallback_notice")


def test_fallback_build_survives_model_string_shorthand(monkeypatch, tmp_path):
    """``model: <id>`` (the string shorthand every other reader accepts) must not crash the fallback
    build while it names the primary in the notice."""
    from tui_gateway import server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    (tmp_path / "config.yaml").write_text("model: gpt-5.6-sol\n", encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_fallback_model",
                        lambda: [{"provider": "anthropic", "model": "claude-sonnet-5", "api_key": "fb"}])

    def _resolve(**kwargs):
        if kwargs.get("requested") != "anthropic":
            raise AuthError("expired")
        return {"api_key": "k", "base_url": "https://example.invalid/v1",
                "provider": "anthropic", "api_mode": "chat_completions"}

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("run_agent.AIAgent", _StubAgent):
        agent = server._make_agent("sid", "key", context_cwd_is_launch_artifact=False)
    assert (agent.kwargs["provider"], agent.kwargs["model"]) == ("anthropic", "claude-sonnet-5")
    assert "gpt-5.6-sol" in agent._pending_fallback_notice and "anthropic/claude-sonnet-5" in agent._pending_fallback_notice
