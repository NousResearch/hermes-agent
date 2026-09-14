"""Dashboard /chat PTY must bind OAuth Abos from the query, never API keys."""

from pathlib import Path

import hermes_cli.main_tui_launch as tui_launch
import hermes_cli.web_server as ws
import hermes_cli.web_server_chat as chat


def _env(monkeypatch, **kwargs):
    monkeypatch.setattr(
        tui_launch, "_make_tui_argv",
        lambda *_a, **_k: (["node", "fake-tui.js"], Path("/tmp")),
    )
    monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_port", 9119, raising=False)
    _argv, _cwd, env = chat._resolve_chat_argv(**kwargs)
    assert env is not None
    return env


def test_openai_codex_oauth_abo_is_bound_without_an_api_key(monkeypatch):
    env = _env(monkeypatch, provider="openai-codex", model="gpt-5.4")
    assert env["HERMES_TUI_PROVIDER"] == "openai-codex"
    assert env["HERMES_INFERENCE_PROVIDER"] == "openai-codex"
    assert env["HERMES_MODEL"] == "gpt-5.4"
    assert env["HERMES_INFERENCE_MODEL"] == "gpt-5.4"


def test_api_key_openai_provider_is_ignored(monkeypatch):
    env = _env(monkeypatch, provider="openai", model="gpt-4o")
    assert env.get("HERMES_TUI_PROVIDER") != "openai"
    assert env.get("HERMES_MODEL") != "gpt-4o"


def test_chatgpt_mode_codex_selects_the_codex_model(monkeypatch):
    env = _env(monkeypatch, provider="openai-codex", chatgpt_mode="codex")
    assert env["HERMES_TUI_PROVIDER"] == "openai-codex"
    assert "terra" in env["HERMES_MODEL"]


def test_resume_does_not_override_the_stored_session_provider(monkeypatch):
    env = _env(
        monkeypatch, resume="20260913_120000_abcdef",
        provider="openai-codex", model="gpt-5.4",
    )
    assert env.get("HERMES_TUI_RESUME") == "20260913_120000_abcdef"
    assert env.get("HERMES_TUI_PROVIDER") != "openai-codex"
