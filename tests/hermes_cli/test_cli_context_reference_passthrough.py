"""The classic CLI leaves ``@`` references to the gateway, which expands them once against the
session's cwd; a client-side pass first expanded them twice (PLG-4)."""

from types import SimpleNamespace

import pytest


class _Staged(Exception):
    pass


def test_chat_submits_at_references_unexpanded(tmp_path, monkeypatch):
    from cli import HermesCLI

    (tmp_path / "example.py").write_text("CLIENT-SIDE-CONTENT", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cli = HermesCLI.__new__(HermesCLI)
    cli.model = "gpt-5.6-terra"
    cli.base_url = "https://chatgpt.com/backend-api/codex"
    cli.api_key = "token"
    cli.provider = "openai-codex"
    cli.agent = SimpleNamespace(_config_context_length=None)
    cli._active_agent_route_signature = "route"
    cli._secret_capture_callback = lambda *_args, **_kwargs: None
    cli._last_turn_interrupted = False
    cli._ensure_runtime_credentials = lambda: True
    cli._resolve_turn_agent_config = lambda _message: {
        "signature": "route", "model": cli.model, "runtime": None, "request_overrides": None}
    cli._init_agent = lambda **_kwargs: True
    staged = []

    def _stage(_agent, message):
        staged.append(message)
        raise _Staged

    cli._chat_stage_user_message = _stage
    with pytest.raises(_Staged):
        cli.chat("inspect @file:example.py")

    assert staged == ["inspect @file:example.py"]
