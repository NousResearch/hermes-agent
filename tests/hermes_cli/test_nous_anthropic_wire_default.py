"""Nous wire policy is configuration input to the canonical invocation route."""

from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp
from providers.routing import InvocationRequest, resolve_invocation_route

PORTAL = "https://inference-api.nousresearch.com/v1"


def _cfg(tmp_path, body: str, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")


def _configured_mode(model: str) -> str:
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly() or {}
    wire = str(((cfg.get("nous") or {}).get("anthropic_wire")) or "chat")
    return resolve_invocation_route(InvocationRequest(
        provider="nous",
        model=model,
        base_url=PORTAL,
        route_options={"anthropic_wire": wire},
    )).api_mode


def _portal_creds(monkeypatch):
    monkeypatch.setattr(
        rp,
        "resolve_nous_runtime_credentials",
        lambda **kw: {
            "base_url": PORTAL,
            "api_key": "jwt",
            "source": "portal",
            "expires_at": None,
        },
    )
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "nous"})


def test_default_is_chat_for_anthropic_and_unchanged_for_everything_else(tmp_path, monkeypatch):
    _cfg(tmp_path, "model:\n  default: anthropic/claude-fable-5.1\n", monkeypatch)
    assert _configured_mode("anthropic/claude-fable-5.1") == "chat_completions"
    assert _configured_mode("openai/gpt-5.6-sol") == "chat_completions"


def test_native_opt_in_restores_the_messages_wire_for_anthropic_only(tmp_path, monkeypatch):
    _cfg(tmp_path, "nous:\n  anthropic_wire: native\n", monkeypatch)
    assert _configured_mode("anthropic/claude-fable-5.1") == "anthropic_messages"
    assert _configured_mode("openai/gpt-5.6-sol") == "chat_completions"


@pytest.mark.parametrize("raw", ["''", "CHAT", "messages", "true", "1"])
def test_anything_but_native_reads_as_chat(tmp_path, monkeypatch, raw):
    _cfg(tmp_path, f"nous:\n  anthropic_wire: {raw}\n", monkeypatch)
    assert _configured_mode("anthropic/claude-fable-5.1") == "chat_completions"


def test_runtime_resolution_hands_a_live_agent_the_selected_wire(tmp_path, monkeypatch):
    _portal_creds(monkeypatch)
    _cfg(tmp_path, "model:\n  provider: nous\n", monkeypatch)
    resolved = rp.resolve_runtime_provider(
        requested="nous", target_model="anthropic/claude-fable-5.1"
    )
    assert (resolved["api_mode"], resolved["base_url"]) == ("chat_completions", PORTAL)

    _cfg(tmp_path, "model:\n  provider: nous\nnous:\n  anthropic_wire: native\n", monkeypatch)
    resolved = rp.resolve_runtime_provider(
        requested="nous", target_model="anthropic/claude-fable-5.1"
    )
    assert resolved["api_mode"] == "anthropic_messages"
