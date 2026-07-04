"""@context budget probe must honor custom_providers per-model context_length.

The ``@file`` reference expansion sizes its injection budget from
``get_model_context_length``.  For a slug-keyed custom-provider override
(LM Studio's ``publisher/slug`` ids) the per-model context_length must reach
that probe — otherwise the budget silently falls back to the default. Regression
residual of PR #18844 (a).
"""
from types import SimpleNamespace

import pytest

from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_context_expansion_uses_custom_provider_slug_budget(monkeypatch):
    import gateway.run as gateway_run

    config = {
        "model": {"default": "lmstudio/phi-4", "provider": "custom"},
        "custom_providers": [
            {
                "name": "lmstudio",
                "base_url": "http://localhost:1234/v1",
                "models": {"phi-4": {"context_length": 1_048_576}},
            }
        ],
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"provider": "custom", "base_url": "http://localhost:1234/v1", "api_key": "x"},
    )

    captured = {}

    async def fake_preprocess(message_text, *, cwd, context_length, allowed_root):
        captured["context_length"] = context_length
        return SimpleNamespace(blocked=False, expanded=False, message=message_text, warnings=[])

    monkeypatch.setattr(
        "agent.context_references.preprocess_context_references_async", fake_preprocess
    )

    runner = object.__new__(GatewayRunner)
    runner._model = "lmstudio/phi-4"
    runner._base_url = "http://localhost:1234/v1"
    runner.config = SimpleNamespace()
    runner.adapters = {}
    runner._session_key_for_source = lambda source: "agent:main:telegram:dm:1"

    event = MessageEvent(
        text="summarise @notes.md please",
        source=SessionSource(platform=None, chat_id="1", chat_type="dm", user_id="9"),
        message_id="1",
    )

    await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[]
    )

    # Budget reflects the slug-keyed 1M override (step-0b), not the 256K default.
    assert captured["context_length"] == 1_048_576
