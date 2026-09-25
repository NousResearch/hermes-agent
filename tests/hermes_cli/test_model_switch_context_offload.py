"""``/model`` context-length resolution must not run on the gateway event loop.

``resolve_display_context_length`` runs two blocking chains — the route
comparison in ``should_clear_context_pin`` and the provider probe ladder in
``get_model_context_length`` (blocking ``requests`` calls to Anthropic
``/v1/models``, Copilot, Nous, Codex, GMI, Ollama, models.dev and OpenRouter).

The gateway message path already offloads both (``get_model_context_length_async``,
``should_clear_context_pin_async``); the ``/model`` slash-command handlers called
the sync helper directly, freezing the loop for every user on every platform for
the duration of the probe ladder.
"""

import threading
import time

import pytest

import agent.model_metadata as model_meta_mod
from hermes_cli import model_switch

PROBE_SECONDS = 0.05

RESOLVE_ARGS = dict(
    model="claude-opus-4",
    provider="anthropic",
    base_url="",
    api_key="",
    custom_providers=None,
    config_context_length=None,
)


@pytest.fixture
def slow_probe(monkeypatch):
    """Stand in for one blocking provider probe inside the resolution chain."""
    calls = {}

    def _probe(model, **kwargs):
        calls["thread"] = threading.current_thread()
        time.sleep(PROBE_SECONDS)
        return 128000

    monkeypatch.setattr(model_meta_mod, "get_model_context_length", _probe)
    return calls


@pytest.mark.asyncio
async def test_async_variant_matches_sync(slow_probe):
    """The async wrapper resolves the same value as the sync helper."""
    sync_value = model_switch.resolve_display_context_length(**RESOLVE_ARGS)
    async_value = await model_switch.resolve_display_context_length_async(
        **RESOLVE_ARGS
    )
    assert async_value == sync_value == 128000


@pytest.mark.asyncio
async def test_resolution_runs_off_the_event_loop_thread(slow_probe):
    """The blocking chain must execute on a worker thread, not the loop thread."""
    loop_thread = threading.current_thread()
    await model_switch.resolve_display_context_length_async(**RESOLVE_ARGS)
    assert slow_probe["thread"] is not loop_thread
