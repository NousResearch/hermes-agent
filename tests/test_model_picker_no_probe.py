"""The /model picker must not live-probe saved custom endpoints on open.

Regression guard for "/model freezes for 1-2s before the picker appears".

``build_models_payload`` defaults to ``probe_custom_providers=True``, which
fires a synchronous HTTP ``/models`` request per saved custom OpenAI-compatible
endpoint.  ``_handle_model_switch`` runs on the prompt_toolkit UI thread, so the
default cost ~1-2s of dead console on this machine (3 custom endpoints) before
the picker could paint.

The GUI (``web_server``) and TUI (``tui_gateway``) pickers already pass
``probe_custom_providers=bool(refresh)``.  The CLI was the only entry point
still taking the default.  These tests pin that contract: a normal open must
not probe, and ``--refresh`` must keep the live path.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cli as cli_mod


def _capture_payload_calls(monkeypatch):
    """Record build_models_payload kwargs and return a fake provider list."""
    import hermes_cli.inventory as inventory

    calls = []

    def _fake_build_models_payload(ctx, **kwargs):
        calls.append(kwargs)
        return {
            "providers": [
                {"name": "deepseek", "slug": "deepseek",
                 "models": ["deepseek-chat"], "is_current": True,
                 "total_models": 1},
            ]
        }

    monkeypatch.setattr(inventory, "build_models_payload", _fake_build_models_payload)

    # load_picker_context is imported alongside; keep it cheap and inert.
    monkeypatch.setattr(
        inventory, "load_picker_context",
        lambda: SimpleNamespace(
            user_providers=None,
            custom_providers=None,
            with_overrides=lambda **kw: SimpleNamespace(
                user_providers=None, custom_providers=None),
        ),
    )
    return calls


def _make_cli(monkeypatch, calls_log):
    """CLI stub that exercises the real /model argument path."""
    obj = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    obj._app = None
    obj._modal_input_snapshot = None
    obj._model_picker_state = None
    obj._last_invalidate = 0.0
    obj._resize_recovery_pending = False
    obj.agent = None
    obj.model = "deepseek-chat"
    obj.provider = "deepseek"
    obj.requested_provider = "deepseek"
    obj.base_url = ""
    obj.api_key = ""
    obj.api_mode = ""
    obj.conversation_history = []
    obj._capture_modal_input_snapshot = lambda: None
    obj._paint_now = lambda: None
    obj._open_model_picker = lambda providers, *a, **kw: calls_log.append(providers)
    return obj


@pytest.mark.parametrize("cmd", ["/model", "/model --global"])
def test_picker_open_does_not_probe_custom_endpoints(monkeypatch, cmd):
    """A plain picker open must not hit the network for custom endpoints."""
    calls = _capture_payload_calls(monkeypatch)
    opened = []
    cli = _make_cli(monkeypatch, opened)

    cli._handle_model_switch(cmd)

    assert len(calls) == 1, "expected exactly one payload build"
    assert calls[0].get("probe_custom_providers") is False, (
        "the CLI picker fell back to build_models_payload's default "
        "(probe_custom_providers=True), which live-probes every custom "
        "endpoint on the UI thread and stalls /model for 1-2s"
    )
    assert opened, "picker was never opened"


def test_refresh_still_probes_live(monkeypatch):
    """`/model --refresh` must keep the explicit live-fetch path."""
    calls = _capture_payload_calls(monkeypatch)
    opened = []
    cli = _make_cli(monkeypatch, opened)

    cleared = []
    import hermes_cli.models as models_mod
    monkeypatch.setattr(
        models_mod, "clear_provider_models_cache",
        lambda: cleared.append(True), raising=False)

    cli._handle_model_switch("/model --refresh")

    assert cleared, "--refresh must still clear the model-id cache"
    # The picker open itself stays offline; refresh only busts the disk cache,
    # so subsequent cached_provider_model_ids() lookups re-fetch live.
    assert calls and calls[0].get("probe_custom_providers") is False
