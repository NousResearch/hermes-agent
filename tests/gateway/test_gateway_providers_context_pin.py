"""Gateway must keep model.context_length when the URL lives in providers.<name>.

Issue #107606: hermes setup / model-switch writes the proxy URL under
``providers.<name>`` and leaves ``model.base_url`` empty. Gateway status
and hygiene used the raw empty ``model.base_url`` as the configured route,
so ``should_clear_context_pin`` compared '' vs the runtime URL and dropped
the pin. Agent startup already resolves ``providers.<name>.base_url`` via
``_configured_default_base_url`` and keeps the same pin.

No live endpoint: runtime is stubbed to the resolved custom route.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT

PINNED_CONTEXT = 1_048_576
PROXY_URL = "https://proxy.example.com/v1"
OTHER_URL = "https://other.example.com/v1"

DETECTION_CONFIG = {
    "model": {
        "default": "my-model",
        "provider": "my-proxy",
        "base_url": "",
        "context_length": PINNED_CONTEXT,
    },
    "providers": {
        "my-proxy": {
            "base_url": PROXY_URL,
            "key_env": "MY_PROXY_KEY",
        }
    },
}

RUNTIME = {
    "provider": "custom",
    "base_url": PROXY_URL,
    "api_key": "test-key",
}


def _write_config(home: Path, data: dict) -> None:
    (home / "config.yaml").write_text(yaml.dump(data), encoding="utf-8")


def _install_home(monkeypatch, home: Path, data: dict) -> None:
    from gateway import run as gateway_run

    home.mkdir(parents=True, exist_ok=True)
    _write_config(home, data)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)


def _no_network_context_length():
    """Keep step-0 pin / fallback only — never probe the fixture proxy URL."""

    def _impl(
        model,
        base_url="",
        api_key="",
        config_context_length=None,
        provider="",
        custom_providers=None,
    ):
        if isinstance(config_context_length, int) and config_context_length > 0:
            return config_context_length
        return DEFAULT_FALLBACK_CONTEXT

    return patch("agent.model_metadata.get_model_context_length", side_effect=_impl)


@pytest.fixture
def detection_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    _install_home(monkeypatch, home, DETECTION_CONFIG)
    return home


def test_resolve_gateway_model_context_keeps_providers_block_pin(detection_home):
    from gateway.run import _resolve_gateway_model_context

    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(RUNTIME)), _no_network_context_length():
        ctx = _resolve_gateway_model_context()

    assert ctx.context_length == PINNED_CONTEXT
    assert ctx.context_source == "config"


def test_resolve_gateway_model_context_clears_pin_on_model_mismatch(detection_home):
    from gateway.run import _resolve_gateway_model_context

    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(RUNTIME)), _no_network_context_length():
        ctx = _resolve_gateway_model_context("other-model")

    assert ctx.context_length != PINNED_CONTEXT
    assert ctx.context_source != "config"


def test_resolve_gateway_model_context_clears_pin_on_explicit_base_url_mismatch(tmp_path, monkeypatch):
    from gateway.run import _resolve_gateway_model_context

    data = {
        "model": {
            "default": "my-model",
            "provider": "my-proxy",
            "base_url": OTHER_URL,
            "context_length": PINNED_CONTEXT,
        },
        "providers": {
            "my-proxy": {
                "base_url": PROXY_URL,
                "key_env": "MY_PROXY_KEY",
            }
        },
    }
    _install_home(monkeypatch, tmp_path / "hermes-home", data)

    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(RUNTIME)), _no_network_context_length():
        ctx = _resolve_gateway_model_context()

    assert ctx.context_length != PINNED_CONTEXT
    assert ctx.context_source != "config"


@pytest.mark.asyncio
async def test_hygiene_keeps_providers_block_pin(detection_home):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._resolve_session_agent_runtime = lambda **_kw: ("my-model", dict(RUNTIME))

    hs = await runner._hmwa_hygiene_settings(source=None, session_key=None)

    assert hs.config_context_length == PINNED_CONTEXT


@pytest.mark.asyncio
async def test_hygiene_clears_pin_on_model_mismatch(detection_home):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._resolve_session_agent_runtime = lambda **_kw: ("other-model", dict(RUNTIME))

    hs = await runner._hmwa_hygiene_settings(source=None, session_key=None)

    assert hs.config_context_length != PINNED_CONTEXT


def test_agent_startup_still_keeps_providers_block_pin():
    """Agent already resolves providers.<name>; do not regress that path."""
    from agent.agent_init import _scope_context_length_to_default_runtime

    agent = SimpleNamespace(model="my-model", provider="custom", base_url=PROXY_URL)
    kept = _scope_context_length_to_default_runtime(
        agent, DETECTION_CONFIG, DETECTION_CONFIG["model"], [], PINNED_CONTEXT, PROXY_URL,
    )
    assert kept == PINNED_CONTEXT


def test_disabled_provider_block_does_not_keep_pin(tmp_path, monkeypatch):
    from gateway.run import _resolve_gateway_model_context

    data = {
        "model": {
            "default": "my-model",
            "provider": "my-proxy",
            "base_url": "",
            "context_length": PINNED_CONTEXT,
        },
        "providers": {
            "my-proxy": {
                "base_url": PROXY_URL,
                "key_env": "MY_PROXY_KEY",
                "enabled": False,
            }
        },
    }
    _install_home(monkeypatch, tmp_path / "hermes-home", data)

    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(RUNTIME)), _no_network_context_length():
        ctx = _resolve_gateway_model_context()

    assert ctx.context_length != PINNED_CONTEXT
    assert ctx.context_source != "config"
