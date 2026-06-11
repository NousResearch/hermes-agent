"""Regression tests for LSP opt-in defaults."""
from __future__ import annotations

import pytest

from hermes_cli.config import DEFAULT_CONFIG


def _create_service(monkeypatch, config):
    from agent.lsp.manager import LSPService

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    return LSPService.create_from_config()


@pytest.fixture
def lsp_service(monkeypatch):
    svc = None

    def create(config):
        nonlocal svc
        if svc is not None:
            svc.shutdown()
        svc = _create_service(monkeypatch, config)
        return svc

    yield create

    if svc is not None:
        svc.shutdown()


def test_default_config_keeps_lsp_opt_in():
    assert "lsp" in DEFAULT_CONFIG, "DEFAULT_CONFIG missing 'lsp' key"
    lsp_cfg = DEFAULT_CONFIG["lsp"]

    assert lsp_cfg["enabled"] is False
    assert lsp_cfg["install_strategy"] == "manual"


def test_service_factory_missing_lsp_keys_uses_opt_in_defaults(lsp_service):
    svc = lsp_service({})

    assert svc is not None
    assert svc.is_active() is False
    assert svc._install_strategy == "manual"


@pytest.mark.parametrize(
    ("config", "expected_active", "expected_strategy"),
    [
        ({"lsp": {"enabled": True}}, True, "manual"),
        ({"lsp": {"install_strategy": "auto"}}, False, "auto"),
    ],
)
def test_service_factory_partially_configured_lsp_uses_secure_defaults(
    lsp_service,
    config,
    expected_active,
    expected_strategy,
):
    svc = lsp_service(config)

    assert svc is not None
    assert svc.is_active() is expected_active
    assert svc._install_strategy == expected_strategy


def test_service_factory_preserves_explicit_auto_install_opt_in(lsp_service):
    svc = lsp_service({"lsp": {"enabled": True, "install_strategy": "auto"}})

    assert svc is not None
    assert svc.is_active() is True
    assert svc._install_strategy == "auto"
