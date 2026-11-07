"""The MiniMax OAuth picker catalog: the account's own model list, live.

Before this hook the picker rendered a hardcoded list, so a model MiniMax released after the
last catalog edit stayed invisible even though the OAuth account could serve it — the picker
asked nobody, it replayed a static list. The hook reads the account's ``/v1/models`` and
falls back to the curated list when no credential or no answer is available.
"""
from __future__ import annotations

import pytest


def _response(payload: dict):
    """A minimal stand-in for ``httpx.Response``."""

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return payload

    return _Response()


@pytest.fixture
def catalog():
    """The MiniMax OAuth profile's picker hook (discovery registers the plugin module)."""
    import model_tools  # importing the plugin discovery side effect
    from plugins.model_providers.minimax import _minimax_oauth_picker_catalog

    return _minimax_oauth_picker_catalog


def test_catalog_returns_the_accounts_model_ids(catalog, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_minimax.resolve_minimax_oauth_runtime_credentials",
        lambda: {"api_key": "tok", "base_url": "https://api.minimax.io/anthropic"},
    )
    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: _response(
            {"data": [{"id": "MiniMax-M3.1-Flash"}, {"id": "MiniMax-M3"}, {"id": "MiniMax-M3.1-Flash"}]}
        ),
    )

    ids = catalog()

    assert ids == ["MiniMax-M3.1-Flash", "MiniMax-M3"]  # order preserved, duplicates dropped


def test_catalog_returns_none_without_a_credential(catalog, monkeypatch):
    def _no_creds():
        raise RuntimeError("not logged in")

    monkeypatch.setattr(
        "hermes_cli.auth_minimax.resolve_minimax_oauth_runtime_credentials", _no_creds
    )

    assert catalog() is None


def test_catalog_returns_none_when_the_endpoint_is_unreachable(catalog, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_minimax.resolve_minimax_oauth_runtime_credentials",
        lambda: {"api_key": "tok", "base_url": "https://api.minimax.io/anthropic"},
    )

    def _boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("httpx.get", _boom)

    assert catalog() is None


def test_catalog_returns_none_for_an_empty_model_list(catalog, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_minimax.resolve_minimax_oauth_runtime_credentials",
        lambda: {"api_key": "tok", "base_url": "https://api.minimax.io/anthropic"},
    )
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: _response({"data": []}))

    assert catalog() is None
