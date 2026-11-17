"""``/model kimi-k3.1-pro`` on a MiniMax session is refused, not pinned to a model MiniMax cannot serve.

MiniMax has no ``/models`` endpoint and no catalog lists ``kimi-k3.1-pro``, so detection finds no
better provider and the MiniMax validator used to soft-accept the id. Every turn then 400'd with
"unknown model" while the session kept reporting a successful switch to MiniMax.
"""

from __future__ import annotations

import pytest

from hermes_cli.model_switch import switch_model

_MINIMAX_SESSION = dict(
    current_provider="minimax",
    current_model="MiniMax-M3",
    current_base_url="https://api.minimax.io/anthropic",
)


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")


def test_other_vendor_id_without_provider_is_refused_and_names_the_fix():
    result = switch_model("kimi-k3.1-pro", **_MINIMAX_SESSION)

    assert result.success is False
    assert "not accepted" in result.error_message
    assert "/model kimi-k3.1-pro --provider kimi-coding" in result.error_message


def test_minimax_family_id_missing_from_the_catalog_still_switches():
    result = switch_model("MiniMax-M3.1-Flash", **_MINIMAX_SESSION)

    assert result.success is True
    assert result.target_provider == "minimax"
    assert "could not verify" in result.warning_message or "not found in the MiniMax catalog" in result.warning_message
