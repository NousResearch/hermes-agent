"""MATRIX_REACTIONS must honor shared truthy/falsy aliases (including 'off')."""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
    )


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", "OFF"])
def test_reactions_falsy_aliases_disable(monkeypatch, raw):
    monkeypatch.setenv("MATRIX_REACTIONS", raw)
    assert _make_adapter()._reactions_enabled is False


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE"])
def test_reactions_truthy_aliases_enable(monkeypatch, raw):
    monkeypatch.setenv("MATRIX_REACTIONS", raw)
    assert _make_adapter()._reactions_enabled is True


def test_reactions_defaults_on(monkeypatch):
    monkeypatch.delenv("MATRIX_REACTIONS", raising=False)
    assert _make_adapter()._reactions_enabled is True
