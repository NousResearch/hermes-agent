"""FEISHU_REACTIONS must honor shared truthy/falsy aliases (including 'off')."""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.fixture()
def adapter():
    return FeishuAdapter(PlatformConfig())


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", "OFF"])
def test_reactions_falsy_aliases_disable(adapter, monkeypatch, raw):
    monkeypatch.setenv("FEISHU_REACTIONS", raw)
    assert adapter._reactions_enabled() is False


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE"])
def test_reactions_truthy_aliases_enable(adapter, monkeypatch, raw):
    monkeypatch.setenv("FEISHU_REACTIONS", raw)
    assert adapter._reactions_enabled() is True


def test_reactions_defaults_on(adapter, monkeypatch):
    monkeypatch.delenv("FEISHU_REACTIONS", raising=False)
    assert adapter._reactions_enabled() is True
