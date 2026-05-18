from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def load_provider():
    from plugins.memory import load_memory_provider

    def _load():
        return load_memory_provider("memory-integration")

    return _load


@pytest.fixture
def fake_vault_adapter(monkeypatch, tmp_path):
    module = types.ModuleType("vault_adapter")
    vault_root = tmp_path / "vault"

    def resolve_vault_path(explicit_path=None, **kwargs):
        if explicit_path:
            return Path(explicit_path)
        return vault_root

    setattr(module, "resolve_vault_path", resolve_vault_path)
    monkeypatch.setitem(sys.modules, "vault_adapter", module)
    return vault_root
