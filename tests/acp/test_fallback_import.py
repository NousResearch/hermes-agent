"""Regression tests for ACP compatibility imports."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path


def test_repo_root_does_not_shadow_real_acp_package():
    repo_root = Path(__file__).resolve().parents[2]
    local_acp = (repo_root / "acp").resolve()

    spec = importlib.util.find_spec("acp")

    if spec is None or spec.origin is None:
        return
    assert local_acp not in Path(spec.origin).resolve().parents


def test_acp_compat_exposes_fallback_surface():
    from acp_adapter import acp_compat

    assert acp_compat.acp is not None
    assert acp_compat.RequestError.method_not_found("ping").code == -32601
    assert acp_compat.TextContentBlock(type="text", text="hello").text == "hello"


def test_acp_compat_uses_private_fallback_when_real_acp_is_unavailable(monkeypatch):
    class _BlockAcp(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "acp" or fullname.startswith("acp."):
                raise ImportError("blocked real acp")
            return None

    import acp_adapter

    for name in list(sys.modules):
        if name == "acp_adapter.acp_compat" or name == "acp" or name.startswith("acp."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delattr(acp_adapter, "acp_compat", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockAcp(), *sys.meta_path])

    compat = importlib.import_module("acp_adapter.acp_compat")

    assert compat.acp.__name__ == "_acp_fallback"
    assert compat.RequestError.method_not_found("ping").code == -32601
    assert compat.TextContentBlock(type="text", text="hello").text == "hello"
