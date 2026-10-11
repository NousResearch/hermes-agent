"""Bundled provider plugins (e.g. solstice's transport) import agent.gemini_native_adapter and
agent.bounded_response at plugin-load time. Those modules must not hard-require httpx: it is not a
base dependency and its absence used to abort plugin discovery with "No module named 'httpx'".
They share the lazy httpx proxy from hermes_cli.auth_constants instead."""
import importlib
import sys


def _block_httpx(monkeypatch):
    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "httpx" or name.startswith("httpx."):
                raise ModuleNotFoundError("No module named 'httpx'")
            return None

    monkeypatch.setattr(sys, "meta_path", [Blocker(), *list(sys.meta_path)])


def test_adapter_chain_imports_without_httpx(monkeypatch):
    _block_httpx(monkeypatch)
    for mod in ("agent.bounded_response", "agent.gemini_native_adapter", "hermes_cli.auth_constants"):
        sys.modules.pop(mod, None)
    for mod in ("agent.bounded_response", "agent.gemini_native_adapter"):
        importlib.import_module(mod)  # must not raise
