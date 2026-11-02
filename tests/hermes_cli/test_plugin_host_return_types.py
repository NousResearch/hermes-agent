"""plugins.isolation: host — values a hosted provider returns reach core as the types its ABC promises.

Each test loads a real plugin into a real host subprocess and drives the core consumer, so a
return value that crosses the wire as a generic placeholder fails here the way it fails users.
"""

import os
import sys

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager


def _home_with_plugin(tmp_path, monkeypatch, name: str, body: str):
    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": [name], "isolation": "host"}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    return home


def _load(name: str) -> PluginManager:
    manager = PluginManager()
    manager.discover_and_load()
    assert manager._plugins[name].error is None
    assert not any(name in module for module in sys.modules)  # the plugin ran only in the host
    return manager


def _shutdown(manager: PluginManager) -> None:
    manager.unload()
    manager._plugin_host().shutdown()


SECRET_SOURCE_PLUGIN = '''
import os
from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource

class DummyVault(SecretSource):
    name = "hostvault"
    label = "Host vault"

    def fetch(self, cfg, home_path):
        if cfg.get("mode") == "locked":
            return FetchResult().fail("vault is locked", ErrorKind.AUTH_FAILED)
        return FetchResult(secrets={"HOSTED_DUMMY_TOKEN": f"dummy-from-{os.getpid()}"},
                           warnings=["dummy warning"], binary_path=home_path / "bin" / "vault")

def register(ctx):
    ctx.register_secret_source(DummyVault())
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_hosted_secret_source_secrets_are_applied(tmp_path, monkeypatch):
    from agent.secret_sources.base import ErrorKind
    from agent.secret_sources.registry import apply_all

    home = _home_with_plugin(tmp_path, monkeypatch, "hostvault", SECRET_SOURCE_PLUGIN)
    manager = _load("hostvault")
    try:
        target: dict[str, str] = {}
        report = apply_all({"hostvault": {"enabled": True}}, home, environ=target)
        (source_report,) = report.sources
        assert source_report.result.ok, source_report.result.error
        assert source_report.applied == ["HOSTED_DUMMY_TOKEN"]
        value = target["HOSTED_DUMMY_TOKEN"]
        assert value.startswith("dummy-from-") and value != f"dummy-from-{os.getpid()}"
        assert source_report.result.warnings == ["dummy warning"]
        assert source_report.result.binary_path == home / "bin" / "vault"

        # A failure keeps its kind, which drives stale-cache fallback and remediation text.
        locked: dict[str, str] = {}
        report = apply_all({"hostvault": {"enabled": True, "mode": "locked"}}, home, environ=locked)
        (source_report,) = report.sources
        assert source_report.result.error == "vault is locked"
        assert source_report.result.error_kind is ErrorKind.AUTH_FAILED
        assert locked == {}
    finally:
        _shutdown(manager)


CONTEXT_REFERENCE_PLUGIN = '''
from agent.context_references import ContextCompletionItem, ContextReferenceProvider

class Tickets(ContextReferenceProvider):
    prefix = "hostticket"
    description = "tickets"

    async def autocomplete(self, query, *, limit=10):
        return [ContextCompletionItem(text=f"{query}-1", meta="open"), ContextCompletionItem(text=f"{query}-2")]

    async def expand(self, target):
        return f"ticket {target}"

def register(ctx):
    ctx.register_context_reference(Tickets())
'''


@pytest.mark.platforms("any")
def test_hosted_context_reference_autocomplete_returns_items(tmp_path, monkeypatch):
    from tui_gateway.methods_complete import _plugin_reference_items

    _home_with_plugin(tmp_path, monkeypatch, "hostticket", CONTEXT_REFERENCE_PLUGIN)
    manager = _load("hostticket")
    try:
        assert _plugin_reference_items("hostticket", "bug") == [
            {"text": "@hostticket:bug-1", "display": "bug-1", "meta": "open"},
            {"text": "@hostticket:bug-2", "display": "bug-2", "meta": ""},
        ]
    finally:
        _shutdown(manager)


CONTEXT_ENGINE_PLUGIN = '''
from agent.context_engine import ContextEngine

class Ledger(ContextEngine):
    @property
    def name(self):
        return "hostledger"
    def update_from_response(self, usage):
        self.last_prompt_tokens = usage["prompt_tokens"]
    def should_compress(self, prompt_tokens=None):
        return False
    def compress(self, messages, current_tokens=None, **kw):
        self.compress_options = {"current_tokens": current_tokens, **kw}
        return messages[-1:]

def register(ctx):
    ctx.register_context_engine(Ledger())
'''


@pytest.mark.platforms("any")
def test_hosted_context_engine_clone_is_an_independent_engine(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent.context_engine import ContextEngine
    from agent.conversation_compression import _resolve_compress_call

    _home_with_plugin(tmp_path, monkeypatch, "hostledger", CONTEXT_ENGINE_PLUGIN)
    manager = _load("hostledger")
    try:
        shared = manager._context_engine
        clone = shared.clone_for_agent()  # what every AIAgent does with a plugin engine
        assert isinstance(clone, ContextEngine) and clone.name == "hostledger"
        clone.update_from_response({"prompt_tokens": 1234})
        assert clone.last_prompt_tokens == 1234
        assert shared.last_prompt_tokens == 0  # the agent's budget state is its own

        # Core passes the options by keyword; an override taking **kw must receive them as keywords.
        compress, options = _resolve_compress_call(
            SimpleNamespace(context_compressor=clone), approx_tokens=900, focus_topic="auth", force=True,
            memory_context="", bypass_cooldown=False)
        messages = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
        assert compress(messages, **options) == messages[-1:]
        assert clone.compress_options == {"current_tokens": 900, "focus_topic": "auth", "force": True}
    finally:
        _shutdown(manager)
