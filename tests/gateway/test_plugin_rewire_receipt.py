"""Regression for #119502 — the ``reload-plugins`` answer attests handler wiring instead of counting
adapter objects.

A ``register_platform_handler`` factory that raises leaves that plugin's callbacks unwired, so the
receipt carries it as ``handler_wiring_failures`` (the caller then refuses to say "active now")
while ``adapters_rewired`` still counts the live adapters. The unwired factory is NOT recorded as
wired either, so the next re-wire retries it instead of skipping it forever.
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from pathlib import Path

from gateway.config import Platform, PlatformConfig
from gateway.run_plugin_rewire import GatewayPluginRewireMixin, reload_plugins_verb
from hermes_cli.plugins import get_plugin_manager
from plugins.platforms.telegram.adapter import TelegramAdapter

_FACTORY_ENV = "HERMES_TEST_REWIRE_FACTORY_FAILS"
_PLUGIN = "rewire_fail"


class _Runner(GatewayPluginRewireMixin):
    """Only what ``reload_plugins_verb`` reads for the launch profile."""

    def __init__(self, adapter):
        self.adapters = {Platform.TELEGRAM: adapter}


def _write_raising_plugin(home: Path) -> None:
    plugin = home / "plugins" / _PLUGIN
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(f"name: {_PLUGIN}\nversion: '0.1'\ndescription: t\n")
    (plugin / "__init__.py").write_text(textwrap.dedent('''
        import os

        def register(ctx):
            def factory(native, adapter):
                if os.environ.get("HERMES_TEST_REWIRE_FACTORY_FAILS") == "1":
                    raise RuntimeError("handler factory boom")
            ctx.register_platform_handler("telegram", factory)
    '''))
    (home / "config.yaml").write_text(f"plugins:\n  enabled: [{_PLUGIN}]\n")


def test_raising_handler_factory_is_reported_as_unwired_in_the_reload_answer(monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    _write_raising_plugin(home)
    monkeypatch.setenv(_FACTORY_ENV, "1")

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._wire_plugin_handlers(None)  # connect(): wired while the plugin did not exist yet
    runner = _Runner(adapter)

    async def scenario():
        runner._subscribe_plugin_rewire(get_plugin_manager())
        handler = reload_plugins_verb(runner, asyncio.get_running_loop())
        return await asyncio.to_thread(handler, {"home": str(home)})

    answer = asyncio.run(scenario())

    assert answer["reloaded"] is True
    assert answer["adapters_rewired"] == 1
    assert answer["handler_wiring_failures"] == [_PLUGIN]

    # Retryability: the factory that raised was never marked wired, so the next re-wire runs it again.
    monkeypatch.delenv(_FACTORY_ENV)
    adapter.rewire_plugin_handlers()
    assert adapter.plugin_handler_wiring_failures() == []
