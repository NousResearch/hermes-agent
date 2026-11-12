"""A module mapping that grows while ``_evict_modules`` walks it must not fail the eviction (refs #139134).

``_evict_modules`` runs at the start of every directory-plugin import, and other threads can add to
``sys.modules`` meanwhile; iterating the live mapping raises "dictionary changed size during
iteration". This reproduces that concurrent mutation deterministically: a private stand-in for
``sys.modules`` gains an entry from a second thread at a fixed point of the walk. It performs no real
import and does not reproduce gateway startup or concurrent profile preparation, so it does not by
itself show this race caused the failures reported in #139134.
"""

from __future__ import annotations

import threading
import types

from hermes_cli import plugins_loader


def test_evict_modules_survives_a_concurrent_mapping_insert(monkeypatch):
    modules: dict[str, types.ModuleType] = {}
    handed_off = threading.Event()

    def _insert_elsewhere() -> None:
        modules["race_probe_sdk.client"] = types.ModuleType("race_probe_sdk.client")

    class _SwitchPoint(str):
        """A key whose prefix test runs a thread that inserts into the mapping mid-walk, deterministically."""

        def startswith(self, *args):
            if not handed_off.is_set():
                handed_off.set()
                worker = threading.Thread(target=_insert_elsewhere)
                worker.start()
                worker.join()
            return str.startswith(self, *args)

    for name in ("hermes_plugins.discord_platform", "hermes_plugins.discord_platform.adapter"):
        modules[name] = types.ModuleType(name)
    modules[_SwitchPoint("unrelated")] = types.ModuleType("unrelated")
    modules["hermes_plugins.discord_platform.tools"] = types.ModuleType("hermes_plugins.discord_platform.tools")
    monkeypatch.setattr(plugins_loader, "sys", types.SimpleNamespace(modules=modules))

    plugins_loader._evict_modules("hermes_plugins.discord_platform")

    assert handed_off.is_set()
    assert sorted(modules) == ["race_probe_sdk.client", "unrelated"]
