"""``PluginContext.has_plugin`` must survive concurrent ``_plugins`` mutation.

``register()`` bodies run on worker threads (``run_with_load_deadline`` in
``hermes_cli/plugins_loader.py``) that insert into the manager's ``_plugins``
dict while sibling plugins — including advisory ``requires_plugins`` probes —
may call ``ctx.has_plugin()`` from their own worker. A bare lazy ``.items()``
iteration raises ``RuntimeError: dictionary changed size during iteration``
under that contention, which surfaces as a random
``Failed to load plugin '<name>-platform'`` warning and the loss of that
plugin for the process lifetime (observed in the wild across unrelated
platform plugins — homeassistant, email, google_chat — the victim is whoever
probes while a sibling finishes loading).

The race window is far too tight to hit reliably by timing, so the test
reproduces its exact shape deterministically: a duck-typed ``LoadedPlugin``
whose ``.enabled`` read inserts a sibling key into ``_plugins`` mid-walk. On
the lazy genexpr that insertion lands between ``next()`` calls of the live
view (RuntimeError); over a snapshot taken first, the insertion happens after
the copy and is harmless.
"""

from __future__ import annotations

from hermes_cli.plugins import (
    LoadedPlugin,
    PluginContext,
    PluginManager,
    PluginManifest,
)


def _manifest(name: str) -> PluginManifest:
    return PluginManifest(name=name, key=f"test/{name}", source="user")


def _loaded(name: str, enabled: bool = True) -> LoadedPlugin:
    return LoadedPlugin(manifest=_manifest(name), enabled=enabled)


class _InsertOnReadPlugin:
    """Duck-typed LoadedPlugin: first ``.enabled`` read mutates ``_plugins``.

    Mirrors a sibling worker completing its load (one dict insert) between two
    steps of ``has_plugin``'s iteration — the observable slice of the race.
    """

    def __init__(self, manager: PluginManager, name: str) -> None:
        self._manager = manager
        self._fired = False
        self.manifest = _manifest(name)

    @property
    def enabled(self) -> bool:
        if not self._fired:
            self._fired = True
            self._manager._plugins["test/late-sibling"] = _loaded("late-sibling")
        return True


def test_has_plugin_survives_dict_mutation_mid_iteration(monkeypatch) -> None:
    manager = PluginManager()
    monkeypatch.setattr(manager, "scope_key", "test-scope", raising=False)
    ctx = PluginContext(_manifest("prober"), manager)

    manager._plugins["test/loads-while-probed"] = _InsertOnReadPlugin(  # type: ignore[assignment]
        manager, "loads-while-probed"
    )
    manager._plugins["test/settled"] = _loaded("settled")

    # Must not raise: the mid-walk insert emulates a sibling worker finishing
    # its load while this probe iterates.
    assert ctx.has_plugin("settled") is True
    assert "test/late-sibling" in manager._plugins
    assert ctx.has_plugin("late-sibling") is True
    assert ctx.has_plugin("never-registered") is False
