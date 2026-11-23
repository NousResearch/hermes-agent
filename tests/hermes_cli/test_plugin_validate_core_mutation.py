"""``no core override`` also catches a plugin rewiring Hermes's private containers in place
(``get_plugin_manager()._hooks["x"].append(fn)``): the hook list is as much core as the function
an assignment would replace. A plugin mutating its own lists or calling a public core method is
not a rebind."""

from __future__ import annotations

from pathlib import Path

from hermes_cli.plugin_validate_core_override import core_override_findings


def _plugin(root: Path, init_py: str) -> Path:
    root.mkdir()
    (root / "__init__.py").write_text(init_py, encoding="utf-8")
    return root


def test_in_place_mutation_of_core_private_state_is_a_rebind(tmp_path):
    bad = _plugin(tmp_path / "bad", (
        "from hermes_cli.plugins import get_plugin_manager\n"
        "def _drop(registry):\n"
        "    registry._middleware.clear()\n"
        "def register(ctx):\n"
        "    get_plugin_manager()._hooks['pre_llm_call'].append(print)\n"
        "    mgr = get_plugin_manager()\n"
        "    mgr._middleware.remove(print)\n"
        "    mgr._hooks[:] = []\n"
        "    mgr._hooks.setdefault('x', []).insert(0, print)\n"
        "    _drop(mgr)\n"
    ))

    assert core_override_findings(bad) == [
        "get_plugin_manager()._hooks[...].append(...) (__init__.py:5)",
        "mgr._middleware.remove(...) (__init__.py:7)",
        "mgr._hooks[...] (__init__.py:8)",
        "mgr._hooks.setdefault(...) (__init__.py:9)",
        "_drop(mgr, ...) (__init__.py:10)",
    ]


def test_own_containers_and_public_core_calls_are_not_rebinds(tmp_path):
    good = _plugin(tmp_path / "good", (
        "from hermes_cli.config import load_config\n"
        "from hermes_cli.plugins import get_plugin_manager\n"
        "class Seen:\n"
        "    def __init__(self):\n"
        "        self._items = []\n"
        "    def add(self, x):\n"
        "        self._items.append(x)\n"
        "_CACHE = {}\n"
        "def register(ctx):\n"
        "    _CACHE.setdefault('k', []).append(1)\n"
        "    Seen().add(get_plugin_manager().list_plugins())\n"
        "    cfg = load_config()\n"
        "    cfg.setdefault('plugins', {}).update({'x': 1})\n"
    ))

    assert core_override_findings(good) == []
