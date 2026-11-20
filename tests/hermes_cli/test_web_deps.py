"""``web_deps`` late binding resolves against a complete module.

Router helpers also run outside the dashboard app: a plugin's host process can call
``config_env.get_env_vars`` from worker threads without ever importing ``hermes_cli.web_server``.
Two such first calls both resolve ``late("_channel_managed_env_keys",
"hermes_cli.web_server_messaging")``; the second one got the module the first was still importing
and failed with "partially initialized module ... (most likely due to a circular import)".
"""

from __future__ import annotations

import importlib
import sys

from hermes_cli import web_deps
from tests.held_import import held_import

_TARGET = "hermes_test_late_target"


def test_late_waits_for_a_module_another_thread_is_importing():
    with held_import(_TARGET, answer=lambda: 42) as held:
        assert held.run_during_import(web_deps.late("answer", _TARGET)) == 42


def test_late_state_waits_for_a_module_another_thread_is_importing():
    with held_import(_TARGET, VALUE=42) as held:
        assert held.run_during_import(lambda: web_deps.LateState("VALUE", _TARGET) == 42) is True


def test_a_module_resolving_itself_during_its_own_import_sees_its_partial_self(tmp_path, monkeypatch):
    """On the importing thread itself (a circular import) a proxy resolves against the partial
    module, as a plain ``import`` would, instead of waiting for its own import."""
    name = "hermes_test_late_self"
    (tmp_path / f"{name}.py").write_text(
        "from hermes_cli import web_deps\n"
        "EARLY = 1\n"
        f"SEEN = web_deps.LateState('EARLY', {name!r}) == 1\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert importlib.import_module(name).SEEN is True
    finally:
        sys.modules.pop(name, None)
