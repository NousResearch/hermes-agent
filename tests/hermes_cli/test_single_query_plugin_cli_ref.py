"""``hermes chat -q``/``-Q`` runs give plugins the CLI reference like the interactive loop does.

Only ``HermesCLI.run()`` set ``PluginManager._cli_ref``, so a plugin tool dispatched from a one-shot
turn saw ``_cli_ref is None`` and ``PluginContext.dispatch_tool`` injected no ``parent_agent`` (#67597).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli
from hermes_cli.plugins import get_plugin_manager


@pytest.fixture(autouse=True)
def _one_shot_seams(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    monkeypatch.setattr(get_plugin_manager(), "_cli_ref", None)


def _stub(**extra):
    return SimpleNamespace(
        _single_query_mode=False, _claim_active_session=lambda *a, **k: True,
        console=SimpleNamespace(print=lambda *a, **k: None), _show_security_advisories=lambda: None,
        chat=lambda *a, **k: "response", _print_exit_summary=lambda **k: None,
        _last_turn_result={"failed": False}, **extra,
    )


def test_one_shot_turn_binds_the_cli_for_plugins():
    stub = _stub()
    with pytest.raises(SystemExit):
        cli._run_single_query_mode(stub, "do the thing", None, False, True)
    assert get_plugin_manager()._cli_ref is stub


def test_quiet_turn_binds_the_cli_for_plugins():
    stub = _stub(_ensure_runtime_credentials=lambda: False, _credentials_rate_limited=False,
                 session_id="s1", model="m")
    with pytest.raises(SystemExit):
        cli._run_single_query_mode(stub, "do the thing", None, True, True)
    assert get_plugin_manager()._cli_ref is stub
