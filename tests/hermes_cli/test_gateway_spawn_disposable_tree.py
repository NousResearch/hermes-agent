"""A daemon must never be spawned from a disposable (scratch or worktree) code tree.

Regression for t_962fc9a1: four `gateway run --idle-exit` orphans ran from a cto scratch venv for
hours, each with its own kanban dispatcher on the ivs board.
"""
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("path, disposable", [
    ("/Users/x/.hermes/profiles/cto/cache/scratch/t_c7ce02ae/base", True),
    ("/Users/x/.hermes/kanban/boards/ivs/workspaces/t_1/.worktrees/t_1", True),
    ("/Users/x/repo/.worktrees/feature", True),
    ("/Users/x/.hermes/hermes-agent", False),
    ("/Users/x/.hermes/profiles/cto", False),
    ("/Users/x/scratch-notes/cache", False),
])
def test_disposable_tree_predicate(path, disposable):
    from hermes_cli import gateway_runtime_start as start

    assert start._disposable_tree(Path(path)) is disposable


def test_spawn_refused_when_code_root_is_a_scratch_checkout(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_start as start
    from hermes_cli.gateway_runtime_service import RuntimeStartError

    scratch_module = (tmp_path / "profiles" / "cto" / "cache" / "scratch" / "t_c7ce02ae"
                      / "base" / "hermes_cli" / "gateway_runtime_start.py")
    monkeypatch.setattr(start, "__file__", str(scratch_module))
    launched = []
    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: launched.append(a))

    with pytest.raises(RuntimeStartError) as refused:
        start.spawn_unmanaged_gateway(tmp_path / "home", deadline=time.monotonic() + 5)

    assert refused.value.reason == "foreign_code_root"
    assert launched == []


def test_spawn_refused_when_profile_home_is_a_scratch_directory(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_start as start
    from hermes_cli.gateway_runtime_service import RuntimeStartError

    launched = []
    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: launched.append(a))
    home = tmp_path / "profiles" / "cto" / "cache" / "scratch" / "t_1"

    with pytest.raises(RuntimeStartError) as refused:
        start.spawn_unmanaged_gateway(home, deadline=time.monotonic() + 5)

    assert refused.value.reason == "foreign_code_root"
    assert launched == []


def test_canonical_code_root_still_spawns(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import gateway_runtime_start as start

    child = SimpleNamespace(pid=321)
    seen = []
    # The test runner's own interpreter may live under a scratch venv: pin the canonical one.
    monkeypatch.setattr(start.sys, "executable", str(Path.home() / ".hermes/hermes-agent/venv/bin/python3"))
    # On darwin the flat Popen is only reached when start_gateway_job declines (returns None);
    # otherwise launchctl probes via Popen-free subprocess machinery would hit this fake first.
    # Pinning it to None keeps the spawn on the exact flat-Popen path this test asserts.
    monkeypatch.setattr(start, "start_gateway_job", lambda *a, **k: None)

    def popen(command, **kwargs):
        seen.append(kwargs["cwd"])
        return child

    monkeypatch.setattr(start.subprocess, "Popen", popen)
    assert start.spawn_unmanaged_gateway(tmp_path, deadline=time.monotonic() + 5) is child
    assert seen == [Path(start.__file__).resolve().parent.parent]


def test_spawn_refused_when_interpreter_is_a_scratch_venv(tmp_path, monkeypatch):
    # The 27/10 orphans ran a scratch venv's python: the root and home are canonical here, only the
    # interpreter path is disposable, and the guard must still refuse before any Popen.
    from hermes_cli import gateway_runtime_start as start
    from hermes_cli.gateway_runtime_service import RuntimeStartError

    venv_python = tmp_path / "profiles" / "cto" / "cache" / "scratch" / "t_c7ce02ae" / "venv" / "bin" / "python"
    monkeypatch.setattr(start.sys, "executable", str(venv_python))
    launched = []
    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: launched.append(a))

    with pytest.raises(RuntimeStartError) as refused:
        start.spawn_unmanaged_gateway(tmp_path / "home", deadline=time.monotonic() + 5)

    assert refused.value.reason == "foreign_code_root"
    assert launched == []


def test_ensure_turns_a_scratch_spawn_refusal_into_a_verdict(tmp_path, monkeypatch):
    # The refusal is raised inside ensure_gateway_runtime's own try: callers (dispatcher, cron, TUI)
    # get a GatewayDiscovery with the reason, never an exception and never a retry loop.
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli import gateway_runtime_service as service
    from hermes_cli import gateway_runtime_start as start
    from hermes_cli.gateway_runtime import GatewayDiscovery

    home = tmp_path / "profiles" / "cto" / "cache" / "scratch" / "t_1"
    home.mkdir(parents=True)
    monkeypatch.setattr(runtime, "_update_fenced", lambda *a, **k: False)
    monkeypatch.setattr(runtime, "_overlay_owner_home", lambda h: h)
    monkeypatch.setattr(runtime, "_observe_replacing_stale",
                        lambda *a, **k: GatewayDiscovery("absent"))
    monkeypatch.setattr(runtime, "_starting_multiplexer_root", lambda h: None)
    monkeypatch.setattr(runtime, "discover_gateway_endpoint",
                        lambda *a, **k: GatewayDiscovery("absent"))
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned from scratch"))
    from hermes_cli import gateway_runtime_multiplex as mux
    monkeypatch.setattr(mux, "multiplexer_serves_home", lambda h: None)
    monkeypatch.setattr(mux, "implied_host_root", lambda h: None)

    verdict = runtime.ensure_gateway_runtime(home, timeout=5)

    assert verdict.reason_code == "foreign_code_root"
