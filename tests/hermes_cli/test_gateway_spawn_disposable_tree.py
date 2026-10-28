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

    def popen(command, **kwargs):
        seen.append(kwargs["cwd"])
        return child

    monkeypatch.setattr(start.subprocess, "Popen", popen)
    assert start.spawn_unmanaged_gateway(tmp_path, deadline=time.monotonic() + 5) is child
    assert seen == [Path(start.__file__).resolve().parent.parent]
