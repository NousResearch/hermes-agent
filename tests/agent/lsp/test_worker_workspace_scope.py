"""A scratch worker must not index the enclosing runtime checkout."""

import pytest

from agent.lsp.servers import find_server_for_file
from agent.lsp.workspace import clear_cache, resolve_workspace_for_file


@pytest.fixture(autouse=True)
def _clear_workspace_cache():
    clear_cache()
    yield
    clear_cache()


def test_worker_scratch_root_bounds_project_marker_walk(tmp_path, monkeypatch):
    repo = tmp_path / "runtime"
    (repo / ".git").mkdir(parents=True)
    task = repo / "kanban" / "workspaces" / "task"
    task.mkdir(parents=True)
    # A neighbouring task's project marker must not widen the worker's root.
    (task.parent / "pyproject.toml").write_text("", encoding="utf-8")
    target = task / "probe.py"
    target.write_text("value: int = 1\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(task))
    monkeypatch.chdir(task)

    root, gated = resolve_workspace_for_file(str(target))

    assert gated
    assert root == str(task)
    assert find_server_for_file(str(target)).resolve_root(str(target), root) == str(task)


def test_nested_checkout_keeps_its_own_root_under_worker_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "runtime"
    (repo / ".git").mkdir(parents=True)
    task = repo / "task"
    nested = task / "candidate"
    nested.mkdir(parents=True)
    (nested / ".git").write_text("gitdir: unused\n", encoding="utf-8")
    target = nested / "probe.py"
    target.write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(task))
    monkeypatch.chdir(task)

    assert resolve_workspace_for_file(str(target)) == (str(nested), True)


def test_scratch_scope_retains_enclosing_repo_imports_without_trust(tmp_path, monkeypatch):
    import sys

    from agent.lsp.servers import ServerContext
    from agent.lsp.workspace import is_trusted_workspace

    repo = tmp_path / "runtime"
    (repo / ".git").mkdir(parents=True)
    task = repo / "task"
    task.mkdir()
    target = task / "probe.py"
    target.write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(task))
    monkeypatch.chdir(task)
    root, gated = resolve_workspace_for_file(str(target))
    assert gated
    context = ServerContext(workspace_root=root, install_strategy="off",
                            binary_overrides={"pyright": [sys.executable]})

    spec = find_server_for_file(str(target)).build_spawn(root, context)

    assert spec.workspace_root == str(task)
    assert spec.initialization_options["python"]["analysis"]["extraPaths"] == [str(repo)]
    assert not is_trusted_workspace(root, [], set())


@pytest.mark.parametrize("worker", [True, False])
def test_external_repository_is_not_narrowed_by_worker_workspace(tmp_path, monkeypatch, worker):
    task = tmp_path / "task"
    task.mkdir()
    external = tmp_path / "external"
    (external / ".git").mkdir(parents=True)
    target = external / "probe.py"
    target.write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    if worker:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(task))
    monkeypatch.chdir(task)

    assert resolve_workspace_for_file(str(target)) == (str(external), True)


def test_workspace_hint_alone_does_not_narrow_interactive_project(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    task = repo / "task"
    task.mkdir()
    target = task / "probe.py"
    target.write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(task))
    monkeypatch.chdir(task)

    assert resolve_workspace_for_file(str(target)) == (str(repo), True)
