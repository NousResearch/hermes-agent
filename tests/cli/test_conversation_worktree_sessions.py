"""CLI ownership tests for durable conversation worktrees."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import cli as cli_module
import hermes_state


@dataclass(frozen=True)
class _Binding:
    root_session_id: str
    path: Path
    branch: str
    base_commit: str
    repo_common_dir: Path


class _SessionDB:
    def __init__(
        self,
        *,
        roots: dict[str, str] | None = None,
        session_cwds: dict[str, str] | None = None,
    ) -> None:
        self.roots = roots or {}
        self.session_cwds = session_cwds or {}
        self.ended: list[str] = []
        self.created: list[str] = []

    def get_conversation_root(self, session_id: str) -> str:
        return self.roots.get(session_id, session_id)

    def end_session(self, session_id: str, _reason: str) -> None:
        self.ended.append(session_id)

    def delete_session_if_empty(self, _session_id: str) -> bool:
        return False

    def create_session(self, **kwargs) -> None:
        self.created.append(kwargs["session_id"])

    def get_session(self, session_id: str) -> dict:
        return {"id": session_id, "cwd": self.session_cwds.get(session_id, "")}

    def close(self) -> None:
        return None


class _Manager:
    def __init__(self, worktree_root: Path) -> None:
        self.worktree_root = worktree_root
        self.worktree_root.mkdir(parents=True)
        self.bound_roots: list[str] = []
        self.resolved_roots: list[str] = []
        self.events: list[str] = []
        self.fail_next = False

    def _binding(self, root_session_id: str) -> _Binding:
        path = self.worktree_root / root_session_id
        path.mkdir(parents=True, exist_ok=True)
        return _Binding(
            root_session_id=root_session_id,
            path=path,
            branch=f"hermes/session/{root_session_id}",
            base_commit="a" * 40,
            repo_common_dir=self.worktree_root,
        )

    def bind_new_root_session(self, root_session_id: str, *, conversation_kind: str):
        assert conversation_kind == "interactive"
        self.events.append("bind")
        if self.fail_next:
            raise RuntimeError("bootstrap failed")
        self.bound_roots.append(root_session_id)
        return self._binding(root_session_id)

    def resolve_existing_session(self, root_session_id: str):
        self.events.append("resolve")
        self.resolved_roots.append(root_session_id)
        return self._binding(root_session_id)


@pytest.fixture
def manager(tmp_path) -> _Manager:
    return _Manager(tmp_path / "worktrees")


@pytest.fixture(autouse=True)
def _restore_process_cwd():
    from agent.runtime_cwd import clear_session_cwd

    original = cli_module.os.getcwd()
    clear_session_cwd()
    try:
        yield
    finally:
        cli_module.os.chdir(original)
        clear_session_cwd()


def _enabled_config() -> dict:
    config = deepcopy(cli_module.CLI_CONFIG)
    config["conversation_worktree"] = {
        "enabled": True,
        "source_worktree": "/repo/stable",
        "worktree_root": "/repo/.worktrees",
        "retain_until_explicit_cleanup": True,
    }
    return config


def _build_cli(monkeypatch, manager: _Manager, db: _SessionDB | None = None, **kwargs):
    db = db or _SessionDB()
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(cli_module, "CLI_CONFIG", _enabled_config())
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
    monkeypatch.setattr(cli_module, "_run_state_db_auto_maintenance", lambda _db: None)
    monkeypatch.setattr(cli_module, "_run_checkpoint_auto_maintenance", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_build_cli_conversation_worktree_manager",
        lambda _config, _db: manager,
    )
    return cli_module.HermesCLI(compact=True, **kwargs), db


def test_enabled_conversation_policy_binds_initial_cli_root(monkeypatch, manager):
    cli, _db = _build_cli(monkeypatch, manager)

    assert cli.working_directory == str(manager.worktree_root / cli.session_id)
    assert manager.bound_roots == [cli.session_id]
    assert cli.agent is None
    assert "certified Git worktree" in cli.system_prompt


def test_cli_resume_reuses_durable_root_binding(monkeypatch, manager):
    db = _SessionDB(roots={"compressed-tip": "durable-root"})

    cli, _db = _build_cli(monkeypatch, manager, db, resume="compressed-tip")

    assert manager.bound_roots == []
    assert manager.resolved_roots == ["durable-root"]
    assert cli.working_directory == str(manager.worktree_root / "durable-root")
    assert cli.session_id == "compressed-tip"


def test_cli_branch_allocates_and_enters_a_distinct_worktree(monkeypatch, manager):
    """Falling back to lineage-root resolution would keep /branch in the parent tree."""
    cli, _db = _build_cli(monkeypatch, manager)
    parent_session_id = cli.session_id
    parent_cwd = cli.working_directory
    cli.conversation_history = [{"role": "user", "content": "investigate this"}]

    cli._handle_branch_command("/branch isolated approach")

    assert cli.session_id != parent_session_id
    assert manager.bound_roots == [parent_session_id, cli.session_id]
    assert cli.working_directory == str(manager.worktree_root / cli.session_id)
    assert cli.working_directory != parent_cwd


def test_cli_resume_fails_closed_when_durable_binding_is_missing(monkeypatch, manager):
    manager.resolve_existing_session = MagicMock(return_value=None)

    with pytest.raises(RuntimeError, match="no ready conversation worktree"):
        _build_cli(monkeypatch, manager, _SessionDB(), resume="missing")


def test_managed_resume_cannot_restore_the_stable_source_cwd(monkeypatch, manager):
    cli, _db = _build_cli(monkeypatch, manager)
    managed = cli.working_directory

    cli._restore_session_cwd({"cwd": "/repo/stable"}, quiet=True)

    assert cli.working_directory == managed
    assert cli_module.os.environ["TERMINAL_CWD"] == managed


def test_managed_resume_retargets_to_the_selected_root_binding(monkeypatch, manager):
    db = _SessionDB(roots={"selected-tip": "selected-root"})
    cli, _db = _build_cli(monkeypatch, manager, db)
    cli.session_id = "selected-tip"

    cli._restore_session_cwd({"cwd": "/repo/stable"}, quiet=True)

    assert manager.resolved_roots == ["selected-root"]
    assert cli.working_directory == str(manager.worktree_root / "selected-root")


def test_cmd_chat_resume_replaces_persisted_stable_cwd_end_to_end(
    monkeypatch, tmp_path, manager
):
    from argparse import Namespace

    from agent.runtime_cwd import clear_session_cwd, resolve_agent_cwd
    import hermes_cli.main as main_module
    from tools.file_tools import _resolve_base_dir

    stable = tmp_path / "stable"
    managed = tmp_path / "managed"
    launch = tmp_path / "launch"
    stable.mkdir()
    managed.mkdir()
    launch.mkdir()
    session_id = "resume-root"
    db = _SessionDB(
        roots={session_id: session_id},
        session_cwds={session_id: str(stable)},
    )
    manager.resolve_existing_session = lambda root_session_id: _Binding(
        root_session_id=root_session_id,
        path=managed,
        branch=f"hermes/session/{root_session_id}",
        base_commit="b" * 40,
        repo_common_dir=manager.worktree_root,
    )
    captured: dict[str, str] = {}
    original_cwd = cli_module.os.getcwd()

    def run_classic_cli(**kwargs):
        app = cli_module.HermesCLI(compact=True, resume=kwargs["resume"])
        captured.update(
            process_cwd=cli_module.os.getcwd(),
            terminal_cwd=cli_module.os.environ["TERMINAL_CWD"],
            agent_cwd=str(resolve_agent_cwd()),
            tool_cwd=str(_resolve_base_dir()),
            working_directory=app.working_directory,
            system_prompt=app.system_prompt,
        )

    args = Namespace(
        accept_hooks=False,
        checkpoints=False,
        cli=True,
        compact=True,
        continue_last=None,
        ignore_rules=False,
        ignore_user_config=False,
        image=None,
        in_dir=None,
        max_turns=None,
        model=None,
        no_restore_cwd=False,
        pass_session_id=False,
        provider=None,
        query=None,
        query_file=None,
        quiet=False,
        reasoning=None,
        resume=session_id,
        run_budget=None,
        safe_mode=False,
        skills=None,
        source=None,
        toolsets=None,
        tui=False,
        tui_dev=False,
        verbose=None,
        worktree=False,
        yolo=False,
    )
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(cli_module, "CLI_CONFIG", _enabled_config())
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
    monkeypatch.setattr(cli_module, "_run_state_db_auto_maintenance", lambda _db: None)
    monkeypatch.setattr(cli_module, "_run_checkpoint_auto_maintenance", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "_build_cli_conversation_worktree_manager",
        lambda _config, _db: manager,
    )
    monkeypatch.setattr(main_module, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_module, "_resolve_session_by_name_or_id", lambda value: value)
    monkeypatch.setattr(main_module, "_sync_bundled_skills_for_startup", lambda: False)
    monkeypatch.setattr(main_module, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_module, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr(cli_module, "main", run_classic_cli)
    clear_session_cwd()

    try:
        cli_module.os.chdir(launch)
        main_module.cmd_chat(args)
    finally:
        cli_module.os.chdir(original_cwd)
        clear_session_cwd()

    expected = str(managed)
    assert captured["process_cwd"] == expected
    assert captured["terminal_cwd"] == expected
    assert captured["agent_cwd"] == expected
    assert captured["tool_cwd"] == expected
    assert captured["working_directory"] == expected
    assert expected in captured["system_prompt"]
    assert "certified Git worktree" in captured["system_prompt"]


def test_chdir_failure_aborts_managed_binding_without_exporting_source_fallback(
    monkeypatch, tmp_path, manager
):
    managed = tmp_path / "managed"
    managed.mkdir()
    manager._binding = lambda root_session_id: _Binding(
        root_session_id=root_session_id,
        path=managed,
        branch=f"hermes/session/{root_session_id}",
        base_commit="c" * 40,
        repo_common_dir=manager.worktree_root,
    )
    source = cli_module.os.getcwd()
    monkeypatch.setenv("TERMINAL_CWD", source)
    real_chdir = cli_module.os.chdir

    def fail_managed_chdir(path):
        if Path(path) == managed:
            raise OSError("permission denied")
        return real_chdir(path)

    monkeypatch.setattr(cli_module.os, "chdir", fail_managed_chdir)

    with pytest.raises(RuntimeError, match="could not enter managed conversation worktree"):
        _build_cli(monkeypatch, manager)

    assert cli_module.os.getcwd() == source
    assert cli_module.os.environ["TERMINAL_CWD"] == source


def test_list_only_cli_does_not_allocate_a_conversation_root(monkeypatch, manager):
    cli, _db = _build_cli(
        monkeypatch,
        manager,
        manage_conversation_worktree=False,
    )

    assert manager.bound_roots == []
    assert cli._conversation_worktree_manager is None


def test_cli_new_allocates_another_worktree_before_agent_reset(monkeypatch, manager):
    from agent.runtime_cwd import resolve_agent_cwd
    from tools.file_tools import _resolve_base_dir

    cli, _db = _build_cli(monkeypatch, manager)
    old = cli.working_directory
    old_session_id = cli.session_id
    manager.events.clear()
    cli.agent = MagicMock()
    cli.agent._memory_manager = None
    cli.agent.reset_session_state.side_effect = lambda: manager.events.append("reset")
    assert cli.new_session(silent=True) is True

    assert cli.working_directory != old
    assert manager.bound_roots == [manager.bound_roots[0], cli.session_id]
    assert manager.events.index("bind") < manager.events.index("reset")
    assert cli_module.os.getcwd() == cli.working_directory
    assert cli_module.os.environ["TERMINAL_CWD"] == cli.working_directory
    assert str(resolve_agent_cwd()) == cli.working_directory
    assert str(_resolve_base_dir()) == cli.working_directory
    assert cli._conversation_worktree_binding.root_session_id == cli.session_id
    assert cli.system_prompt.count("certified Git worktree") == 1
    assert old_session_id not in cli.system_prompt


def test_cli_new_candidate_chdir_failure_preserves_complete_old_boundary(
    monkeypatch, manager
):
    cli, db = _build_cli(monkeypatch, manager)
    cli.conversation_history = [{"role": "user", "content": "keep this turn"}]
    old_session_id = cli.session_id
    old_process_cwd = cli_module.os.getcwd()
    old_terminal_cwd = cli_module.os.environ["TERMINAL_CWD"]
    old_binding = cli._conversation_worktree_binding
    old_system_prompt = cli.system_prompt
    old_transcript = list(cli.conversation_history)
    old_ended = list(db.ended)
    old_created = list(db.created)
    real_chdir = cli_module.os.chdir
    candidate_lease = MagicMock()
    original_acquire = cli._acquire_conversation_root_lease

    def acquire(binding, *, surface):
        if binding.root_session_id != old_session_id:
            return candidate_lease
        return original_acquire(binding, surface=surface)

    monkeypatch.setattr(cli, "_acquire_conversation_root_lease", acquire)

    def fail_candidate_chdir(path):
        candidate = Path(path)
        if candidate.parent == manager.worktree_root and str(candidate) != old_process_cwd:
            raise OSError("candidate transition denied")
        return real_chdir(path)

    monkeypatch.setattr(cli_module.os, "chdir", fail_candidate_chdir)

    assert cli.new_session(silent=True) is False

    assert cli.session_id == old_session_id
    assert cli_module.os.getcwd() == old_process_cwd
    assert cli_module.os.environ["TERMINAL_CWD"] == old_terminal_cwd
    assert cli._conversation_worktree_binding is old_binding
    assert cli.system_prompt == old_system_prompt
    assert cli.conversation_history == old_transcript
    assert db.ended == old_ended
    assert db.created == old_created
    candidate_lease.release.assert_called_once_with()


def test_cli_new_binding_failure_leaves_old_session_usable(monkeypatch, manager):
    cli, db = _build_cli(monkeypatch, manager)
    old_id = cli.session_id
    old_working_directory = cli.working_directory
    cli.conversation_history = [{"role": "user", "content": "keep me"}]
    manager.fail_next = True

    assert cli.new_session(silent=True) is False

    assert cli.session_id == old_id
    assert cli.working_directory == old_working_directory
    assert cli.conversation_history == [{"role": "user", "content": "keep me"}]
    assert db.ended == []


@pytest.mark.parametrize(
    ("source", "kanban_task", "expected"),
    [
        ("cli", "", False),
        ("tool", "", True),
        ("kanban", "task-1", True),
    ],
)
def test_legacy_worktree_is_mutually_exclusive_only_for_interactive_roots(
    monkeypatch, source, kanban_task, expected
):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", source)
    if kanban_task:
        monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task)
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    assert cli_module._should_use_legacy_worktree(
        worktree=True,
        shorthand=False,
        config=_enabled_config(),
    ) is expected


def test_disabled_policy_preserves_legacy_worktree_flag(monkeypatch):
    config = _enabled_config()
    config["conversation_worktree"]["enabled"] = False
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")

    assert cli_module._should_use_legacy_worktree(
        worktree=True,
        shorthand=False,
        config=config,
    ) is True
