"""Strict-runtime and profile-specific Kanban concurrency guards."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _guard(
    root: Path,
    *,
    protected_cap: int = 3,
    include_linked_worktrees: bool = False,
) -> dict:
    return {
        "enabled": True,
        "project_roots": [str(root)],
        "entrypoints": ["main.py"],
        "include_linked_worktrees": include_linked_worktrees,
        "normal_max_in_progress": 8,
        "max_in_progress": protected_cap,
    }


@pytest.fixture
def isolated_kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_exact_runtime_snapshot_lowers_configured_performance_cap(tmp_path):
    root = tmp_path / "private-project"
    root.mkdir()
    scan = kb.ProcessScan(
        snapshots=(
            kb.ProcessSnapshot(
                pid=41,
                argv=("python3", "main.py"),
                cwd=str(root),
            ),
        ),
        complete=True,
    )

    assert (
        kb.resolve_max_in_progress(
            8,
            priority_runtime_guard=_guard(root),
            process_scan=scan,
        )
        == 3
    )


def test_guard_has_explicit_normal_performance_lane_when_runtime_is_absent(tmp_path):
    root = tmp_path / "private-project"
    root.mkdir()

    assert (
        kb.resolve_max_in_progress(
            None,
            priority_runtime_guard=_guard(root),
            process_scan=kb.ProcessScan(snapshots=(), complete=True),
        )
        == 8
    )

def test_absolute_entrypoint_under_exact_root_matches(tmp_path):
    root = tmp_path / "private-project"
    root.mkdir()
    scan = kb.ProcessScan(
        snapshots=(
            kb.ProcessSnapshot(
                pid=42,
                argv=("/usr/bin/python3", str(root / "main.py")),
                cwd="/tmp",
            ),
        ),
        complete=True,
    )

    assert kb.priority_runtime_state(_guard(root), process_scan=scan) == "active"


def test_unrelated_main_py_does_not_lower_cap(tmp_path):
    guarded_root = tmp_path / "guarded"
    unrelated_root = tmp_path / "unrelated"
    guarded_root.mkdir()
    unrelated_root.mkdir()
    scan = kb.ProcessScan(
        snapshots=(
            kb.ProcessSnapshot(
                pid=43,
                argv=("python3", "main.py"),
                cwd=str(unrelated_root),
            ),
            kb.ProcessSnapshot(
                pid=44,
                argv=("python3", "-c", "run main.py"),
                cwd=str(guarded_root),
            ),
            kb.ProcessSnapshot(
                pid=46,
                argv=("python3", "-m", "main.py"),
                cwd=str(guarded_root),
            ),
            kb.ProcessSnapshot(
                pid=47,
                argv=("echo", "main.py"),
                cwd=str(guarded_root),
            ),
        ),
        complete=True,
    )

    assert (
        kb.resolve_max_in_progress(
            8,
            priority_runtime_guard=_guard(guarded_root),
            process_scan=scan,
        )
        == 8
    )


def test_descendant_main_py_is_not_the_configured_root_entrypoint(tmp_path):
    root = tmp_path / "private-project"
    nested = root / "other"
    nested.mkdir(parents=True)
    scan = kb.ProcessScan(
        snapshots=(
            kb.ProcessSnapshot(
                pid=45,
                argv=("python3", "main.py"),
                cwd=str(nested),
            ),
        ),
        complete=True,
    )

    assert kb.priority_runtime_state(_guard(root), process_scan=scan) == "inactive"


def test_verified_linked_worktree_runtime_lowers_cap_when_enabled(tmp_path):
    root = tmp_path / "guarded"
    linked = tmp_path / "linked"
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Hermes Test",
            "-c",
            "user.email=hermes@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "initial",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "worktree", "add", "--quiet", "-b", "linked", str(linked)],
        check=True,
    )
    scan = kb.ProcessScan(
        snapshots=(
            kb.ProcessSnapshot(
                pid=48,
                argv=("python3", str(linked / "main.py")),
                cwd=str(linked),
            ),
        ),
        complete=True,
    )

    assert (
        kb.resolve_max_in_progress(
            8,
            priority_runtime_guard=_guard(root, include_linked_worktrees=True),
            process_scan=scan,
        )
        == 3
    )


def test_unrelated_repository_runtime_is_not_a_linked_worktree_match(tmp_path):
    guarded = tmp_path / "guarded"
    unrelated = tmp_path / "unrelated"
    subprocess.run(["git", "init", "--quiet", str(guarded)], check=True)
    subprocess.run(["git", "init", "--quiet", str(unrelated)], check=True)
    scan = kb.ProcessScan(
        snapshots=(
            kb.ProcessSnapshot(
                pid=49,
                argv=("python3", str(unrelated / "main.py")),
                cwd=str(unrelated),
            ),
        ),
        complete=True,
    )

    assert (
        kb.priority_runtime_state(
            _guard(guarded, include_linked_worktrees=True),
            process_scan=scan,
        )
        == "inactive"
    )


def test_incomplete_process_scan_fails_safe_to_protected_cap(tmp_path):
    root = tmp_path / "private-project"
    root.mkdir()

    assert (
        kb.resolve_max_in_progress(
            8,
            priority_runtime_guard=_guard(root),
            process_scan=kb.ProcessScan(snapshots=(), complete=False),
        )
        == 3
    )


def test_unreadable_login_process_cannot_hide_python_runtime():
    assert kb._process_name_can_hide_python_runtime("login") is False
    assert kb._process_name_can_hide_python_runtime("zsh") is False
    assert kb._process_name_can_hide_python_runtime("python3.11") is True


def test_disabled_or_unconfigured_guard_preserves_normal_cap(tmp_path):
    scan = kb.ProcessScan(snapshots=(), complete=False)

    assert (
        kb.resolve_max_in_progress(
            8,
            priority_runtime_guard={"enabled": False},
            process_scan=scan,
        )
        == 8
    )
    assert (
        kb.resolve_max_in_progress(
            8,
            priority_runtime_guard={"enabled": True, "project_roots": []},
            process_scan=scan,
        )
        == 8
    )


def test_protected_cap_never_raises_a_lower_operator_cap(tmp_path):
    root = tmp_path / "private-project"
    root.mkdir()

    assert (
        kb.resolve_max_in_progress(
            1,
            priority_runtime_guard=_guard(root, protected_cap=2),
            process_scan=kb.ProcessScan(snapshots=(), complete=False),
        )
        == 1
    )


def test_profile_cap_map_only_limits_selected_profile(
    isolated_kanban_home,
    all_assignees_spawnable,
):
    spawns: list[tuple[str, str]] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append((task.id, task.assignee))
        return 100 + len(spawns)

    with kb.connect() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"local-{i}", assignee="local-heavy")
            kb.create_task(conn, title=f"cloud-{i}", assignee="cloud-fast")
        result = kb.dispatch_once(
            conn,
            spawn_fn=fake_spawn,
            max_in_progress=8,
            max_in_progress_by_profile={"local-heavy": 1},
        )

    assert [assignee for _, assignee in spawns].count("local-heavy") == 1
    assert [assignee for _, assignee in spawns].count("cloud-fast") == 3
    assert len(result.skipped_per_profile_capped) == 2
