from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli.kanban_runtime import (
    KANBAN_TERMINAL_RUNTIME_ENV,
    KanbanRuntimeError,
    build_kanban_terminal_runtime,
    decode_kanban_terminal_runtime,
    encode_kanban_terminal_runtime,
    is_remote_docker_host,
    translate_host_path,
    translate_runtime_mounts,
)


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_runtime_env_name_is_kanban_scoped():
    assert KANBAN_TERMINAL_RUNTIME_ENV == "HERMES_KANBAN_TERMINAL_RUNTIME"


def test_dir_runtime_is_exact_workspace_mount(tmp_path: Path):
    ws = tmp_path / "task"
    ws.mkdir()
    runtime = build_kanban_terminal_runtime(
        task_id="t_1", workspace_kind="dir", workspace=ws
    )
    assert runtime["task_id"] == "t_1"
    assert runtime["workspace_kind"] == "dir"
    assert runtime["container_cwd"] == "/workspace"
    assert runtime["mounts"] == [
        {
            "source": str(ws.resolve()),
            "target": "/workspace",
            "read_only": False,
            "purpose": "workspace",
        }
    ]


def test_encode_decode_checks_task_and_workspace(tmp_path: Path):
    ws = tmp_path / "task"
    ws.mkdir()
    raw = encode_kanban_terminal_runtime(
        build_kanban_terminal_runtime(
            task_id="t_2", workspace_kind="scratch", workspace=ws
        )
    )
    decoded = decode_kanban_terminal_runtime(
        raw, expected_task_id="t_2", expected_workspace=ws
    )
    assert decoded["workspace"] == str(ws.resolve())

    with pytest.raises(KanbanRuntimeError, match="task mismatch"):
        decode_kanban_terminal_runtime(raw, expected_task_id="t_other")


def test_runtime_rejects_workspace_retarget(tmp_path: Path):
    ws = tmp_path / "task"
    other = tmp_path / "other"
    ws.mkdir()
    other.mkdir()
    runtime = build_kanban_terminal_runtime(
        task_id="t_3", workspace_kind="dir", workspace=ws
    )
    runtime["mounts"][0]["source"] = str(other)
    with pytest.raises(KanbanRuntimeError, match="workspace mount"):
        encode_kanban_terminal_runtime(runtime)


def test_remote_docker_requires_explicit_mapping(tmp_path: Path):
    ws = tmp_path / "projects" / "task"
    ws.mkdir(parents=True)
    assert is_remote_docker_host("ssh://docker@example.com")
    assert not is_remote_docker_host("unix:///var/run/docker.sock")
    assert not is_remote_docker_host("tcp://127.0.0.1:2375")

    with pytest.raises(KanbanRuntimeError, match="remote"):
        translate_host_path(
            ws, path_map=[], docker_host="ssh://docker@example.com"
        )

    mapped = translate_host_path(
        ws,
        path_map=[
            {
                "local_root": str(tmp_path / "projects"),
                "host_root": "/mnt/unraid/projects",
            }
        ],
        docker_host="ssh://docker@example.com",
    )
    assert mapped == "/mnt/unraid/projects/task"


def test_longest_path_map_wins(tmp_path: Path):
    root = tmp_path / "projects"
    special = root / "special"
    ws = special / "task"
    ws.mkdir(parents=True)
    mapped = translate_host_path(
        ws,
        path_map=[
            {"local_root": str(root), "host_root": "/host/projects"},
            {"local_root": str(special), "host_root": "/host/special"},
        ],
        docker_host="ssh://docker@example.com",
    )
    assert mapped == "/host/special/task"


def test_worktree_runtime_adds_only_common_git_metadata(tmp_path: Path):
    repo = tmp_path / "repo"
    _git("init", str(repo))
    _git("-C", str(repo), "config", "user.email", "t@example.com")
    _git("-C", str(repo), "config", "user.name", "t")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git("-C", str(repo), "add", "README.md")
    _git("-C", str(repo), "commit", "-m", "init")

    wt = tmp_path / "wt"
    _git("-C", str(repo), "worktree", "add", "-b", "feature/t", str(wt))

    runtime = build_kanban_terminal_runtime(
        task_id="t_wt", workspace_kind="worktree", workspace=wt
    )
    assert len(runtime["mounts"]) == 2
    assert runtime["mounts"][0]["target"] == "/workspace"
    git_mount = runtime["mounts"][1]
    assert git_mount["purpose"] == "git-common-dir"
    assert git_mount["source"] == str((repo / ".git").resolve())
    assert git_mount["target"] == str((repo / ".git").resolve())
    assert str(repo.resolve()) not in {
        m["source"] for m in runtime["mounts"] if m["purpose"] != "git-common-dir"
    }


def test_worktree_runtime_rejects_forged_extra_host_mount(tmp_path: Path):
    repo = tmp_path / "repo"
    _git("init", str(repo))
    _git("-C", str(repo), "config", "user.email", "t@example.com")
    _git("-C", str(repo), "config", "user.name", "t")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git("-C", str(repo), "add", "README.md")
    _git("-C", str(repo), "commit", "-m", "init")
    wt = tmp_path / "wt"
    _git("-C", str(repo), "worktree", "add", "-b", "feature/forged", str(wt))

    runtime = build_kanban_terminal_runtime(
        task_id="t_forged", workspace_kind="worktree", workspace=wt
    )
    sibling = tmp_path / "sibling-project"
    sibling.mkdir()
    runtime["mounts"][1]["source"] = str(sibling)
    runtime["mounts"][1]["target"] = str(sibling)

    with pytest.raises(KanbanRuntimeError, match="does not match"):
        encode_kanban_terminal_runtime(runtime)


def test_translate_runtime_mounts_maps_workspace_and_git_metadata(tmp_path: Path):
    repo = tmp_path / "projects" / "repo"
    _git("init", str(repo))
    _git("-C", str(repo), "config", "user.email", "t@example.com")
    _git("-C", str(repo), "config", "user.name", "t")
    (repo / "a").write_text("a", encoding="utf-8")
    _git("-C", str(repo), "add", "a")
    _git("-C", str(repo), "commit", "-m", "init")
    wt = tmp_path / "projects" / "wt"
    _git("-C", str(repo), "worktree", "add", "-b", "feature/w", str(wt))

    runtime = build_kanban_terminal_runtime(
        task_id="t_w",
        workspace_kind="worktree",
        workspace=wt,
        authorized_roots=[tmp_path / "projects"],
    )
    mounts = translate_runtime_mounts(
        runtime,
        path_map=[
            {
                "local_root": str(tmp_path / "projects"),
                "host_root": "/srv/projects",
            }
        ],
        docker_host="ssh://docker@example.com",
    )
    assert mounts[0]["source"] == "/srv/projects/wt"
    assert mounts[0]["target"] == "/workspace"
    assert mounts[1]["source"] == "/srv/projects/repo/.git"


def test_json_is_deterministic(tmp_path: Path):
    ws = tmp_path / "task"
    ws.mkdir()
    runtime = build_kanban_terminal_runtime(
        task_id="t_json", workspace_kind="dir", workspace=ws
    )
    raw = encode_kanban_terminal_runtime(runtime)
    assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("outside_name", ["opt-data", ".hermes", "sibling-project"])
def test_runtime_authority_rejects_self_consistent_outside_workspace(
    tmp_path: Path, outside_name: str
):
    project = tmp_path / "project"
    outside = tmp_path / outside_name
    project.mkdir()
    outside.mkdir()

    # Envelope and expected workspace agree.  That is intentionally not enough:
    # physical Docker bind authority comes from the independent project root.
    raw = encode_kanban_terminal_runtime(
        build_kanban_terminal_runtime(
            task_id="t_outside",
            workspace_kind="dir",
            workspace=outside,
            authorized_roots=[project],
        )
    )
    runtime = decode_kanban_terminal_runtime(
        raw, expected_task_id="t_outside", expected_workspace=outside
    )
    with pytest.raises(KanbanRuntimeError, match="outside authorized workspace roots"):
        translate_runtime_mounts(runtime, path_map=[], docker_host=None)


def test_runtime_authority_allows_project_child(tmp_path: Path):
    project = tmp_path / "project"
    workspace = project / "task"
    workspace.mkdir(parents=True)
    runtime = build_kanban_terminal_runtime(
        task_id="t_allowed",
        workspace_kind="dir",
        workspace=workspace,
        authorized_roots=[project],
    )
    mounts = translate_runtime_mounts(runtime, path_map=[], docker_host=None)
    assert mounts[0]["source"] == str(workspace.resolve())
    assert mounts[0]["target"] == "/workspace"
