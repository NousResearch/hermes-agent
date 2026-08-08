import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from agent.verification_evidence import (
    mark_workspace_edited,
    record_terminal_result,
)
from agent.verification_stop import (
    build_verify_on_stop_nudge,
    verify_on_stop_enabled,
)


def _node_project(root: Path) -> None:
    (root / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest", "lint": "eslint ."}}),
        encoding="utf-8",
    )
    (root / "pnpm-lock.yaml").write_text("", encoding="utf-8")


def _make_project(root: Path) -> None:
    root.mkdir()
    _node_project(root)


def _init_git_project(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _commit_all(root: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def clear_verify_env(monkeypatch):
    """Clear every env signal verify_on_stop_enabled consults.

    Tests then set only the variable they exercise, mirroring how the CLI/TUI
    set HERMES_SESSION_SOURCE and the gateway sets HERMES_SESSION_PLATFORM.
    """
    for var in (
        "HERMES_VERIFY_ON_STOP",
        "HERMES_PLATFORM",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_SOURCE",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch












def test_verify_on_stop_env_can_enable(clear_verify_env):
    # Env "1" forces ON regardless of surface (here a messaging platform).
    clear_verify_env.setenv("HERMES_VERIFY_ON_STOP", "1")
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert verify_on_stop_enabled({"agent": {}}) is True














@pytest.mark.parametrize("source", ["cli", "tui", "desktop", "codex", "local"])
def test_verify_on_stop_auto_on_for_interactive_surfaces(clear_verify_env, source):
    # Under "auto", CLI/TUI/desktop coding surfaces resolve ON.
    clear_verify_env.setenv("HERMES_SESSION_SOURCE", source)
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "auto"}}) is True










def test_verify_on_stop_default_path_through_load_config(tmp_path, clear_verify_env):
    # E2E: the sole production caller passes no config, so verify_on_stop_enabled
    # resolves through load_config() + DEFAULT_CONFIG. The default is now the
    # surface-aware "auto" sentinel. This is the path the unit-level tests above
    # cannot exercise.
    clear_verify_env.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from hermes_cli.config import load_config

    merged = load_config()
    assert merged["agent"]["verify_on_stop"] == "auto"

    # Interactive surface resolves ON through the real loader.
    clear_verify_env.setenv("HERMES_SESSION_SOURCE", "cli")
    assert verify_on_stop_enabled() is True

    # A messaging platform resolves OFF.
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert verify_on_stop_enabled() is False




def test_nudge_checks_all_edited_workspaces(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    _make_project(project_a)
    _make_project(project_b)
    changed_a = str(project_a / "src" / "app.ts")
    changed_b = str(project_b / "src" / "app.ts")

    record_terminal_result(
        command="pnpm test",
        cwd=project_a,
        session_id="s1",
        exit_code=0,
        output="green",
    )
    mark_workspace_edited(session_id="s1", cwd=project_b, paths=[changed_b])

    nudge = build_verify_on_stop_nudge(
        session_id="s1",
        changed_paths=[changed_a, changed_b],
    )

    assert nudge is not None
    assert "fresh passing verification evidence" in nudge








def test_no_suite_nudge_uses_canonical_temp_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text("{}", encoding="utf-8")
    real_temp = tmp_path / "real-temp"
    real_temp.mkdir()
    linked_temp = tmp_path / "linked-temp"
    linked_temp.symlink_to(real_temp, target_is_directory=True)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(linked_temp))

    nudge = build_verify_on_stop_nudge(
        session_id="s1",
        changed_paths=[str(project / "src" / "app.ts")],
    )

    assert nudge is not None
    assert str(real_temp) in nudge
    assert str(linked_temp) not in nudge




def test_ad_hoc_pass_satisfies_no_suite_stop_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    changed = str(tmp_path / "src" / "app.ts")
    script = Path(tempfile.gettempdir()) / f"hermes-ad-hoc-stop-{tmp_path.name}.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    try:
        record_terminal_result(
            command=f"python {script}",
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output="ok",
        )
    finally:
        script.unlink(missing_ok=True)

    assert build_verify_on_stop_nudge(session_id="s1", changed_paths=[changed]) is None


def test_nudge_attempts_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    changed = str(tmp_path / "src" / "app.ts")
    mark_workspace_edited(session_id="s1", cwd=tmp_path, paths=[changed])

    assert build_verify_on_stop_nudge(
        session_id="s1",
        changed_paths=[changed],
        attempts=2,
        max_attempts=2,
    ) is None


def test_committed_paths_do_not_keep_stale_verification_loop_alive(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    project = tmp_path / "project"
    _make_project(project)
    src = project / "src"
    src.mkdir()
    changed = src / "app.ts"
    changed.write_text("export const value = 1;\n", encoding="utf-8")

    _init_git_project(project)
    _commit_all(project, "initial")

    record_terminal_result(
        command="pnpm test",
        cwd=project,
        session_id="s1",
        exit_code=0,
        output="old green output",
    )
    changed.write_text("export const value = 2;\n", encoding="utf-8")
    mark_workspace_edited(session_id="s1", cwd=project, paths=[str(changed)])
    _commit_all(project, "change value")

    assert build_verify_on_stop_nudge(
        session_id="s1",
        changed_paths=[str(changed)],
    ) is None


def test_uncommitted_paths_still_require_fresh_verification(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    project = tmp_path / "project"
    _make_project(project)
    src = project / "src"
    src.mkdir()
    changed = src / "app.ts"
    changed.write_text("export const value = 1;\n", encoding="utf-8")

    _init_git_project(project)
    _commit_all(project, "initial")

    changed.write_text("export const value = 2;\n", encoding="utf-8")
    mark_workspace_edited(session_id="s1", cwd=project, paths=[str(changed)])

    nudge = build_verify_on_stop_nudge(
        session_id="s1",
        changed_paths=[str(changed)],
    )

    assert nudge is not None
    assert str(changed) in nudge


# ---------------------------------------------------------------------------
# Fix C: documentation/prose edits carry no verifiable behavior and must never
# trip the nudge, even on an unverified workspace.
# ---------------------------------------------------------------------------



def test_mixed_doc_and_code_edit_still_nudges(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    doc = str(tmp_path / "README.md")
    code = str(tmp_path / "src" / "app.ts")
    mark_workspace_edited(session_id="s1", cwd=tmp_path, paths=[code])

    nudge = build_verify_on_stop_nudge(
        session_id="s1", changed_paths=[doc, code]
    )
    assert nudge is not None
    # The doc path is filtered out of the reported set; the code path remains.
    assert code in nudge
    assert doc not in nudge


def test_is_non_code_path_classification():
    from agent.verification_stop import _is_non_code_path

    assert _is_non_code_path("docs/SKILL.md") is True
    assert _is_non_code_path("README") is False  # README has no extension and isn't in the prose-filename set
    assert _is_non_code_path("LICENSE") is True
    assert _is_non_code_path("src/app.ts") is False
    assert _is_non_code_path("config.yaml") is False
    assert _is_non_code_path("run_agent.py") is False
