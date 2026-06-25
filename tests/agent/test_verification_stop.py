import json
import tempfile
from pathlib import Path

from agent.verification_evidence import (
    mark_workspace_edited,
    record_terminal_result,
)
from agent.verification_stop import (
    _is_non_code_path,
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


def test_verify_on_stop_default_is_on(monkeypatch):
    monkeypatch.delenv("HERMES_VERIFY_ON_STOP", raising=False)
    assert verify_on_stop_enabled({"agent": {}}) is True


def test_verify_on_stop_env_can_disable(monkeypatch):
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": True}}) is False


def test_verify_on_stop_config_can_disable(monkeypatch):
    monkeypatch.delenv("HERMES_VERIFY_ON_STOP", raising=False)
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": False}}) is False


def test_no_nudge_after_fresh_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    changed = str(tmp_path / "src" / "app.ts")

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=0,
        output="green",
    )

    assert build_verify_on_stop_nudge(session_id="s1", changed_paths=[changed]) is None


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


def test_nudge_after_unverified_edit_with_known_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    changed = str(tmp_path / "src" / "app.ts")
    mark_workspace_edited(session_id="s1", cwd=tmp_path, paths=[changed])

    nudge = build_verify_on_stop_nudge(session_id="s1", changed_paths=[changed])

    assert nudge is not None
    assert "fresh passing verification evidence" in nudge
    assert "`pnpm run test`" in nudge
    assert changed in nudge


def test_nudge_includes_failed_output_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    changed = str(tmp_path / "src" / "app.ts")

    record_terminal_result(
        command="pnpm test",
        cwd=tmp_path,
        session_id="s1",
        exit_code=1,
        output="expected 1 got 2",
    )

    nudge = build_verify_on_stop_nudge(session_id="s1", changed_paths=[changed])

    assert nudge is not None
    assert "failed" in nudge
    assert "expected 1 got 2" in nudge
    assert "repair the code" in nudge


def test_no_suite_nudge_requests_temp_script(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    changed = str(tmp_path / "src" / "app.ts")

    nudge = build_verify_on_stop_nudge(session_id="s1", changed_paths=[changed])

    assert nudge is not None
    assert tempfile.gettempdir() in nudge
    assert "ad-hoc verification" in nudge
    assert "suite green" in nudge


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


# --- Non-code path filtering (issue #52612) ---


def test_no_nudge_when_only_non_code_paths(tmp_path, monkeypatch):
    """Editing docs, gitignore, LICENSE etc. should NOT trigger the nudge."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    mark_workspace_edited(
        session_id="s1",
        cwd=tmp_path,
        paths=[
            str(tmp_path / "README.md"),
            str(tmp_path / ".gitignore"),
            str(tmp_path / "LICENSE"),
            str(tmp_path / ".github" / "workflows" / "ci.yml"),
        ],
    )

    assert build_verify_on_stop_nudge(
        session_id="s1",
        changed_paths=[
            str(tmp_path / "README.md"),
            str(tmp_path / ".gitignore"),
            str(tmp_path / "LICENSE"),
            str(tmp_path / ".github" / "workflows" / "ci.yml"),
        ],
    ) is None


def test_nudge_still_fires_for_mixed_code_and_non_code(tmp_path, monkeypatch):
    """Editing code + docs together should still trigger the nudge."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    code_path = str(tmp_path / "src" / "app.ts")
    mark_workspace_edited(
        session_id="s1", cwd=tmp_path, paths=[code_path],
    )

    nudge = build_verify_on_stop_nudge(
        session_id="s1",
        changed_paths=[code_path, str(tmp_path / "README.md")],
    )

    assert nudge is not None
    assert "fresh passing verification evidence" in nudge


def test_is_non_code_path_markdown():
    assert _is_non_code_path("README.md") is True
    assert _is_non_code_path("/repo/docs/architecture.rst") is True
    assert _is_non_code_path("notes.txt") is True


def test_is_non_code_path_git_metadata():
    assert _is_non_code_path(".gitignore") is True
    assert _is_non_code_path(".gitattributes") is True
    assert _is_non_code_path(".gitmodules") is True


def test_is_non_code_path_license():
    assert _is_non_code_path("LICENSE") is True
    assert _is_non_code_path("LICENCE") is True
    assert _is_non_code_path("COPYING") is True


def test_is_non_code_path_github_dir():
    assert _is_non_code_path(".github/workflows/ci.yml") is True
    assert _is_non_code_path(".github/ISSUE_TEMPLATE/bug.yml") is True
    assert _is_non_code_path(".github/CODEOWNERS") is True


def test_is_non_code_path_code_files_not_filtered():
    assert _is_non_code_path("src/app.ts") is False
    assert _is_non_code_path("agent/verification_stop.py") is False
    assert _is_non_code_path("package.json") is False
    assert _is_non_code_path("tsconfig.json") is False
    assert _is_non_code_path("pyproject.toml") is False
