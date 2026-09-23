import json
import sys
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


@pytest.fixture(autouse=True)
def _ledger_on(monkeypatch):
    """The ledger is inert unless verify-on-stop is enabled; ``clear_verify_env`` (requested
    explicitly, so it runs after this) strips it again for the enabled()-logic tests."""
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")


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
    # resolves through load_config() + DEFAULT_CONFIG. The default is now False
    # (opt-in): fresh installs must not fire the nudge on any surface. This is
    # the path the unit-level tests above cannot exercise.
    clear_verify_env.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from hermes_cli.config import load_config

    merged = load_config()
    assert merged["agent"]["verify_on_stop"] is False

    # Interactive surface resolves OFF through the real loader (opt-in default).
    clear_verify_env.setenv("HERMES_SESSION_SOURCE", "cli")
    assert verify_on_stop_enabled() is False

    # A messaging platform also resolves OFF.
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert verify_on_stop_enabled() is False


def test_verify_on_stop_missing_value_defaults_off(clear_verify_env):
    # A missing/unrecognized config value falls back OFF on every surface,
    # matching the opt-in DEFAULT_CONFIG default — only an explicit "auto"
    # opts into the legacy surface-aware behavior.
    clear_verify_env.setenv("HERMES_SESSION_SOURCE", "cli")
    assert verify_on_stop_enabled({"agent": {}}) is False
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "bogus"}}) is False
    assert verify_on_stop_enabled({}) is False




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








@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Symlinks require elevated privileges on Windows",
)
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


def test_ephemeral_verify_artifacts_do_not_nudge(tmp_path):
    """The gate must not feed on its own evidence scripts or deleted scratch.

    Regression for the self-feeding loop observed 2026-08-19: each round's
    `hermes-verify-*` temp script became the next round's "changed path", so
    the gate had no terminal state.
    """
    import tempfile as _tempfile

    from agent.verification_stop import (
        _filter_verifiable_paths,
        _is_ephemeral_verify_artifact,
    )

    temp_root = Path(_tempfile.gettempdir())

    # 1. Ad-hoc verify scripts under temp are ephemeral even while they exist.
    probe = temp_root / "hermes-verify-abc123.sh"
    probe.write_text("#!/bin/bash\necho ok\n")
    try:
        assert _is_ephemeral_verify_artifact(str(probe)) is True
    finally:
        probe.unlink()
    # ... and after deletion.
    assert _is_ephemeral_verify_artifact(str(probe)) is True
    assert _is_ephemeral_verify_artifact(str(temp_root / "hermes-ad-hoc-x.py")) is True

    # 2. A DELETED ordinary temp path has nothing left to verify.
    gone = temp_root / "task_check_gone_xyz.sh"
    assert not gone.exists()
    assert _is_ephemeral_verify_artifact(str(gone)) is True

    # 3. An EXISTING ordinary temp script still verifies (only deletion or the
    #    verify prefix exempts it).
    live = temp_root / "hermes_stop_gate_live_fixture.sh"
    live.write_text("#!/bin/bash\n")
    try:
        assert _is_ephemeral_verify_artifact(str(live)) is False
    finally:
        live.unlink()

    # 4. Non-temp paths that EXIST, and deleted paths inside a real workspace,
    #    never get the exemption (a deleted tracked file is a real change to
    #    verify). Use the live repo as the workspace fixture: pytest's tmp_path
    #    is under the system temp dir, so it cannot stand in for one, and a
    #    made-up path like /Users/nobody/repo is neither existing nor inside a
    #    workspace -- it would now be treated as deleted scratch and silently
    #    stop testing what this case is about.
    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / ".git").exists(), "fixture expects the real checkout"

    repo_file = str(repo_root / "agent" / "verification_stop.py")
    assert Path(repo_file).exists()
    assert _is_ephemeral_verify_artifact(repo_file) is False

    # Deleted, but inside a .git workspace -> still a real change.
    repo_deleted = str(repo_root / "agent" / "deleted_fixture_never_created.py")
    assert not Path(repo_deleted).exists()
    assert _is_ephemeral_verify_artifact(repo_deleted) is False

    # A magic-named lookalike inside a real workspace is NOT exempt either:
    # the hermes-verify-/hermes-ad-hoc- prefix only applies under temp.
    lookalike = str(repo_root / "agent" / "hermes-verify-lookalike.sh")
    assert _is_ephemeral_verify_artifact(lookalike) is False

    # The same contract must hold for a real workspace located UNDER /tmp.
    # Update rehearsals use a disposable worktree there, whose ``.git`` is
    # a file rather than a directory. A magic verify name inside that repo
    # is still tracked code, never disposable evidence.
    temp_repo = tmp_path / "repo"
    temp_repo.mkdir()
    (temp_repo / ".git").write_text("gitdir: /tmp/fake-worktree-meta\n")
    temp_lookalike = temp_repo / "hermes-verify-lookalike.sh"
    temp_lookalike.write_text("#!/bin/bash\n")
    assert _is_ephemeral_verify_artifact(str(temp_lookalike)) is False

    # ...and the boundary still holds the other way: a stray marker sitting
    # DIRECTLY in the temp root must not promote loose scratch to project code.
    # Mutation-proven 2026-08-28: swapping the temp-aware walk for the generic
    # _in_workspace() makes this assertion fail whenever /tmp/package.json
    # exists, which is exactly how the deleted-scratch exemption died before.
    stray_marker = Path(tempfile.gettempdir()) / "package.json"
    created_stray = not stray_marker.exists()
    if created_stray:
        stray_marker.write_text("{}\n")
    try:
        loose_probe = Path(tempfile.gettempdir()) / "hermes-ad-hoc-boundary-probe.py"
        loose_probe.write_text("x = 1\n")
        try:
            assert _is_ephemeral_verify_artifact(str(loose_probe)) is True
        finally:
            loose_probe.unlink(missing_ok=True)
    finally:
        if created_stray:
            stray_marker.unlink(missing_ok=True)

    # 4b. Deleted scratch OUTSIDE temp and outside any workspace IS exempt.
    #     A file that no longer exists has no behavior left to verify, so
    #     demanding evidence for it can never be satisfied and re-nudges every
    #     turn (regression guard for the 2026-08-22 loop: a throwaway
    #     ~/.hermes/scripts restart script nudged three turns running).
    #     $HOME itself must not count as a workspace -- a stray ~/package.json
    #     otherwise makes every path under home look like a project.
    home_scratch = str(Path.home() / ".hermes" / "scripts" / "deleted-scratch-fixture.sh")
    assert not Path(home_scratch).exists()
    assert _is_ephemeral_verify_artifact(home_scratch) is True

    # ...but the same directory's LIVE files still nudge.
    for sibling in (Path.home() / ".hermes" / "scripts").glob("*.sh"):
        assert _is_ephemeral_verify_artifact(str(sibling)) is False
        break

    # 5. Relative paths are left alone.
    assert _is_ephemeral_verify_artifact("src/app.ts") is False

    # 6. End to end through the filter: only the real repo file survives.
    kept = _filter_verifiable_paths(
        [str(probe), str(gone), repo_file, "notes.md"]
    )
    assert kept == [repo_file]
