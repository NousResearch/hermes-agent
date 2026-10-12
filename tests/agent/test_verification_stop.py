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
    _verification_snapshot,
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
    assert changed_b in nudge

@pytest.mark.platforms("posix")  # Symlinks require elevated privileges on Windows
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
    # The workspace must have a recorded edit for a nudge to fire: an unverified
    # workspace with an empty changed_paths ledger is now skipped entirely.
    mark_workspace_edited(
        session_id="s1", cwd=project, paths=[str(project / "src" / "app.ts")]
    )

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

# ---------------------------------------------------------------------------
# Fix D: a workspace that was never edited (`unverified` + empty ledger) is
# proof-complete, not outstanding.  It must not trip the nudge on its own, and
# it must not be resurrected as a fallback when every candidate is skipped.
# ---------------------------------------------------------------------------

def test_snapshot_none_for_lone_never_edited_workspace(tmp_path, monkeypatch):
    """No state row at all → verification_status() reports unverified with
    changed_paths: [] → _verification_snapshot returns None, no nudge fires."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    project = tmp_path / "project"
    _make_project(project)
    changed = str(project / "src" / "app.ts")

    assert _verification_snapshot(session_id="s1", changed_paths=[changed]) is None
    assert build_verify_on_stop_nudge(session_id="s1", changed_paths=[changed]) is None

def test_snapshot_none_when_all_candidates_skipped(tmp_path, monkeypatch):
    """proj_a verified(passed) + proj_b unverified with an empty ledger → no
    candidate needs proof → None, and the pre-filter snapshot is NOT returned
    as a fallback."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    _make_project(proj_a)
    _make_project(proj_b)
    changed_a = str(proj_a / "src" / "app.ts")
    changed_b = str(proj_b / "src" / "app.ts")

    record_terminal_result(
        command="pnpm test", cwd=proj_a, session_id="s1", exit_code=0, output="green",
    )
    mark_workspace_edited(session_id="s1", cwd=proj_b, paths=[])

    assert _verification_snapshot(
        session_id="s1", changed_paths=[changed_a, changed_b]
    ) is None
    assert build_verify_on_stop_nudge(
        session_id="s1", changed_paths=[changed_a, changed_b]
    ) is None

def test_snapshot_still_returns_workspace_with_pending_edits(tmp_path, monkeypatch):
    """Over-skip guard: an empty-ledger workspace alongside a genuinely edited
    one must not suppress the nudge, and the *edited* workspace must be the one
    selected (the nudge body lists every changed path, so the discriminator is
    the returned status, not the nudge text)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    proj_clean = tmp_path / "clean"
    proj_edited = tmp_path / "edited"
    _make_project(proj_clean)
    _make_project(proj_edited)
    changed_clean = str(proj_clean / "src" / "app.ts")
    changed_edited = str(proj_edited / "src" / "app.ts")

    mark_workspace_edited(session_id="s1", cwd=proj_edited, paths=[changed_edited])

    snapshot = _verification_snapshot(
        session_id="s1", changed_paths=[changed_clean, changed_edited]
    )
    assert snapshot is not None
    status, _facts = snapshot
    assert status.get("changed_paths"), (
        "the edited workspace (non-empty ledger) must be the snapshot selected, "
        f"got {status!r}"
    )
    assert build_verify_on_stop_nudge(
        session_id="s1", changed_paths=[changed_clean, changed_edited]
    ) is not None
