import json
from pathlib import Path

import pytest

from hermes_constants import get_scratch_dir

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








def test_no_suite_nudge_uses_active_hermes_scratch(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text("{}", encoding="utf-8")

    nudge = build_verify_on_stop_nudge(
        session_id="s1",
        changed_paths=[str(project / "src" / "app.ts")],
    )

    assert nudge is not None
    assert str(get_scratch_dir(home, prune=False).resolve()) in nudge


def test_no_suite_nudge_preserves_scope_and_names_active_hermes_scratch(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text("{}", encoding="utf-8")

    nudge = build_verify_on_stop_nudge(
        session_id="s1", changed_paths=[str(project / "src" / "app.ts")]
    )

    assert nudge is not None
    assert str(project.resolve()) in nudge
    assert str(get_scratch_dir(home, prune=False).resolve()) in nudge
    assert "does not authorize writes outside the task scope" in nudge
    assert nudge.index(str(get_scratch_dir(home, prune=False).resolve())) < nudge.index(str(project.resolve()), nudge.index("No canonical"))
    assert "never stage or commit" in nudge
    assert "delete the script before finishing" in nudge




def test_no_suite_nudge_follows_profile_scope_a_b_a(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text("{}", encoding="utf-8")
    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for home in [homes[0], homes[1], homes[0]]:
        token = set_hermes_home_override(home)
        try:
            nudge = build_verify_on_stop_nudge(
                session_id="scoped", changed_paths=[str(project / "app.py")]
            )
            assert nudge is not None
            assert str(get_scratch_dir(home, prune=False).resolve()) in nudge
            other = homes[1] if home == homes[0] else homes[0]
            assert str(other.resolve()) not in nudge
        finally:
            reset_hermes_home_override(token)


def test_ad_hoc_pass_satisfies_no_suite_stop_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    changed = str(tmp_path / "src" / "app.ts")
    script = get_scratch_dir(tmp_path / ".hermes", prune=False) / f"hermes-ad-hoc-stop-{tmp_path.name}.py"
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
