import json
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


def test_verify_on_stop_default_is_off(clear_verify_env):
    # No env, no explicit config -> default OFF (new default as of v31).
    assert verify_on_stop_enabled({"agent": {}}) is False


def test_verify_on_stop_missing_agent_section_is_off(clear_verify_env):
    assert verify_on_stop_enabled({}) is False


def test_verify_on_stop_auto_sentinel_resolves_to_surface_default(clear_verify_env):
    # The legacy "auto" sentinel is still honored when set explicitly: it falls
    # through to the surface-aware default (ON interactive, OFF messaging).
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "auto"}}) is True
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "auto"}}) is False


def test_verify_on_stop_env_can_disable(clear_verify_env):
    clear_verify_env.setenv("HERMES_VERIFY_ON_STOP", "0")
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": True}}) is False


def test_verify_on_stop_env_can_enable(clear_verify_env):
    # Env wins over the default-off config.
    clear_verify_env.setenv("HERMES_VERIFY_ON_STOP", "1")
    assert verify_on_stop_enabled({"agent": {}}) is True


def test_verify_on_stop_config_true_enables(clear_verify_env):
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": True}}) is True


def test_verify_on_stop_config_can_disable(clear_verify_env):
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": False}}) is False


def test_verify_on_stop_auto_off_on_gateway_messaging_platform(clear_verify_env):
    # With explicit "auto", a real Telegram turn resolves OFF.
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "auto"}}) is False


@pytest.mark.parametrize(
    "platform",
    ["discord", "whatsapp_cloud", "signal", "slack", "matrix", "email", "sms"],
)
def test_verify_on_stop_auto_off_for_each_messaging_platform(clear_verify_env, platform):
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", platform)
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "auto"}}) is False


def test_verify_on_stop_auto_messaging_platform_is_case_insensitive(clear_verify_env):
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", "  Telegram  ")
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "auto"}}) is False


def test_verify_on_stop_auto_uses_hermes_platform_override(clear_verify_env):
    # HERMES_PLATFORM mirrors the sibling platform resolution and also flags a
    # messaging surface under the "auto" sentinel.
    clear_verify_env.setenv("HERMES_PLATFORM", "discord")
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "auto"}}) is False


@pytest.mark.parametrize("source", ["cli", "tui", "desktop", "codex", "local"])
def test_verify_on_stop_auto_on_for_interactive_surfaces(clear_verify_env, source):
    # Under "auto", CLI/TUI/desktop coding surfaces resolve ON.
    clear_verify_env.setenv("HERMES_SESSION_SOURCE", source)
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "auto"}}) is True


@pytest.mark.parametrize("platform", ["api_server", "webhook", "msgraph_webhook"])
def test_verify_on_stop_auto_on_for_programmatic_surfaces(clear_verify_env, platform):
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", platform)
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": "auto"}}) is True


def test_default_off_overrides_interactive_surface(clear_verify_env):
    # The new default is OFF even on an interactive coding surface — only an
    # explicit "auto"/true turns it back on.
    clear_verify_env.setenv("HERMES_SESSION_SOURCE", "cli")
    assert verify_on_stop_enabled({"agent": {}}) is False


def test_env_forces_verify_on_stop_on_for_messaging(clear_verify_env):
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    clear_verify_env.setenv("HERMES_VERIFY_ON_STOP", "1")
    assert verify_on_stop_enabled({"agent": {}}) is True


def test_config_forces_verify_on_stop_on_for_messaging(clear_verify_env):
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert verify_on_stop_enabled({"agent": {"verify_on_stop": True}}) is True


def test_verify_on_stop_default_path_through_load_config(tmp_path, clear_verify_env):
    # E2E: the sole production caller passes no config, so verify_on_stop_enabled
    # resolves through load_config() + DEFAULT_CONFIG. The default is now the
    # boolean False, so even an interactive surface resolves OFF without an
    # explicit opt-in. This is the path the unit-level tests above cannot
    # exercise.
    clear_verify_env.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from hermes_cli.config import load_config

    merged = load_config()
    assert merged["agent"]["verify_on_stop"] is False

    # Interactive surface still resolves OFF through the real loader.
    clear_verify_env.setenv("HERMES_SESSION_SOURCE", "cli")
    assert verify_on_stop_enabled() is False

    # A messaging platform also resolves OFF.
    clear_verify_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert verify_on_stop_enabled() is False


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


# ---------------------------------------------------------------------------
# Fix C: documentation/prose edits carry no verifiable behavior and must never
# trip the nudge, even on an unverified workspace.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "doc_name",
    [
        "SKILL.md",
        "README.md",
        "guide.markdown",
        "page.mdx",
        "manual.rst",
        "notes.txt",
        "data.csv",
        "LICENSE",
        "CHANGELOG",
    ],
)
def test_doc_only_edit_does_not_nudge(tmp_path, monkeypatch, doc_name):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _node_project(tmp_path)
    changed = str(tmp_path / doc_name)
    mark_workspace_edited(session_id="s1", cwd=tmp_path, paths=[changed])

    # Unverified workspace, but the only edit is a doc — nothing to verify.
    assert build_verify_on_stop_nudge(session_id="s1", changed_paths=[changed]) is None


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
