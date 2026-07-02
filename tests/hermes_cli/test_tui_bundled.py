import sys

import pytest


def test_tui_finds_bundled_entry_js(tmp_path):
    """_find_bundled_tui finds entry.js bundled in the package."""
    tui_dist = tmp_path / "hermes_cli" / "tui_dist"
    tui_dist.mkdir(parents=True)
    entry = tui_dist / "entry.js"
    entry.write_text("// bundled TUI", encoding="utf-8")

    from hermes_cli.main import _find_bundled_tui
    result = _find_bundled_tui(hermes_cli_dir=tmp_path / "hermes_cli")
    assert result is not None
    assert result.name == "entry.js"


def test_tui_returns_none_when_no_bundle(tmp_path):
    """_find_bundled_tui returns None when no bundle exists."""
    from hermes_cli.main import _find_bundled_tui
    result = _find_bundled_tui(hermes_cli_dir=tmp_path / "hermes_cli")
    assert result is None


def test_prebuilt_bundle_used_before_workspace_guard(tmp_path, monkeypatch):
    """Regression: a wheel install (no ``ui-tui/`` source, not a git checkout) must
    run the prebuilt bundle instead of aborting in ``_ensure_tui_workspace()``.

    Previously ``_make_tui_argv`` ran the workspace guard *before* the
    prebuilt-bundle checks, so pip/uv installs died with a bogus
    "git restore -- ui-tui" message and never reached the bundle the wheel ships.
    """
    import hermes_cli.main as m

    tui_dir = tmp_path / "ui-tui"  # deliberately absent, as on a wheel install
    assert not tui_dir.exists()

    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setenv("HERMES_NODE", sys.executable)  # a real executable file
    monkeypatch.setattr(m, "_ensure_tui_node", lambda: None)

    bundled = tmp_path / "hermes_cli" / "tui_dist" / "entry.js"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("// bundled TUI", encoding="utf-8")
    monkeypatch.setattr(m, "_find_bundled_tui", lambda *a, **k: bundled)

    def _guard_must_not_run(_td):
        raise AssertionError(
            "_ensure_tui_workspace ran before the prebuilt-bundle check"
        )

    monkeypatch.setattr(m, "_ensure_tui_workspace", _guard_must_not_run)

    argv, cwd = m._make_tui_argv(tui_dir, tui_dev=False)

    assert argv[0] == sys.executable
    assert argv[-1] == str(bundled)
    assert cwd == bundled.parent


def test_workspace_guard_runs_when_no_bundle(tmp_path, monkeypatch):
    """The relocated guard still runs on the source-build path: when there is no
    prebuilt bundle and no ``HERMES_TUI_DIR``, ``_make_tui_argv`` must fall through
    to ``_ensure_tui_workspace()`` before attempting an npm/esbuild build.
    """
    import hermes_cli.main as m

    tui_dir = tmp_path / "ui-tui"

    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setenv("HERMES_NODE", sys.executable)
    monkeypatch.setattr(m, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(m, "_find_bundled_tui", lambda *a, **k: None)

    class _GuardReached(Exception):
        pass

    def _guard(td):
        assert td == tui_dir
        raise _GuardReached

    monkeypatch.setattr(m, "_ensure_tui_workspace", _guard)

    with pytest.raises(_GuardReached):
        m._make_tui_argv(tui_dir, tui_dev=False)
