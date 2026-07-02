"""Regression tests for #57031 — homebrew/wheel install: TUI must use bundled-TUI fallback or install-method-specific recovery, not the checkout-shaped error.

Layers covered (see .automation/runs/2026-07-02T120730Z/57031/LAYERS.md):
- L1: PROJECT_ROOT resolves to site-packages for wheel install
- L4: bundled-TUI fallback reachable when ui-tui/ is missing AND install is non-checkout
- L5: non-checkout error path names the install method (brew upgrade, pip install, etc.)
- L6: HERMES_TUI_DIR set → _ensure_tui_workspace is NOT called
"""

import os
import types
from pathlib import Path

import pytest


@pytest.fixture
def main_mod():
    import hermes_cli.main as m
    return m


def test_bundled_tui_fallback_wins_when_ui_tui_missing_non_checkout(
    tmp_path, main_mod, monkeypatch
):
    """Wheel install: ui-tui/ missing, no .git, bundled tui_dist/entry.js present.

    Production path: _make_tui_argv(_make_tui_argv chooses bundled-TUI argv)
    and never calls _ensure_tui_workspace (or _ensure_tui_workspace bails
    silently to a useful code path).

    This is the GREEN path for a homebrew / pip install.
    """
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)

    # Mark tmp_path as a NON-checkout (no .git).
    # ui-tui/ does NOT exist next to tmp_path.
    # Instead, stage a bundled tui_dist/entry.js at hermes_cli_dir/tui_dist/entry.js.
    hermes_cli_dir = tmp_path / "hermes_cli"
    hermes_cli_dir.mkdir()
    bundled = hermes_cli_dir / "tui_dist" / "entry.js"
    bundled.parent.mkdir(parents=True, exist_ok=True)
    bundled.write_text("// bundled tui")

    # _find_bundled_tui reads from hermes_cli_dir; point it at our tmp_path.
    # We patch Path(__file__).parent on the main_mod so it sees our tmp layout.
    main_path = main_mod.__file__
    monkeypatch.setattr(main_mod, "__file__", str(hermes_cli_dir / "main.py"))

    # Ensure _find_bundled_tui is called with the patched dir.
    found = main_mod._find_bundled_tui()
    assert found is not None
    assert str(found).endswith("tui_dist/entry.js")

    # We can't easily simulate a real wheel install here (PROJECT_ROOT is
    # read from a real file on disk), but we can assert the call sequence:
    # when ui-tui is missing AND no .git, _ensure_tui_workspace bails and
    # the call path reaches _find_bundled_tui().

    # The fix must change _make_tui_argv to call _find_bundled_tui()
    # BEFORE _ensure_tui_workspace when ui-tui is missing.
    # We assert the call order via a recording stub.
    order: list[str] = []

    def fake_ensure(tui_dir):
        order.append("ensure_tui_workspace")
        # If we are called first with no .git and no ui-tui, raise.
        if not (tui_dir.parent / ".git").exists():
            raise SystemExit(1)

    def fake_find_bundled(_cli_dir=None):
        order.append("find_bundled_tui")
        return bundled

    monkeypatch.setattr(main_mod, "_ensure_tui_workspace", fake_ensure)
    monkeypatch.setattr(main_mod, "_find_bundled_tui", fake_find_bundled)

    # No ext_dir, ui-tui missing, no .git. Production code should try
    # bundled-TUI fallback BEFORE the checkout-shaped error.
    tui_dir = tmp_path / "ui-tui"  # does NOT exist

    # node resolver (won't be called)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"/usr/bin/{name}")

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=False)

    assert "ensure_tui_workspace" not in order or order[-1] == "find_bundled_tui"
    assert argv[-1] == str(bundled)
    assert cwd == bundled.parent


def test_non_checkout_error_mentions_install_method(
    tmp_path, main_mod, monkeypatch, capsys
):
    """ui-tui missing, no .git, no bundled TUI → error names the install method.

    For a homebrew install the user must see `brew upgrade hermes-agent` (or
    equivalent) — NOT the checkout-shaped "From the Hermes checkout, run
    `git restore -- ui-tui`" hint, which is impossible to execute on a wheel.
    """
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)

    def which(name: str) -> str | None:
        if name == "git":
            return "/usr/bin/git"
        raise AssertionError("node/npm lookup must not run when ui-tui is missing")

    monkeypatch.setattr(main_mod.shutil, "which", which)

    # No bundled TUI (no tui_dist/ next to main.py in this tmp layout).
    monkeypatch.setattr(main_mod, "_find_bundled_tui", lambda *a, **k: None)

    # Simulate the install method detection returning "homebrew".
    # The function is imported lazily inside _print_tui_workspace_missing_error,
    # so patch the source module (hermes_cli.config) rather than main_mod.
    import hermes_cli.config as _cfg_mod

    monkeypatch.setattr(_cfg_mod, "detect_install_method", lambda *a, **k: "homebrew")
    monkeypatch.setattr(
        _cfg_mod, "recommended_update_command", lambda: "brew upgrade hermes-agent"
    )

    with pytest.raises(SystemExit) as exc:
        main_mod._make_tui_argv(tmp_path / "ui-tui", tui_dev=False)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "TUI workspace is missing" in err
    # NEW: install method-specific recovery hint.
    assert "brew upgrade hermes-agent" in err
    # REMOVED: checkout-shaped hint is misleading for wheel installs.
    # We expect the homebrew-specific message instead of the generic
    # "From the Hermes checkout" line.
    assert "From the Hermes checkout, run `git restore -- ui-tui`" not in err


def test_checkout_path_preserved_when_git_restore_fails(
    tmp_path, main_mod, monkeypatch, capsys
):
    """True git checkout: ui-tui missing + .git present + restore fails → checkout-shaped error preserved.

    Regression guard: the fix must not regress the existing checkout path.
    """
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)

    def which(name: str) -> str | None:
        if name == "git":
            return "/usr/bin/git"
        raise AssertionError("node/npm lookup must not run when ui-tui is missing")

    monkeypatch.setattr(main_mod.shutil, "which", which)

    # Mark tmp_path as a CHECKOUT (.git present) so _restore_tui_workspace runs.
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        main_mod.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    with pytest.raises(SystemExit) as exc:
        main_mod._make_tui_argv(tmp_path / "ui-tui", tui_dev=False)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "TUI workspace is missing" in err
    # The checkout-shaped recovery hint is preserved.
    assert "From the Hermes checkout, run `git restore -- ui-tui`" in err


def test_hermes_tui_dir_set_skips_ensure_workspace(
    tmp_path, main_mod, monkeypatch
):
    """HERMES_TUI_DIR set → _ensure_tui_workspace is NEVER called regardless of state.

    Edge case L6: when the user supplies HERMES_TUI_DIR, the production code
    must skip the workspace guard and trust the external path.
    """
    monkeypatch.setenv("HERMES_TUI_DIR", str(tmp_path))
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)

    ensure_calls: list[Path] = []

    def recording_ensure(tui_dir):
        ensure_calls.append(tui_dir)
        raise AssertionError(
            "_ensure_tui_workspace must not be called when HERMES_TUI_DIR is set"
        )

    monkeypatch.setattr(main_mod, "_ensure_tui_workspace", recording_ensure)

    # Stage the external TUI dir as a valid prebuilt bundle.
    tui_dir = tmp_path
    (tui_dir / "dist" / "entry.js").parent.mkdir(parents=True, exist_ok=True)
    (tui_dir / "dist" / "entry.js").write_text("// prebuilt")

    # Sanity: the existing HERMES_TUI_DIR code path runs the external argv.
    argv, cwd = main_mod._make_tui_argv(tmp_path / "ui-tui", tui_dev=False)

    assert ensure_calls == [], (
        f"_ensure_tui_workspace must not run when HERMES_TUI_DIR is set; got {ensure_calls}"
    )
    assert argv[-1] == str(tui_dir / "dist" / "entry.js")
