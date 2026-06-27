"""Tests for follow-up fixes to the LSP integration (PR after #24168).

Covers:

1. ``typescript-language-server`` install recipe pulls in ``typescript``
   alongside the server, so the npm install command targets both.
2. ``hermes lsp status`` surfaces a ``Backend warnings`` section when
   bash-language-server is installed but ``shellcheck`` is missing.
3. ``_check_lint`` returns ``skipped`` (not ``error``) when the linter
   command exists on PATH but couldn't actually run — e.g. ``npx tsc``
   without the typescript SDK installed.  This is what unblocks the
   LSP semantic tier on TypeScript files when the user doesn't also
   have a project-level ``tsc``.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from agent.lsp.install import INSTALL_RECIPES


# ---------------------------------------------------------------------------
# Fix 1: typescript install recipe carries the typescript SDK
# ---------------------------------------------------------------------------


def test_typescript_recipe_includes_typescript_sdk():
    recipe = INSTALL_RECIPES["typescript-language-server"]
    extras = recipe.get("extra_pkgs") or []
    assert "typescript" in extras, (
        "typescript-language-server requires the `typescript` SDK as a "
        "sibling install — without it `initialize` fails with "
        "'Could not find a valid TypeScript installation'."
    )


def test_install_npm_passes_extras_to_npm_command(tmp_path, monkeypatch):
    """Verify the npm subprocess is invoked with both pkg AND extras."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Pretend npm succeeded but binary doesn't exist — install code
        # will return None, which is fine for this test.
        return MagicMock(returncode=0, stderr="")

    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(install_mod.shutil, "which", lambda c: "/usr/bin/npm" if c == "npm" else None)

    install_mod._install_npm("typescript-language-server", "typescript-language-server",
                             extra_pkgs=["typescript"])

    cmd = captured["cmd"]
    assert "typescript-language-server" in cmd
    assert "typescript" in cmd
    # Both must come AFTER the npm flags, in install-target position
    install_idx = cmd.index("install")
    assert cmd.index("typescript-language-server") > install_idx
    assert cmd.index("typescript") > install_idx


def test_install_npm_works_without_extras(tmp_path, monkeypatch):
    """Backwards compat: pyright-style recipes (no extras) still install."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stderr="")

    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(install_mod.shutil, "which", lambda c: "/usr/bin/npm" if c == "npm" else None)

    install_mod._install_npm("pyright", "pyright-langserver")

    cmd = captured["cmd"]
    assert "pyright" in cmd
    # Should not blow up when extra_pkgs is omitted/None
    install_targets = [c for c in cmd if not c.startswith("-") and c not in (
        "install", "--prefix", str(install_mod.hermes_lsp_bin_dir().parent),
        "/usr/bin/npm",
    )]
    assert install_targets == ["pyright"]


# ---------------------------------------------------------------------------
# Fix 2: ``hermes lsp status`` surfaces shellcheck-missing for bash
# ---------------------------------------------------------------------------


def test_backend_warnings_quiet_when_bash_not_installed(tmp_path, monkeypatch):
    """No bash → no warning."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import cli as lsp_cli

    with patch("shutil.which", return_value=None):
        notes = lsp_cli._backend_warnings()
    assert notes == []


def test_backend_warnings_quiet_when_bash_and_shellcheck_both_present(tmp_path, monkeypatch):
    """Both installed → no warning."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import cli as lsp_cli

    def which(name):
        return f"/usr/bin/{name}"  # both found

    with patch("shutil.which", side_effect=which):
        notes = lsp_cli._backend_warnings()
    assert notes == []


def test_backend_warnings_fires_when_bash_installed_but_shellcheck_missing(tmp_path, monkeypatch):
    """The exact scenario from the bug report."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import cli as lsp_cli

    def which(name):
        if name == "bash-language-server":
            return "/fake/bin/bash-language-server"
        return None  # shellcheck missing

    with patch("shutil.which", side_effect=which):
        notes = lsp_cli._backend_warnings()
    assert len(notes) == 1
    assert "shellcheck" in notes[0].lower()
    assert "bash-language-server" in notes[0].lower()


def test_status_output_includes_backend_warnings_section(tmp_path, monkeypatch):
    """End-to-end: status command output includes the warning section."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Pretend bash-language-server is installed but shellcheck is missing
    def which(name):
        if name == "bash-language-server":
            return "/fake/bin/bash-language-server"
        return None

    from agent.lsp import cli as lsp_cli

    buf = io.StringIO()
    with patch("shutil.which", side_effect=which), redirect_stdout(buf):
        lsp_cli._cmd_status(emit_json=False)

    output = buf.getvalue()
    assert "Backend warnings" in output
    assert "shellcheck" in output


# ---------------------------------------------------------------------------
# Fix 3: tier-1 lint treats unusable linters as ``skipped``, not ``error``
# ---------------------------------------------------------------------------


def test_npx_tsc_missing_treated_as_skipped():
    """The original bug: ``npx tsc`` errors when tsc isn't installed.

    Without this fix, the lint result is ``error``, which means the LSP
    semantic tier (gated on ``success or skipped``) is skipped — the user
    gets a useless tooling-error message instead of real diagnostics.
    """
    from tools.file_operations import _looks_like_linter_unusable

    npx_failure_output = (
        "                                                                               \n"
        "                This is not the tsc command you are looking for                \n"
        "                                                                               \n"
        "\n"
        "To get access to the TypeScript compiler, tsc, from the command line either:\n"
        "- Use npm install typescript to first add TypeScript to your project before using npx\n"
    )

    assert _looks_like_linter_unusable("npx", npx_failure_output) is True


def test_real_lint_error_not_classified_as_unusable():
    """A genuine TypeScript type error must NOT be misclassified."""
    from tools.file_operations import _looks_like_linter_unusable

    real_error = (
        "bad.ts:5:1 - error TS2322: Type 'number' is not assignable to type 'string'.\n"
        "5 const x: string = greet(42);\n"
        "  ~~~~~~~~~~~~~~~\n"
    )

    assert _looks_like_linter_unusable("npx", real_error) is False


def test_unknown_base_cmd_returns_false():
    """Unfamiliar linters fall through and use the normal error path."""
    from tools.file_operations import _looks_like_linter_unusable

    assert _looks_like_linter_unusable("eslint", "any output") is False
    assert _looks_like_linter_unusable("", "anything") is False


def test_check_lint_returns_skipped_when_npx_tsc_unusable(tmp_path):
    """Integration: _check_lint sees npx exit non-zero with the npx banner
    and returns a ``skipped`` LintResult so LSP can still run."""
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    ts_file = tmp_path / "bad.ts"
    ts_file.write_text("const x: string = 42;\n")

    env = LocalEnvironment()
    fops = ShellFileOperations(env)

    # Patch _exec to simulate ``npx tsc`` failing because tsc is missing.
    npx_banner = (
        "                                                                               \n"
        "                This is not the tsc command you are looking for                \n"
    )

    def fake_exec(cmd, **kwargs):
        result = MagicMock()
        result.exit_code = 1
        result.stdout = npx_banner
        return result

    with patch.object(fops, "_exec", side_effect=fake_exec), \
         patch.object(fops, "_has_command", return_value=True):
        lint = fops._check_lint(str(ts_file))

    assert lint.skipped is True, (
        f"expected skipped (so LSP runs); got success={lint.success}, "
        f"output={lint.output!r}"
    )
    assert "not usable" in (lint.message or "")


def test_check_lint_returns_error_for_real_ts_type_errors(tmp_path):
    """Sanity: real TypeScript errors still go through the error path."""
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    ts_file = tmp_path / "bad.ts"
    ts_file.write_text("const x: string = 42;\n")

    env = LocalEnvironment()
    fops = ShellFileOperations(env)

    real_tsc_error = (
        "bad.ts:1:7 - error TS2322: Type 'number' is not assignable to type 'string'.\n"
        "1 const x: string = 42;\n"
        "        ~\n"
        "Found 1 error.\n"
    )

    def fake_exec(cmd, **kwargs):
        result = MagicMock()
        result.exit_code = 1
        result.stdout = real_tsc_error
        return result

    with patch.object(fops, "_exec", side_effect=fake_exec), \
         patch.object(fops, "_has_command", return_value=True):
        lint = fops._check_lint(str(ts_file))

    assert lint.skipped is False
    assert lint.success is False
    assert "TS2322" in lint.output


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
