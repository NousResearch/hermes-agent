"""Skip the per-file shell linter when LSP will handle the same file.

The per-file ``npx tsc --noEmit FILE.ts`` shell linter cannot see
``tsconfig.json`` (a documented ``tsc`` quirk: explicit file args bypass
the project config), so it defaults to no-lib / ES5 and floods the
agent's lint field with phantom "Cannot find 'Promise' / 'Map' / 'Set' /
'ReadonlySet' / 'Iterable' / 'imul' / …" errors on every edit — up to
25K tokens per patch.  The LSP tier (``tsserver`` via
typescript-language-server) reads tsconfig correctly and surfaces real
diagnostics in the ``lsp_diagnostics`` field of the WriteResult /
PatchResult.

These tests pin the contract:

  - When LSP is active AND ``enabled_for(path)`` for a ``.ts`` / ``.tsx``
    / ``.go`` / ``.rs`` file, ``_check_lint`` returns ``skipped`` without
    invoking the shell linter at all.
  - When LSP is inactive or disabled-for-path, the shell linter runs
    exactly as before (regression guard for the default config).
  - The skip only applies to extensions in
    ``_SHELL_LINTER_LSP_REDUNDANT`` — Python ``py_compile`` and
    ``node --check`` keep running unconditionally because they're fast,
    file-local, and correct.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_fops(tmp_path):
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    return ShellFileOperations(LocalEnvironment())


@pytest.mark.parametrize("ext", [".ts", ".tsx", ".go", ".rs"])
def test_shell_linter_skipped_when_lsp_will_handle(ext, tmp_path):
    """When LSP is active and enabled_for(path), shell linter is skipped.

    The shell linter's _exec must NOT be called — that's the whole
    point.  We assert by patching ``_exec`` to raise, so any accidental
    invocation surfaces as a test failure.
    """
    fops = _make_fops(tmp_path)
    src = tmp_path / f"bad{ext}"
    src.write_text("intentionally invalid content\n")

    def _exec_must_not_run(*args, **kwargs):  # pragma: no cover
        raise AssertionError(
            "shell linter was invoked despite LSP claiming the file"
        )

    with patch.object(fops, "_lsp_will_handle", return_value=True), \
         patch.object(fops, "_exec", side_effect=_exec_must_not_run), \
         patch.object(fops, "_has_command", return_value=True):
        result = fops._check_lint(str(src))

    assert result.skipped is True
    assert "LSP" in (result.message or "")


@pytest.mark.parametrize("ext", [".ts", ".tsx", ".go", ".rs"])
def test_shell_linter_runs_when_lsp_inactive(ext, tmp_path):
    """When LSP is inactive (default config, no service, remote backend, ...),
    the shell linter runs as before — no behavior change."""
    fops = _make_fops(tmp_path)
    src = tmp_path / f"clean{ext}"
    src.write_text("// content\n")

    fake_result = MagicMock()
    fake_result.exit_code = 0
    fake_result.stdout = ""

    with patch.object(fops, "_lsp_will_handle", return_value=False), \
         patch.object(fops, "_exec", return_value=fake_result) as exec_mock, \
         patch.object(fops, "_has_command", return_value=True):
        result = fops._check_lint(str(src))

    # _exec must have been called — proving the shell linter ran.
    assert exec_mock.called, "shell linter did NOT run when LSP was inactive"
    assert result.success is True


@pytest.mark.parametrize("ext", [".py", ".js"])
def test_lsp_does_not_skip_non_redundant_extensions(ext, tmp_path):
    """``py_compile`` and ``node --check`` keep running even when an LSP
    server (pyright/pylsp/typescript-language-server-for-JS) is active —
    they're fast, file-local, and correct, so there's no upside to
    suppressing them.
    """
    fops = _make_fops(tmp_path)
    src = tmp_path / f"clean{ext}"
    src.write_text("# valid\n" if ext == ".py" else "// valid\n")

    fake_result = MagicMock()
    fake_result.exit_code = 0
    fake_result.stdout = ""

    # Even with LSP claiming the file, the shell linter must still run
    # for these extensions.
    with patch.object(fops, "_lsp_will_handle", return_value=True), \
         patch.object(fops, "_exec", return_value=fake_result) as exec_mock, \
         patch.object(fops, "_has_command", return_value=True):
        fops._check_lint(str(src))

    assert exec_mock.called, (
        f"shell linter for {ext} did not run despite being in the "
        "'always-run' set (py_compile / node --check)"
    )


def test_lsp_will_handle_returns_false_when_service_is_none(tmp_path):
    """``_lsp_will_handle`` must return False when the LSP service hasn't
    been initialized — otherwise we'd accidentally skip the shell linter
    on systems where LSP isn't configured at all."""
    fops = _make_fops(tmp_path)
    src = tmp_path / "foo.ts"
    src.write_text("const x = 1\n")

    with patch.object(fops, "_lsp_local_only", return_value=True), \
         patch("agent.lsp.get_service", return_value=None):
        assert fops._lsp_will_handle(str(src)) is False


def test_lsp_will_handle_returns_false_on_remote_backend(tmp_path):
    """LSP servers run on the host process — remote backends (Docker,
    SSH, Modal, …) keep files inside the sandbox where the host LSP
    can't reach them.  ``_lsp_will_handle`` must short-circuit before
    calling into the service in that case."""
    fops = _make_fops(tmp_path)
    src = tmp_path / "foo.ts"
    src.write_text("const x = 1\n")

    with patch.object(fops, "_lsp_local_only", return_value=False), \
         patch("agent.lsp.get_service") as get_service_mock:
        result = fops._lsp_will_handle(str(src))

    assert result is False
    # Importantly: we never even consulted the service.
    assert not get_service_mock.called


def test_lsp_will_handle_swallows_enabled_for_exception(tmp_path):
    """A flaky LSP service must never break the shell-linter fallback —
    if ``enabled_for`` raises, we treat the file as "not handled" so the
    shell linter still runs."""
    fops = _make_fops(tmp_path)
    src = tmp_path / "foo.ts"
    src.write_text("const x = 1\n")

    fake_svc = MagicMock()
    fake_svc.enabled_for.side_effect = RuntimeError("server crashed")

    with patch.object(fops, "_lsp_local_only", return_value=True), \
         patch("agent.lsp.get_service", return_value=fake_svc):
        assert fops._lsp_will_handle(str(src)) is False


def test_tsx_now_in_linters_table():
    """Regression: ``.tsx`` was missing from ``LINTERS`` before this PR,
    so TypeScript React files got no post-edit syntax check at all.
    Now they get the same ``npx tsc --noEmit`` linter as ``.ts`` (which
    is then skipped when LSP claims the file)."""
    from tools.file_operations import LINTERS, _SHELL_LINTER_LSP_REDUNDANT

    assert ".tsx" in LINTERS
    assert LINTERS[".tsx"] == LINTERS[".ts"]
    assert ".tsx" in _SHELL_LINTER_LSP_REDUNDANT


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
