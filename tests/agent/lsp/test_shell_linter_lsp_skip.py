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

  - When LSP is active AND ``enabled_for(path)`` for a ``.ts`` / ``.go``
    / ``.rs`` file, ``_check_lint`` returns ``skipped`` without invoking
    the shell linter at all.
  - When LSP is inactive or disabled-for-path, the shell linter runs
    exactly as before (regression guard for the default config).
  - The skip only applies to extensions in
    ``_SHELL_LINTER_LSP_REDUNDANT`` — Python ``py_compile`` and
    ``node --check`` keep running unconditionally because they're fast,
    file-local, and correct.
  - ``.tsx`` is intentionally NOT in either ``LINTERS`` or
    ``_SHELL_LINTER_LSP_REDUNDANT``: it had no ``LINTERS`` entry
    pre-PR (so it was already implicitly ``skipped`` via the
    ``ext not in LINTERS`` branch) and adding one would have inherited
    ``.ts``'s broken ``tsc --noEmit FILE`` invocation for LSP-disabled
    users.  When LSP IS enabled, ``.tsx`` is still covered by
    typescript-language-server via ``_maybe_lsp_diagnostics`` — the
    diagnostics show up on ``lsp_diagnostics``, not ``lint``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_fops():
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    return ShellFileOperations(LocalEnvironment())


@pytest.mark.parametrize("ext", [".ts", ".go", ".rs"])
def test_shell_linter_skipped_when_lsp_will_handle(ext, tmp_path):
    """When LSP is active and enabled_for(path), shell linter is skipped.

    The shell linter's _exec must NOT be called — that's the whole
    point.  We assert by patching ``_exec`` to raise, so any accidental
    invocation surfaces as a test failure.
    """
    fops = _make_fops()
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










def test_lsp_will_handle_swallows_enabled_for_exception(tmp_path):
    """A flaky LSP service must never break the shell-linter fallback —
    if ``enabled_for`` raises, we treat the file as "not handled" so the
    shell linter still runs."""
    fops = _make_fops()
    src = tmp_path / "foo.ts"
    src.write_text("const x = 1\n")

    fake_svc = MagicMock()
    fake_svc.enabled_for.side_effect = RuntimeError("server crashed")

    with patch.object(fops, "_lsp_local_only", return_value=True), \
         patch("agent.lsp.get_service", return_value=fake_svc):
        assert fops._lsp_will_handle(str(src)) is False




def test_tsx_default_check_lint_returns_skipped(tmp_path):
    """End-to-end: ``.tsx`` files get ``LintResult(skipped=True)`` from
    ``_check_lint`` regardless of LSP status — this is the no-regression
    contract that addresses Copilot review #3271017282."""
    fops = _make_fops()
    src = tmp_path / "foo.tsx"
    src.write_text("export const X = () => <div/>\n")

    # Even with LSP claiming the file, no shell linter runs for .tsx
    # because there's no LINTERS entry — the ``ext not in LINTERS``
    # branch fires before the LSP short-circuit is consulted.
    with patch.object(fops, "_lsp_will_handle", return_value=True), \
         patch.object(fops, "_exec") as exec_mock:
        result = fops._check_lint(str(src))

    assert result.skipped is True
    assert not exec_mock.called, "no shell linter should run for .tsx"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
