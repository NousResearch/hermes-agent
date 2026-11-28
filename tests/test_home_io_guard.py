"""Regression tests for tests/home_io_guard.py's interpreter-prefix exemption.

The bug: only the RESOLVED spelling of each interpreter prefix was exempt, so on
a PM-managed runtime — ``sys.base_prefix`` is a version symlink
(``cpython-<minor> -> cpython-<full-version>``) — imports and traceback
formatting reading stdlib sources under the UNRESOLVED spelling were refused,
aborting a linked-worktree run while formatting its first failure.

These tests construct that symlinked-prefix scenario themselves instead of
reading the host interpreter's layout: the unresolved-spelling test discriminates
against the base guard on every machine (a host whose ``sys.base_prefix`` has no
symlink cannot provide a spelling difference, which is exactly why reading the
real layout made it vacuous), and the protected root is an ordinary directory
under ``tmp_path`` — never the real interpreter's parent, which lands on a
filesystem root for system-wide installs and made the negative test
layout-dependent.
"""
import importlib.util
from pathlib import Path
import sys

import pytest


def _module_with_symlinked_base_prefix(monkeypatch, tmp_path):
    """A private copy of the guard module whose import-time prefix computation
    saw a symlinked ``sys.base_prefix`` — the PM-runtime shape. Loaded under a
    distinct module name so the session's live guard instance (and every other
    test in the process) keeps the real interpreter's prefixes. Returns the
    fresh module and the symlink spelling."""
    target = tmp_path / "cpython-3.11.15"
    target.mkdir()
    link = tmp_path / "cpython-3.11"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"cannot create symlinks on this system: {error}")
    monkeypatch.setattr(sys, "base_prefix", str(link))
    spec = importlib.util.spec_from_file_location(
        "_home_io_guard_under_test", Path(__file__).with_name("home_io_guard.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_home_io_guard_under_test", module)
    spec.loader.exec_module(module)
    return module, link


def test_unresolved_interpreter_prefix_paths_are_exempt(monkeypatch, tmp_path):
    """A path spelled through ``sys.base_prefix``'s own (symlinked) form must
    be exempt, not only the resolved spelling: refusing the interpreter's own
    files turns a worktree run into an infrastructure-looking crash instead of
    a report. Fails against the base guard on every machine — the fixture
    builds the symlink itself rather than relying on this host's runtime."""
    module, link = _module_with_symlinked_base_prefix(monkeypatch, tmp_path)
    guard = module.HomeIOGuard(lambda: [tmp_path], lambda: ())
    guard.check(str(link / "lib" / "os.py"))  # must not raise AssertionError


def test_resolved_interpreter_prefix_paths_stay_exempt(monkeypatch, tmp_path):
    """The resolved spelling stays exempt — the only form the base guard
    covered, so this pins that the widening did not drop it."""
    module, link = _module_with_symlinked_base_prefix(monkeypatch, tmp_path)
    guard = module.HomeIOGuard(lambda: [tmp_path], lambda: ())
    guard.check(str((link / "lib" / "os.py").resolve()))


def test_non_interpreter_paths_under_root_still_refused(monkeypatch, tmp_path):
    """Exempting both interpreter spellings must not neuter the guard: the
    protected root CONTAINS the interpreter (the default-install shape — a PM
    runtime living under the guarded hermes home), yet a state file under that
    same root is still refused. Rooted at ``tmp_path`` — an ordinary directory
    on every layout, unlike ``sys.base_prefix``'s parent, which is a filesystem
    root for some system-wide installs (and there ``_within``'s drive-root
    string join refuses nothing)."""
    module = _module_with_symlinked_base_prefix(monkeypatch, tmp_path)[0]
    guard = module.HomeIOGuard(lambda: [tmp_path], lambda: ())
    with pytest.raises(AssertionError, match="TEST BUG"):
        guard.check(str(tmp_path / "not-the-interpreter" / "state.yaml"))
