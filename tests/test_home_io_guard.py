"""Regression tests for tests/home_io_guard.py's interpreter-prefix exemption."""
from pathlib import Path
import sys

import pytest

from tests.home_io_guard import HomeIOGuard


def _guard_rooted_at_interpreter() -> HomeIOGuard:
    """A guard whose protected root CONTAINS the running interpreter — the
    default-install shape, where the PM-managed runtime lives under ~/.hermes,
    which the guard protects."""
    return HomeIOGuard(lambda: [Path(sys.base_prefix).parent], lambda: ())


def test_unresolved_interpreter_prefix_paths_are_exempt():
    """Paths spelled through sys.base_prefix's own (possibly symlinked) form
    must be exempt, not only the resolved spelling: a PM runtime exposes the
    interpreter as cpython-<minor> -> cpython-<full-version>, imports and
    traceback formatting read stdlib sources through the unresolved form, and
    refusing those reads aborts a worktree test run instead of letting it
    report.

    Only discriminating where the two spellings differ (symlinked PM
    runtimes — where the guard used to refuse); vacuous where
    ``sys.base_prefix`` has no symlink, since both spellings then coincide
    and were already exempt."""
    guard = _guard_rooted_at_interpreter()
    unresolved = Path(sys.base_prefix) / "lib" / "os.py"
    guard.check(str(unresolved))  # must not raise AssertionError


def test_resolved_interpreter_prefix_paths_stay_exempt():
    """The resolved spelling stays exempt (this was the only covered form)."""
    guard = _guard_rooted_at_interpreter()
    resolved = Path(sys.base_prefix).resolve() / "lib" / "os.py"
    guard.check(str(resolved))


def test_non_interpreter_paths_under_root_still_refused():
    """Exempting both interpreter spellings must not neuter the guard: a
    state file under the same root is still refused."""
    guard = _guard_rooted_at_interpreter()
    state_file = Path(sys.base_prefix).parent / "not-the-interpreter" / "state.yaml"
    with pytest.raises(AssertionError, match="TEST BUG"):
        guard.check(str(state_file))
