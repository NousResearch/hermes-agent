"""Regression test: a symlinked interpreter home must not be refused as real-home I/O.

The guard exempts the interpreter's own installation (stdlib source reads from
linecache and traceback formatting) by resolved prefix. When a PM-managed runtime's
lexical path differs from its resolved path — a symlinked generation directory under
the default install (repo checked out inside the Hermes home) — the lexical root
refusal fired before the resolved-prefix exemption. A *failing* test then crashed
its own failure report with ``KeyError: <_pytest.stash.StashKey object>`` instead of
printing the failure, because formatting the traceback reads the interpreter's
stdlib source from the unresolved path.
"""

from __future__ import annotations

import os

from tests import home_io_guard
from tests.home_io_guard import HomeIOGuard


def test_symlinked_interpreter_prefix_is_not_refused(tmp_path, monkeypatch):
    # A guarded "home" containing a symlinked interpreter installation whose
    # lexical name differs from its resolved name.
    home = tmp_path / "fake-home"
    interp = home / "runtime" / "generation-1"
    stdlib = interp / "lib" / "python3.11"
    stdlib.mkdir(parents=True)
    source = stdlib / "pathlib.py"
    source.write_text("# stdlib source\n")
    link = home / "runtime" / "generation-link"
    link.symlink_to(interp, target_is_directory=True)

    # The prefix table is resolved once at import; point the fast-path strings at
    # the resolved form of our fake installation (as a real runtime would be).
    monkeypatch.setattr(
        home_io_guard,
        "_INTERPRETER_PREFIX_STRS",
        (os.path.normcase(str(interp.resolve())),),
    )

    guard = HomeIOGuard(roots=lambda: [str(home)])

    # Lexically under the guarded root, but it resolves into the interpreter
    # installation: this is the stdlib read that used to crash failure formatting.
    guard.check(str(link / "lib" / "python3.11" / "pathlib.py"))

    # A genuinely foreign path under the same root is still refused.
    foreign = home / "state" / "config.yaml"
    foreign.parent.mkdir()
    foreign.write_text("x: 1\n")
    try:
        guard.check(str(foreign))
    except AssertionError as exc:
        assert "REAL hermes home" in str(exc)
    else:  # pragma: no cover - the guard must refuse foreign home I/O
        raise AssertionError("guard did not refuse foreign home I/O")
