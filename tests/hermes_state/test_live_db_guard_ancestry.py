"""The live-DB guard must survive a scrubbed child environment (#82770).

Forensic background: a read-only sweep of production ``state.db`` files found
hundreds of zero-message "open" gateway session rows carrying test-fixture
identities (``chat-1`` / ``user-1`` / ``wx-chat``), with matching
``gateway_routing`` scopes pointing at ``pytest-of-*`` temp directories.

The escape is structural, not a one-off test bug.  Hermetic isolation rides
entirely on the process environment: ``HERMES_HOME`` says *where* to write and
``PYTEST_CURRENT_TEST`` / ``PYTEST_VERSION`` say *whether the guard is armed*.
Both live in the same carrier, so a child spawned with a rebuilt environment
loses them together — it aims at the developer's real ``state.db`` and
silences the only check that would have stopped it, in one step.

Process ancestry is the signal that survives an env rebuild, so these tests
pin that the guard is armed by ancestry when the environment no longer says
"pytest".

These tests drive ``_ensure_test_isolation`` rather than constructing a real
``SessionDB``: if the guard regresses, the assertion must fail *without* the
test itself writing to the developer's live database.
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import psutil
import pytest

import hermes_state
import hermes_state_guard

REPO_ROOT = Path(__file__).resolve().parents[2]

# Probe run in the child: resolve the REAL platform state root (not a
# hardcoded ~/.hermes — that root is %LOCALAPPDATA%\hermes on Windows) and
# report whether the guard refuses it.
_CHILD_PROBE = """
import sys
sys.path.insert(0, {repo!r})
import hermes_state

root = hermes_state._real_platform_state_root()
if root is None:
    print("NO-ROOT")
else:
    try:
        hermes_state._ensure_test_isolation(root / "state.db")
    except RuntimeError:
        print("REFUSED")
    else:
        print("ALLOWED")
"""


def _scrubbed_env(**overrides):
    """The environment a rebuilt-from-scratch child spawn ends up with.

    Also strips ``HERMES_TEST_ISOLATION`` — the conftest-exported marker
    layer would otherwise arm the guard first and these tests would no
    longer prove anything about the ancestry fallback they exist to pin.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("PYTEST_")
        and k not in ("HERMES_HOME", "HERMES_TEST_ISOLATION")
    }
    env.update(overrides)
    return env


def _run_probe(env):
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROBE.format(repo=str(REPO_ROOT))],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
        check=False,
    )
    verdict = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if verdict == "NO-ROOT":
        pytest.skip("no real platform state root resolvable on this machine")
    assert verdict in ("REFUSED", "ALLOWED"), (
        f"probe produced no verdict.\nstdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}"
    )
    return verdict


class TestScrubbedChildEnvironment:
    def test_child_without_pytest_env_still_refuses_production_db(self):
        """The #82770 escape: no PYTEST_* and no HERMES_HOME, yet still a test.

        This is the exact shape of the leak — the child resolves the real
        ``state.db`` because ``HERMES_HOME`` is gone, and the env-only guard
        sees a "normal user run" because ``PYTEST_*`` is gone with it.
        """
        assert _run_probe(_scrubbed_env()) == "REFUSED"

    def test_child_inheriting_pytest_env_still_refuses_production_db(self):
        """The pre-existing env path must keep working unchanged."""
        env = dict(os.environ)
        env.pop("HERMES_HOME", None)
        env.setdefault("PYTEST_CURRENT_TEST", "tests/x.py::test_x (call)")
        assert _run_probe(env) == "REFUSED"

    def test_env_bypass_lets_a_deliberate_child_through(self):
        """A test that genuinely needs the live DB in a child can opt out.

        ``_STATE_DB_GUARD_BYPASS`` is a module global and cannot cross a
        process boundary, so ancestry-armed children need an env-carried
        escape hatch or they would have no way to opt out at all.
        """
        env = _scrubbed_env(**{hermes_state._STATE_DB_GUARD_BYPASS_ENV: "1"})
        assert _run_probe(env) == "ALLOWED"


class TestPytestProcessRecognition:
    """Unit-level checks for the ancestry predicate's matching rules."""

    class _FakeProc:
        def __init__(self, cmdline):
            self._cmdline = cmdline

        def cmdline(self):
            return self._cmdline

    @pytest.mark.parametrize(
        "cmdline",
        [
            ["/usr/bin/python", "-m", "pytest", "tests/"],
            ["/venv/bin/pytest", "-q"],
            [r"C:\venv\Scripts\pytest.exe", "-q"],
            ["/usr/bin/py.test", "tests/"],
        ],
    )
    def test_recognises_pytest_invocations(self, cmdline):
        assert hermes_state_guard._process_looks_like_pytest(self._FakeProc(cmdline))

    @pytest.mark.parametrize(
        "cmdline",
        [
            ["hermes", "gateway", "start"],
            ["/usr/bin/python", "-m", "hermes_cli.main", "sessions", "list"],
            # A path that merely *contains* "pytest" is not a pytest process:
            # tmp paths like /tmp/pytest-of-dev/... show up in real argv.
            ["hermes", "run", "--file", "/tmp/pytest-of-dev/test0/input.txt"],
        ],
    )
    def test_ignores_non_pytest_invocations(self, cmdline):
        assert not hermes_state_guard._process_looks_like_pytest(self._FakeProc(cmdline))

    def test_unreadable_process_is_not_pytest(self):
        class _Denied:
            def cmdline(self):
                raise PermissionError("access denied")

        assert not hermes_state_guard._process_looks_like_pytest(_Denied())


class TestAncestryWalkStopsAtAccessDenied:
    """The walk must not read past the first ``psutil.AccessDenied`` (#137948).

    On native Windows the chain above a user session climbs into protected
    system processes (``svchost`` → ``services`` → ``wininit``), and each denied
    ``cmdline()`` makes psutil fall back to a full process snapshot
    (``NtQuerySystemInformation``) — hundreds of ms per process, ~30 s for the
    whole walk on a slow-snapshot host, which blew the dashboard chat backend's
    15 s ``gateway.ready`` budget. No pytest of the current user can hide above
    a process it is not allowed to read, so the walk stops there.
    """

    class _Proc:
        """Ancestor stub, near-to-far order, counting ``cmdline()`` reads."""

        def __init__(self, cmdline=None, *, denied=False, error=None):
            self._cmdline = cmdline or []
            self._denied = denied
            self._error = error
            self.cmdline_reads = 0

        def cmdline(self):
            self.cmdline_reads += 1
            if self._denied:
                raise psutil.AccessDenied()
            if self._error is not None:
                raise self._error
            return self._cmdline

    @staticmethod
    def _walk(monkeypatch, procs):
        psutil_stub = types.SimpleNamespace(
            Process=lambda: types.SimpleNamespace(parents=lambda: procs),
            # The guard's except clauses resolve these through its module-global
            # psutil, so the stub must carry the real exception classes.
            AccessDenied=psutil.AccessDenied,
            NoSuchProcess=psutil.NoSuchProcess,
            Error=psutil.Error,
        )
        monkeypatch.setattr(hermes_state_guard, "psutil", psutil_stub)
        monkeypatch.setattr(hermes_state_guard, "_PYTEST_ANCESTOR", None)
        return hermes_state_guard._has_pytest_ancestor()

    def test_does_not_read_ancestors_above_an_access_denied_one(self, monkeypatch):
        above = [
            self._Proc(["C:\\Windows\\system32\\services.exe"]),
            self._Proc(["C:\\Windows\\system32\\wininit.exe"]),
        ]
        procs = [
            self._Proc(["C:\\Windows\\Explorer.EXE"]),
            self._Proc(["C:\\Windows\\System32\\svchost.exe", "-k", "netsvcs"], denied=True),
            *above,
        ]
        assert self._walk(monkeypatch, procs) is False
        assert [p.cmdline_reads for p in above] == [0, 0]

    def test_pytest_below_the_unreadable_ancestor_is_still_found(self, monkeypatch):
        """The #82770 ancestry protection must survive the early stop."""
        procs = [
            self._Proc(["/venv/bin/pytest", "-q"]),
            self._Proc(["C:\\Windows\\System32\\svchost.exe"], denied=True),
            self._Proc(["C:\\Windows\\system32\\services.exe"]),
        ]
        assert self._walk(monkeypatch, procs) is True

    def test_a_vanished_ancestor_does_not_stop_the_walk(self, monkeypatch):
        """A dead ancestor (e.g. NoSuchProcess) is skipped, not treated as a wall."""
        procs = [
            self._Proc(["/usr/bin/zsh"], error=psutil.NoSuchProcess(1234)),
            self._Proc(["/venv/bin/pytest", "-q"]),
        ]
        assert self._walk(monkeypatch, procs) is True
