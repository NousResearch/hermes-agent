"""managed_scope's test-context predicate must answer for the SESSION, not one phase.

``PYTEST_CURRENT_TEST`` is set only while a test function runs. pytest unsets it at
collection, in any thread that outlives its test, and at interpreter shutdown — so a
predicate that reads only that var lets a real ``/etc/hermes`` managed scope back into the
suite in exactly those phases and pin admin policy onto the tests. Measured on pytest 9.x.

These cases drive the CURRENT_TEST-absent phases directly (by scrubbing the var, which is
what pytest itself does outside the in-test phase) against a stand-in for the system default.
"""
import os
from pathlib import Path

import pytest

from hermes_cli import managed_scope

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Phases where pytest leaves ``PYTEST_CURRENT_TEST`` unset but the session IS a test run:
#: module import / collection, a straggler thread, session-scoped teardown. These three are
#: faithfully stood in for by scrubbing the var from inside a test — the process state that
#: matters is identical. ``atexit`` is NOT in this list: it is the one phase where the
#: stand-in diverges (pytest tears down ``PYTEST_VERSION`` as well before interpreter
#: shutdown, which an in-test scrub leaves set), so it gets a real-shutdown case below.
_CURRENT_TEST_ABSENT_PHASES = ("collection", "straggler-thread", "sessionfinish")


@pytest.fixture
def system_managed_dir(tmp_path, monkeypatch):
    """A populated stand-in for the system ``/etc/hermes``, with no override set."""
    managed = tmp_path / "etc-hermes"
    managed.mkdir()
    (managed / "config.yaml").write_text("model: pinned/by-admin\n", encoding="utf-8")
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    monkeypatch.setattr(managed_scope, "_DEFAULT_MANAGED_DIR", managed)
    managed_scope.invalidate_managed_cache()
    yield managed
    managed_scope.invalidate_managed_cache()


@pytest.mark.parametrize("phase", _CURRENT_TEST_ABSENT_PHASES)
def test_system_scope_ignored_when_current_test_is_absent(phase, system_managed_dir, monkeypatch):
    """The system scope stays invisible in every CURRENT_TEST-absent phase of a test session."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert managed_scope._under_pytest() is True, f"{phase}: predicate lost the session"
    assert managed_scope.get_managed_dir() is None, f"{phase}: system managed scope leaked"
    assert managed_scope.managed_config_keys() == set(), f"{phase}: admin policy leaked"


def test_system_scope_ignored_in_test_phase(system_managed_dir):
    """The original in-test behaviour is unchanged."""
    assert managed_scope._under_pytest() is True
    assert managed_scope.get_managed_dir() is None


def test_explicit_override_still_wins_when_current_test_is_absent(tmp_path, monkeypatch):
    """Widening the predicate must not shadow the IT bootstrap override (it outranks the guard)."""
    override = tmp_path / "override"
    override.mkdir()
    (override / "config.yaml").write_text("model: pinned/by-it\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(override))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    managed_scope.invalidate_managed_cache()
    try:
        assert managed_scope.get_managed_dir() == override
        assert managed_scope.managed_config_keys() == {"model"}
    finally:
        managed_scope.invalidate_managed_cache()


def test_predicate_delegates_and_is_not_a_second_private_copy(monkeypatch):
    """One definition, proven at RUNTIME: swapping the guard's predicate swaps this one.

    An identity/source-text assertion would only grade the wiring's appearance. Patching the
    single definition and observing managed_scope's answer change proves the delegation is
    live — a private copy would be unaffected by the patch.
    """
    import hermes_state_guard

    monkeypatch.setattr(managed_scope, "_in_test_context", lambda: False)
    assert managed_scope._under_pytest() is False
    monkeypatch.setattr(managed_scope, "_in_test_context", lambda: True)
    assert managed_scope._under_pytest() is True

    # ...and the name it binds is the guard's, not a re-implementation.
    monkeypatch.undo()
    assert managed_scope._in_test_context is hermes_state_guard._in_test_context


def test_test_context_module_is_a_leaf():
    """The predicate module must not drag in a dependency graph.

    This is the assertion that keeps the delegation safe to make. ``hermes_state``
    transitively imports ``agent.redact``, which snapshots its enable-toggle at import
    time — so delegating *through* ``hermes_state`` would pull that snapshot forward and
    freeze the toggle before the config bridge ran. (Measured: that is exactly what broke
    tests/hermes_cli/test_redact_config_bridge.py on the fork arm of this fix, where the
    predicate has not yet been extracted.) Importing the leaf must pull neither.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import hermes_state_guard; "
        "print('redact', 'agent.redact' in sys.modules); "
        "print('state', 'hermes_state' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "redact False" in out.stdout, out.stdout
    assert "state False" in out.stdout, out.stdout


def test_system_scope_ignored_at_real_interpreter_shutdown(tmp_path):
    """The ``atexit`` phase, driven for real — not stood in for by scrubbing one env var.

    At true interpreter shutdown pytest has torn down ``PYTEST_VERSION`` as well as
    ``PYTEST_CURRENT_TEST`` (measured, pytest 9.0.2: ``ATEXIT CURRENT_TEST=None
    VERSION=None``), so every environment leg of the predicate is gone. An in-test scrub
    of ``PYTEST_CURRENT_TEST`` leaves ``PYTEST_VERSION`` set and therefore can never
    observe this; that is why this case runs a real nested pytest session instead.

    The inner session is launched through a shell that exits, so the pytest process has NO
    pytest ancestor either — ``_has_pytest_ancestor()`` walks ``parents()`` only and never
    inspects the process itself, so with env gone and ancestry False the only remaining
    signal is that pytest IS this process. Without ``_is_pytest_self()`` the managed scope
    resolves at shutdown and admin policy leaks into the suite.
    """
    import json
    import subprocess
    import sys
    import time

    managed = tmp_path / "etc-hermes"
    managed.mkdir()
    (managed / "config.yaml").write_text("model: pinned/by-admin\n", encoding="utf-8")
    out_path = tmp_path / "atexit.json"

    (tmp_path / "test_inner_atexit.py").write_text(
        "import atexit, json, os, pathlib, sys\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "from hermes_cli import managed_scope\n"
        "def _report():\n"
        "    json.dump({\n"
        "        'current_test': os.environ.get('PYTEST_CURRENT_TEST'),\n"
        "        'pytest_version': os.environ.get('PYTEST_VERSION'),\n"
        "        'isolation': os.environ.get('HERMES_TEST_ISOLATION'),\n"
        "        'ancestor': __import__('hermes_state_guard')._has_pytest_ancestor(),\n"
        "        'predicate': managed_scope._under_pytest(),\n"
        "        'managed_dir': str(managed_scope.get_managed_dir()),\n"
        f"    }}, open({str(out_path)!r}, 'w'))\n"
        "def test_register():\n"
        f"    managed_scope._DEFAULT_MANAGED_DIR = pathlib.Path({str(managed)!r})\n"
        "    managed_scope.invalidate_managed_cache()\n"
        "    atexit.register(_report)\n",
        encoding="utf-8",
    )

    env = {k: v for k, v in os.environ.items()}
    for leaked in ("PYTEST_CURRENT_TEST", "PYTEST_VERSION", "HERMES_TEST_ISOLATION",
                   "HERMES_MANAGED_DIR"):
        env.pop(leaked, None)

    # DETACH the inner session. A plain subprocess.run would leave THIS pytest in its
    # ancestor chain — `psutil.Process().parents()` walks the whole chain, not just the
    # immediate parent — and the ancestry leg would answer the question the self leg is
    # supposed to answer, making this case vacuous (measured: the mutation below did not
    # bite until the run was detached). Backgrounding it and letting the intermediate
    # shell exit reparents the inner pytest away from us, which is the real shape of
    # `python -m pytest` typed at a terminal: pytest SELF, no pytest ANCESTOR.
    inner = (
        f"nohup {sys.executable} -m pytest -q -p no:cacheprovider "
        f"test_inner_atexit.py >inner.log 2>&1 &"
    )
    subprocess.run(["sh", "-c", inner], cwd=str(tmp_path), env=env, timeout=120)

    deadline = time.time() + 300
    while time.time() < deadline and not out_path.exists():
        time.sleep(0.25)

    inner_log = tmp_path / "inner.log"
    assert out_path.exists(), (
        "inner session never reached its atexit handler: "
        + (inner_log.read_text(encoding="utf-8") if inner_log.exists() else "<no log>")
    )
    rec = json.loads(out_path.read_text(encoding="utf-8"))
    # Pin the premise: this really is the all-env-legs-absent phase.
    assert rec["current_test"] is None, rec
    assert rec["pytest_version"] is None, rec
    assert rec["isolation"] is None, rec
    # ...and the ancestry leg is genuinely unavailable here, so this case grades the self
    # leg rather than silently riding on the outer session's process tree.
    assert rec["ancestor"] is False, f"atexit: not detached, ancestry answered instead: {rec}"
    # ...and the predicate still holds the session.
    assert rec["predicate"] is True, f"atexit: predicate lost the session: {rec}"
    assert rec["managed_dir"] == "None", f"atexit: system managed scope leaked: {rec}"


def test_self_check_matches_launcher_position_not_any_pytest_token(monkeypatch):
    """``pytest`` appearing as an ARGUMENT must not arm the predicate.

    The ancestry matcher scans every cmdline token, which is right for a parent (whatever
    its shape, a pytest is above us). Applying that rule to SELF would be wrong: a real
    invocation like ``hermes exec pytest -q`` would declare itself a test run and silently
    suppress the user's own managed scope. The self check matches by position — argv[0]'s
    basename or an explicit ``-m pytest``.
    """
    import hermes_state_guard

    def _self(argv, cmdline):
        monkeypatch.setattr(hermes_state_guard, "_PYTEST_SELF", None)
        monkeypatch.setattr(hermes_state_guard.sys, "argv", argv)
        monkeypatch.setattr(
            hermes_state_guard, "psutil",
            type("P", (), {"Process": staticmethod(lambda: type("Q", (), {"cmdline": staticmethod(lambda: cmdline)})())}),
        )
        return hermes_state_guard._is_pytest_self()

    # Launcher shapes -> True
    assert _self(["/usr/bin/pytest", "-q"], ["/usr/bin/pytest", "-q"]) is True
    assert _self(["/x/pytest/__main__.py"], ["/usr/bin/python3", "-m", "pytest", "-q"]) is True
    # Argument shapes -> False
    assert _self(["/usr/local/bin/hermes", "exec", "pytest", "-q"],
                 ["/usr/local/bin/hermes", "exec", "pytest", "-q"]) is False
    assert _self(["/usr/local/bin/hermes"], ["/usr/bin/python3", "-m", "hermes_cli", "pytest"]) is False
    # Plain production run -> False
    assert _self(["hermes", "--version"], ["/usr/bin/python3", "hermes", "--version"]) is False
