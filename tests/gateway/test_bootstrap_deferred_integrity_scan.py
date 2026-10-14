"""start_gateway runs the unclean-exit state.db scan off the event loop, after the runner is live."""
import ast
import inspect
import threading

import pytest

from gateway import run_bootstrap


def _config(monkeypatch, agent):
    from hermes_cli import config
    monkeypatch.setattr(config, "read_raw_config_readonly", lambda: {"agent": agent})


@pytest.mark.parametrize("raw, expected", [
    (None, 1800.0), (600, 600.0), ("90", 90.0), (0.5, 0.5),
    (0, 1800.0), (-5, 1800.0), (float("inf"), 1800.0), ("1e999", 1800.0), ("nan", 1800.0),
    ("soon", 1800.0), (True, 1800.0), ([], 1800.0),
])
def test_deadline_is_finite_and_positive(monkeypatch, raw, expected):
    _config(monkeypatch, {} if raw is None else {"gateway_state_db_integrity_deadline_s": raw})
    assert run_bootstrap._state_db_integrity_deadline_s() == expected


def test_deadline_survives_unreadable_config(monkeypatch):
    from hermes_cli import config

    def broken():
        raise OSError("unreadable")
    monkeypatch.setattr(config, "read_raw_config_readonly", broken)
    assert run_bootstrap._state_db_integrity_deadline_s() == 1800.0


def test_scan_runs_on_a_daemon_thread_with_the_configured_deadline(monkeypatch):
    from gateway import lifecycle_ledger as ledger
    _config(monkeypatch, {"gateway_state_db_integrity_deadline_s": 42})
    seen, release, done = {}, threading.Event(), threading.Event()

    def scan(evidence, *, deadline_s):
        seen.update(evidence=evidence, deadline_s=deadline_s, thread=threading.current_thread())
        release.wait(5)  # still running when the caller returns: nothing waits on it
        done.set()
    monkeypatch.setattr(ledger, "run_deferred_integrity_check", scan)

    evidence = {"prior_pid": 1234}
    run_bootstrap._start_deferred_state_db_integrity_check(evidence)
    assert not done.is_set()
    release.set()
    assert done.wait(5)
    assert seen["evidence"] is evidence and seen["deadline_s"] == 42.0
    assert seen["thread"].daemon and seen["thread"].name == "state-db-integrity-check"


def test_start_gateway_starts_the_scan_only_after_the_runner_is_live():
    """The listeners answer from ``start_gateway_runtime_api`` on; the scan must not run before
    ``_start_gateway_run_runner`` returns, and only for an unclean previous exit."""
    tree = ast.parse(inspect.getsource(run_bootstrap.start_gateway))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    line = {node.func.id: node.lineno for node in calls}
    assert line["_start_deferred_state_db_integrity_check"] > line["_start_gateway_run_runner"]
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If) and any(
        isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_start_deferred_state_db_integrity_check"
        for c in ast.walk(node)))
    assert ast.unparse(guard.test) == "_unclean_exit is not None"
