"""Run a deterministic, machine-readable subagent lifecycle contract eval.

This harness deliberately uses only the public service methods for lifecycle
observations.  The fake child/executor stand in for the model and scheduler so
the eval does not need credentials, wall-clock sleeps, or a live provider.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# When this file is executed directly (`python evals/subagent_lifecycle/runner.py`),
# the script directory is on sys.path, not the repo root.  Insert the repo root
# first so the ``agent`` package is importable without the caller setting PYTHONPATH.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
)

EVAL_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = EVAL_DIR / "fixture.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def transition(state: str, event: str, fixture: dict | None = None) -> str:
    """Apply the documented state graph, rejecting edges not in the contract."""
    fixture = fixture or load_fixture()
    for edge in fixture["transitions"]:
        if edge["from"] == state and edge["event"] == event:
            return edge["to"]
    raise ValueError(f"invalid lifecycle transition: {state} + {event}")


class _Child:
    def __init__(self, subagent_id: str):
        self._subagent_id = subagent_id
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "eval-provider"
        self.model = "eval-model"
        self.interrupted = False

    def hard_interrupt(self, _reason, *, tool_reason=None):
        del tool_reason
        self.interrupted = True


class _ControlledExecutor:
    """Queue submissions and start them only when the scenario asks."""

    def __init__(self):
        self.pending: list[tuple[object, tuple, Future]] = []
        self.thread: threading.Thread | None = None

    def submit(self, fn, *args):
        future = Future()
        self.pending.append((fn, args, future))
        return future

    def start_next(self):
        fn, args, future = self.pending.pop(0)

        def invoke():
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # pragma: no cover - defensive
                future.set_exception(exc)

        self.thread = threading.Thread(target=invoke, daemon=True)
        self.thread.start()

    def join(self):
        if self.thread is not None:
            self.thread.join(timeout=2)
            assert not self.thread.is_alive()


def _run_service_cases() -> dict[str, object]:
    parent = SimpleNamespace(session_id="eval-parent", enabled_toolsets=["file"])
    executor = _ControlledExecutor()
    children: dict[str, _Child] = {}
    gates: dict[str, tuple[threading.Event, threading.Event]] = {}
    counter = iter(range(1, 100))

    def build(**_kwargs):
        ident = f"eval-{next(counter)}"
        child = _Child(ident)
        children[ident] = child
        return child

    def run(_index, goal, child, _parent):
        running, release = gates[goal]
        running.set()
        if goal == "failure":
            release.wait(5)
            return {"status": "failed", "error": "fixture failure", "api_calls": 1}
        if goal == "interruption":
            # A child-reported interruption is distinct from cancellation.
            release.wait(5)
            return {"status": "interrupted", "api_calls": 0}
        if goal == "cancel":
            # Cooperative cancellation: cancel() flips child.interrupted before
            # the release gate is opened, so the terminal mapping is stable.
            release.wait(5)
            return {"status": "interrupted" if child.interrupted else "completed", "api_calls": 0}
        if goal == "timeout":
            # The child blocks until the fixture releases it; the caller's wait
            # timeout therefore fires while the child is still RUNNING.  The
            # bound (5s) is a safety net only — the fixture releases the child
            # immediately after the caller's timeout, and "timeout" maps to
            # FAILED under the service's status->state mapping.
            release.wait(5)
            return {"status": "timeout", "error": "fixture timeout", "api_calls": 0}
        release.wait(5)
        return {"status": "completed", "summary": "fixture success", "api_calls": 1}

    observations: dict[str, object] = {}
    with patch("agent.subagent_lifecycle._EXECUTOR", executor), patch(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    ), patch("tools.delegate_tool._run_child_lifecycle", run):
        service = SubagentLifecycleService(lambda: parent)

        def launch(goal: str):
            running, release = threading.Event(), threading.Event()
            gates[goal] = (running, release)
            handle = service.launch(SubagentLaunchRequest(goal=goal))
            assert service.status(handle).state is SubagentState.PENDING
            executor.start_next()
            assert running.wait(5)
            assert service.status(handle).state is SubagentState.RUNNING
            return handle, release

        success, success_release = launch("success")
        success_release.set()
        assert service.wait(success, timeout_seconds=1).state is SubagentState.SUCCEEDED
        observations["success"] = service.result(success).terminal_state.value
        executor.join()

        failure, failure_release = launch("failure")
        failure_release.set()
        assert service.wait(failure, timeout_seconds=1).state is SubagentState.FAILED
        observations["failure"] = service.result(failure).error_classification
        executor.join()

        interruption, interruption_release = launch("interruption")
        interruption_release.set()
        assert service.wait(interruption, timeout_seconds=1).state is SubagentState.INTERRUPTED
        observations["interruption"] = service.result(interruption).terminal_state.value
        executor.join()

        cancellation, cancellation_release = launch("cancel")
        cancel = service.cancel(cancellation, reason="fixture cancellation")
        assert cancel.accepted and cancel.state is SubagentState.CANCEL_REQUESTED
        cancellation_release.set()
        assert service.wait(cancellation, timeout_seconds=1).state is SubagentState.CANCELLED
        observations["cancellation"] = {
            "requested": cancel.state.value,
            "terminal": service.result(cancellation).terminal_state.value,
        }
        executor.join()

        timeout, timeout_release = launch("timeout")
        # The caller's wait timeout fires while the child is still RUNNING; the
        # child is blocked on the release gate, so this is deterministic.
        timed = service.wait(timeout, timeout_seconds=0.001)
        assert timed.timed_out and not timed.completed and timed.state is SubagentState.RUNNING
        # After the caller's timeout, the fixture deterministically drives the
        # child to a terminal state: it reports a timeout and the service marks
        # it FAILED.
        timeout_release.set()
        assert service.wait(timeout, timeout_seconds=1).state is SubagentState.FAILED
        observations["timeout"] = {"timed_out": timed.timed_out, "terminal": service.result(timeout).terminal_state.value}
        executor.join()

        forged = dataclasses.replace(success, capability="invalid")
        assert service.status(forged).state is SubagentState.UNKNOWN
        assert service.result(forged).error_classification == "UNKNOWN_HANDLE"
        observations["unknown_handle"] = service.status(forged).state.value

        try:
            service.launch(SubagentLaunchRequest(goal="", role="invalid"))
        except SubagentLifecycleError:
            observations["invalid_request"] = True
        else:  # pragma: no cover - assertion gives a useful contract failure
            raise AssertionError("invalid request was accepted")
    return observations


def run_eval() -> dict:
    fixture = load_fixture()
    checks = []
    for edge in fixture["transitions"]:
        checks.append({"name": f"{edge['from']}+{edge['event']}", "ok": transition(edge["from"], edge["event"], fixture) == edge["to"]})
    for edge in fixture["invalid_transitions"]:
        try:
            transition(edge["from"], edge["event"], fixture)
        except ValueError:
            checks.append({"name": f"invalid:{edge['from']}+{edge['event']}", "ok": True})
        else:
            checks.append({"name": f"invalid:{edge['from']}+{edge['event']}", "ok": False})
    observations = _run_service_cases()
    expected = {"success": "SUCCEEDED", "failure": "FAILED", "interruption": "INTERRUPTED", "cancellation": {"requested": "CANCEL_REQUESTED", "terminal": "CANCELLED"}, "timeout": {"timed_out": True, "terminal": "FAILED"}, "unknown_handle": "UNKNOWN", "invalid_request": True}
    checks.extend({"name": f"service:{key}", "ok": observations.get(key) == value} for key, value in expected.items())
    return {"contract": "subagent_lifecycle", "contract_version": fixture["contract_version"], "ok": all(c["ok"] for c in checks), "checks": checks, "observations": observations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    args = parser.parse_args()
    print(json.dumps(run_eval(), sort_keys=True, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
