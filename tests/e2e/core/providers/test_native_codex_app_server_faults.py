"""codex_app_server faults and approval edges: real ``hermes chat -q`` against a fake ``codex app-server``.

See ``test_native_codex_app_server.py`` for the fake. Here: the app-server crashing mid-item, a failed turn,
a JSON-RPC error on ``turn/start``, a retrying ``error`` notification, and the server-initiated requests
Hermes must answer (approval in single-query mode, permissions) plus process-tree teardown.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.e2e.core.providers._native_helpers import KnownSymptom, messages, wait_until
from tests.fakes.providers.codex_app_server import CodexRun, pid_alive, run_codex_scenario

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX sh wrapper + /proc PID checks"),
    # The orphaned own-session grandchild (#121298) is released cooperatively by CodexRun.cleanup(); the
    # bypass covers its SIGKILL fallback for anything still alive (reparented to init by then).
    pytest.mark.live_system_guard_bypass,
]

KNOWN = {
    "q_approval": "#121296 approval in `chat -q` waits the full approvals.timeout instead of single_query_mode",
    "permissions": "#121297 reply to item/permissions/requestApproval omits required `permissions`",
    "orphan": "#121298 `chat -q` exit never closes the codex session; own-session descendants orphaned",
}

YOLO = ["--yolo"]
APPROVAL_TIMEOUT_S = 8


def _one(turn: dict, *, args=YOLO, config=None) -> dict:
    return dict(turns=[turn], runs=[{"prompt": "go", "args": args}], config=config)


SCENARIOS = {
    "crash": _one({"steps": [{"kind": "message_partial", "text": "PARTIAL-A-CRASH"},
                             {"kind": "crash", "code": 3, "stderr": "fatal: CRASH-MARKER-77"}]}),
    "failed": _one({"steps": [{"kind": "fail", "message": "FAIL-MARKER-88 stream disconnected before completion"}]}),
    "start_error": _one({"start_error": "START-ERR-99 model is overloaded"}),
    "retry_note": _one({"steps": [{"kind": "error_note", "message": "RECONNECT-NOTE 1/5", "will_retry": True},
                                  {"kind": "message", "text": "RETRY-OK"}]}),
    "failed_hidden": _one({"steps": [{"kind": "message", "text": "PARTIAL-B"},
                                     {"kind": "fail", "message": "FAIL-MARKER-89 stream disconnected"}]}),
    "q_approval": _one({"steps": [{"kind": "command", "command": "echo Q", "output": "Q\n"},
                                  {"kind": "message", "text": "Q-DONE"}]},
                       args=[], config={"approvals": {"timeout": APPROVAL_TIMEOUT_S}}),
    "permissions": _one({"steps": [{"kind": "permissions", "reason": "needs network"},
                                   {"kind": "message", "text": "PERM-DONE"}]}),
    "orphan": _one({"steps": [{"kind": "grandchild"}, {"kind": "message", "text": "REAP-DONE"}]}),
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Iterator[dict[str, CodexRun]]:
    with ThreadPoolExecutor(max_workers=len(SCENARIOS)) as pool:
        futures = {name: pool.submit(run_codex_scenario, tmp_path_factory.mktemp(f"codex_{name}"), **spec)
                   for name, spec in SCENARIOS.items()}
        done = {name: future.result() for name, future in futures.items()}
    yield done
    for run in done.values():
        run.cleanup()


def _assistant_texts(run: CodexRun) -> list[str]:
    return [r["content"] for r in messages(run.home, run.session_id) if r["role"] == "assistant" and r["content"]]


def test_crash_mid_item_surfaced_once_without_partial_content_or_orphan(runs):
    run = runs["crash"]
    result = run.results[0]
    assert result.returncode != 0, result.describe()
    assert run.output.count("exited unexpectedly") == 1, result.describe()
    assert run.output.count("CRASH-MARKER-77") == 1, "the app-server's stderr tail must reach the user once"
    assert "PARTIAL-A-CRASH" not in run.output, "an unfinished item must not be shown as the answer"
    assert not [t for t in _assistant_texts(run) if "PARTIAL-A-CRASH" in t], "unfinished item persisted"
    assert len(run.fake.spawned_pids()) == 1 and len(run.fake.requests("turn/start")) == 1, \
        "a crashed turn must not be replayed on a respawned app-server"
    assert not pid_alive(run.fake.spawned_pids()[0]), "app-server process survived the CLI"


@pytest.mark.parametrize("name, marker", [("failed", "FAIL-MARKER-88"), ("start_error", "START-ERR-99")])
def test_terminal_turn_error_surfaced_once_not_retried(runs, name, marker):
    run = runs[name]
    assert run.results[0].returncode != 0, run.results[0].describe()
    assert run.output.count(marker) == 1, run.results[0].describe()
    assert len(run.fake.requests("turn/start")) == 1, "non-retryable turn error was retried"
    assert not [t for t in _assistant_texts(run) if marker in t], "error text persisted as assistant content"
    run.fake.assert_wire_clean()


def test_will_retry_error_notification_is_not_terminal(runs):
    run = runs["retry_note"]
    assert run.results[0].returncode == 0, run.results[0].describe()
    assert run.results[0].stdout.strip().endswith("RETRY-OK"), run.results[0].describe()
    assert "RECONNECT-NOTE" not in run.output, "a willRetry notice is codex's own retry, not a user-facing error"
    assert len(run.fake.requests("turn/start")) == 1
    assert _assistant_texts(run) == ["RETRY-OK"]


def test_failed_turn_after_agent_message_surfaces_reason(runs):
    run = runs["failed_hidden"]
    assert run.results[0].returncode != 0 and "PARTIAL-B" in run.results[0].stdout, run.results[0].describe()
    assert "FAIL-MARKER-89" in run.output, run.results[0].describe()


@pytest.mark.xfail(strict=True, raises=KnownSymptom, reason=KNOWN["q_approval"])
def test_single_query_approval_resolves_without_waiting_for_a_human(runs):
    run = runs["q_approval"]
    entries = run.fake.entries()
    sent = [e for e in entries if e.get("dir") == "out"
            and e["msg"].get("method") == "item/commandExecution/requestApproval"]
    replies = [e for e in entries if e.get("reply_to") == "item/commandExecution/requestApproval"]
    assert len(sent) == 1 and len(replies) == 1, f"approval request/reply not exchanged once: {sent} {replies}"
    reply = replies[0]
    assert reply["msg"]["id"] == sent[0]["msg"]["id"] and not reply.get("violation"), reply
    waited = reply["t"] - sent[0]["t"]
    if waited >= APPROVAL_TIMEOUT_S / 2:
        raise KnownSymptom(f"approval parked {waited:.1f}s on a prompt nobody can answer in -q")


@pytest.mark.xfail(strict=True, raises=KnownSymptom, reason=KNOWN["permissions"])
def test_permissions_request_reply_matches_protocol(runs):
    run = runs["permissions"]
    replies = run.fake.replies_to("item/permissions/requestApproval")
    assert len(replies) == 1, f"permissions request unanswered: {replies}"
    violation = replies[0].get("violation") or ""
    if "permissions" in violation:
        raise KnownSymptom(f"PermissionsRequestApprovalResponse without `permissions`: {replies[0]}")
    assert not violation, f"invalid PermissionsRequestApprovalResponse: {replies[0]}"


@pytest.mark.xfail(strict=True, raises=KnownSymptom, reason=KNOWN["orphan"])
def test_cli_exit_reaps_app_server_descendants(runs):
    run = runs["orphan"]
    assert run.results[0].returncode == 0 and "REAP-DONE" in run.results[0].stdout, run.results[0].describe()
    children = run.fake.grandchild_pids()
    assert len(children) == 1, f"the fake must have spawned exactly one descendant: {children}"
    wait_until(lambda: not pid_alive(children[0]), 3.0,
               f"app-server descendant {children[0]} to be reaped after CLI exit", error=KnownSymptom)
