"""Copilot ACP wire conformance, part 2: ACP error responses and agent crashes.

The agent (``tests/fakes/providers/copilot_acp.py``) answers ``session/prompt`` with JSON-RPC errors from
the ACP / JSON-RPC 2.0 error space or dies mid-stream. Each row drives one real ``hermes chat -q`` and
asserts the retry semantics the user sees:

* transient failures (``-32603`` internal error, a crash after a partial chunk) are retried with a
  fresh agent process and the turn then succeeds, with none of the failed attempt's text persisted;
* ``-32000 Authentication required`` is not retried: surfaced once, one model call;
* persistent failures (``-32602`` invalid params, repeated crashes) stop after the configured retry
  budget (``agent.api_max_retries: 2``) and are surfaced ONCE — never a loop, never a duplicate;
* no agent process survives the CLI in any row.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core._pending_fixes import known_gate
from tests.e2e.core.providers._native_helpers import (
    ChatResult,
    KnownSymptom,
    NativeHome,
    latest_session,
    make_home,
    messages,
    run_chat,
    wait_until,
)
from tests.fakes.providers import copilot_acp as acp

pytest.importorskip("acp.schema", reason="the fake validates against the agent-client-protocol package (acp extra)")

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX launcher script + /proc pid checks"),
    pytest.mark.live_system_guard_bypass,
]

API_MAX_RETRIES = 2  # _native_helpers.make_home pins agent.api_max_retries
OK_TEXT = "RECOVERED-ANSWER-OK"
PARTIAL = "PARTIAL-BEFORE-CRASH "
CRASH_STDERR = "fatal: agent segfaulted (fake)"


# Red on current main for a tracked, open bug: key -> (the bug's own failure-message pattern, reason).
KNOWN: dict[str, tuple[str, str]] = {
    "auth_remedy": (r"^remedy 'hermes [^']+' is not implemented for copilot-acp: ",
                    "#121290 copilot-acp auth failure tells the user to run a hermes command that is not implemented"),
}


@dataclass(frozen=True)
class Row:
    turns: list[list[dict[str, Any]]]
    succeeds: bool
    model_calls: int
    visible: str  # text the user must see exactly once (answer or the agent's error message)
    never_persisted: tuple[str, ...] = ()


ROWS: dict[str, Row] = {
    "internal_error_retried": Row(
        [[acp.rpc_error(-32603, "Internal error: upstream hiccup")], [acp.message(OK_TEXT)]],
        succeeds=True, model_calls=2, visible=OK_TEXT, never_persisted=("upstream hiccup",)),
    "crash_mid_stream_retried": Row(
        [[acp.message(PARTIAL), acp.crash(3, CRASH_STDERR)], [acp.message(OK_TEXT)]],
        succeeds=True, model_calls=2, visible=OK_TEXT, never_persisted=(PARTIAL.strip(),)),
    "auth_required_not_retried": Row(
        [[acp.rpc_error(-32000, "Authentication required")]] * 3,
        succeeds=False, model_calls=1, visible="Authentication required"),
    "invalid_params_bounded": Row(
        [[acp.rpc_error(-32602, "Invalid params: prompt exceeds agent limit")]] * 4,
        succeeds=False, model_calls=API_MAX_RETRIES, visible="prompt exceeds agent limit"),
    "crash_every_time_bounded": Row(
        [[acp.message(PARTIAL), acp.crash(3, CRASH_STDERR)]] * 4,
        succeeds=False, model_calls=API_MAX_RETRIES, visible=CRASH_STDERR, never_persisted=(PARTIAL.strip(),)),
}


@dataclass
class Outcome:
    nh: NativeHome
    fake: acp.AcpFake
    run: ChatResult
    extra: dict[str, Any] = field(default_factory=dict)


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def _drive(root: Path, row: Row) -> Outcome:
    fake = acp.AcpFake(root / "acp", row.turns)
    nh = make_home(root, {"provider": "copilot-acp", "default": "copilot-acp"}, env_file=fake.env())
    return Outcome(nh, fake, run_chat(nh, "Answer the question, please."))


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("copilot_acp_errors")
    with ThreadPoolExecutor(max_workers=len(ROWS)) as pool:
        futures = {name: pool.submit(_drive, base / name, row) for name, row in ROWS.items()}
        done = {name: fut.result() for name, fut in futures.items()}
    yield done
    for out in done.values():
        for pid in out.fake.pids():
            if _alive(pid):
                with contextlib.suppress(ProcessLookupError):  # exited between the check and the kill
                    os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("name", list(ROWS))
def test_acp_failure_is_retried_per_semantics_and_surfaced_once(outcomes, name):
    row, out = ROWS[name], outcomes[name]
    run, fake = out.run, out.fake
    assert fake.invalid() == [], f"requests rejected by the ACP schema: {fake.invalid()}"
    assert len(fake.main_prompts()) == row.model_calls, (
        f"{name}: {len(fake.main_prompts())} model calls, expected {row.model_calls}\n{run.describe()}")
    assert (run.returncode == 0) is row.succeeds, run.describe()
    assert run.stdout.count(row.visible) == 1, f"{row.visible!r} must be shown exactly once\n{run.describe()}"
    rows = messages(out.nh, latest_session(out.nh))
    persisted = "\n".join(r["content"] or "" for r in rows if r["role"] == "assistant")
    for text in row.never_persisted:
        assert text not in persisted, f"failed-attempt text {text!r} was persisted: {rows}"
    if row.succeeds:
        assert persisted.count(OK_TEXT) == 1, rows
    pids = fake.pids()
    assert len(pids) == row.model_calls, f"one agent process per model call expected: {pids}"
    wait_until(lambda: not [p for p in pids if _alive(p)], 10.0, f"agent processes {pids} to exit")


REMEDY_RE = re.compile(r"`(hermes [^`]+)`")


def test_auth_failure_remedy_is_an_actionable_command(outcomes):
    """The sign-in remedy printed for an ACP ``Authentication required`` must not be a dead end: any
    ``hermes ...`` command it names has to be implemented for this provider."""
    out = outcomes["auth_required_not_retried"]
    assert out.run.returncode != 0 and "Authentication required" in out.run.stdout, out.run.describe()
    for command in REMEDY_RE.findall(out.run.stdout):
        argv = [sys.executable, "-m", "hermes_cli.main", *command.split()[1:]]
        proc = subprocess.run(argv, cwd=out.nh.project, env=out.nh.env(), capture_output=True, text=True,
                              timeout=60, stdin=subprocess.DEVNULL)
        said = (proc.stdout + proc.stderr).lower()
        with known_gate(KNOWN, "auth_remedy", raises=KnownSymptom):
            if "not implemented" in said:
                raise KnownSymptom(f"remedy {command!r} is not implemented for copilot-acp: {said.strip()[:300]}")
