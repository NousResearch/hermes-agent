"""Entrypoint drivers for the parity matrix: CLI subprocess entrypoints.

Driver contract (every ``_drive_*.py`` module follows it)::

    def drive(ph: ParityHome, srv: FakeLLMServer, prompt: str) -> DriveResult

* spawn the REAL entrypoint with ``ph.env()`` (hermetic fake HOME, no real
  credentials) — cwd ``ph.project`` unless the surface documents another cwd
  channel (then use that channel and say so in ``DriveResult.cwd_channel``);
* run exactly ONE user turn with ``prompt`` against ``srv`` (the scripted
  responder makes the model call the MCP canary tool, then answer
  ``FINAL_ANSWER``);
* return what the surface delivered to ITS client in ``final_text``;
* stop the entrypoint through its NORMAL shutdown path (exit, stdin EOF,
  SIGTERM, RPC) before returning; never SIGKILL except as a last resort after a
  bounded graceful wait (and then report it via ``graceful_exit=False``).
"""

from __future__ import annotations

import subprocess

from tests.e2e.core.parity._helpers import TURN_TIMEOUT, DriveResult, ParityHome, hermes_argv
from tests.fakes.fake_llm_provider import FakeLLMServer


def _run_cli(ph: ParityHome, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        hermes_argv(*args), cwd=ph.project, env=ph.env(), capture_output=True, text=True,
        timeout=TURN_TIMEOUT, stdin=subprocess.DEVNULL,
    )


def drive_oneshot(ph: ParityHome, srv: FakeLLMServer, prompt: str) -> DriveResult:
    proc = _run_cli(ph, "-z", prompt)
    assert proc.returncode == 0, f"hermes -z exited {proc.returncode}: {proc.stderr[-2000:]}"
    return DriveResult(final_text=proc.stdout.strip(), toolset="hermes-cli")


def drive_chat_q(ph: ParityHome, srv: FakeLLMServer, prompt: str) -> DriveResult:
    proc = _run_cli(ph, "chat", "-q", prompt, "-Q")
    assert proc.returncode == 0, f"hermes chat -q exited {proc.returncode}: {proc.stderr[-2000:]}"
    return DriveResult(final_text=proc.stdout.strip(), toolset="hermes-cli")
