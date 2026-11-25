"""Vertex AI through the classic CLI's gateway path: ``/config`` and ``/bg`` (#134889 T6).

A real ``hermes chat --cli`` runs in a private tmux server with ``provider: vertex`` and a synthetic
service-account JSON. The fake (``tests/fakes/providers/vertex.py``) stands in for both Google
boundaries on loopback: the OAuth token endpoint (the real ``google-auth`` signs and exchanges a
JWT) and the Vertex OpenAI-compatible endpoint (reached through ``HTTPS_PROXY`` with a test CA).
No request leaves the machine and no real credential exists.

* ``/config`` names the credential "Google OAuth (Vertex AI)", never Azure's "Microsoft Entra ID",
  and showing it mints no token.
* ``/bg <prompt>`` runs in the gateway on the session's Vertex route: the side agent's request
  reaches Vertex with a bearer the token endpoint minted, the answer prints in the CLI, and the
  foreground session's history is untouched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("google.auth", reason="Vertex minting needs google-auth (CI installs it)")

from tests.e2e.core.cli_tmux._classic import kill_home_processes, require_tmux
from tests.e2e.core.providers._native_helpers import make_home, messages
from tests.fakes.providers.vertex import PROJECT, REGION, SA_EMBEDDED_PROJECT, FakeVertex, Say, hermes_setup

pytestmark = [
    pytest.mark.platforms("linux"),  # cleanup finds this home's processes through /proc
]

BG_MARK = "VERTEXBG"
BG_ANSWER = "VERTEXBG-ANSWER from the background agent"


def _user_text(body: dict[str, Any]) -> str:
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    content = users[-1].get("content") if users else ""
    if isinstance(content, list):
        content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content or "")


def _respond(record: dict[str, Any]) -> Say:
    return Say(BG_ANSWER if BG_MARK in _user_text(record["body"]) else "foreground ok")


def _drive(fake: FakeVertex, tmp_path: Path):
    nh = make_home(tmp_path / "h", **hermes_setup(fake, extra_config={"display": {"interface": "cli"}}))
    sock = f"hermes-e2e-vertex-{uuid.uuid4().hex[:8]}"
    env = nh.env({**fake.child_env(), "TERM": "xterm-256color"})

    def tmux(*args: str) -> str:
        return subprocess.run(["tmux", "-L", sock, *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=30, check=False).stdout

    def screen() -> str:
        return tmux("capture-pane", "-p", "-J", "-t", "p", "-S", "-", "-E", "-")

    def wait_for(needle: str, timeout: float = 90.0) -> str:
        end = time.monotonic() + timeout
        while needle not in (text := screen()):
            if time.monotonic() >= end:
                pytest.fail(f"{needle!r} never appeared:\n{text[-4000:]}\nvertex requests: "
                            f"{[(r.get('kind'), r.get('status'), r.get('rejected')) for r in fake.requests]}")
            time.sleep(0.1)
        return text

    def submit(line: str) -> None:
        tmux("send-keys", "-t", "p", "-l", line)
        wait_for(f"\u276f {line}", timeout=20)
        # An Enter within 50 ms of the last buffer change reads as a pasted newline.
        time.sleep(0.5)  # health: allow HX013 -- the gap IS the behaviour: the CLI's paste detection is time-based
        tmux("send-keys", "-t", "p", "Enter")

    # `-t file` like the sibling Vertex suites: the default toolset trips #109115 (terminal.notify's
    # anyOf schema), tracked in test_native_vertex_tools.py, which is not what this test is about.
    argv = [sys.executable, "-m", "hermes_cli.main", "chat", "--cli", "--yolo", "-t", "file"]
    try:
        subprocess.run(["tmux", "-L", sock, "-f", os.devnull, "new-session", "-d", "-s", "p", "-x", "160", "-y", "48",
                        "-c", str(nh.project), *argv], env=env, check=True, timeout=30)
        tmux("set", "-g", "remain-on-exit", "on")
        wait_for("\u2624 ", timeout=120)  # the status bar: the app owns the terminal (raw mode)

        # The CLI resolves its route at the first turn (as main did); /config reads that route.
        submit("hello vertex")
        wait_for("foreground ok", timeout=120)

        mints_before_config = len(fake.token_requests)
        submit("/config")
        config_screen = wait_for("Google OAuth (Vertex AI)", timeout=30)
        assert "Microsoft Entra ID" not in config_screen
        assert len(fake.token_requests) == mints_before_config, "/config minted a Vertex token to display it"

        submit(f"/bg {BG_MARK} compute something")
        wait_for(BG_ANSWER, timeout=120)
    finally:
        # Independent steps: a tmux failure or a graceful stop that times out still kills every
        # process on this home (the fake server is stopped by the caller's own finally).
        try:
            tmux("kill-server")
        finally:
            kill_home_processes(nh.hermes_home, lambda: subprocess.run(
                [sys.executable, "-m", "hermes_cli.main", "gateway", "stop"], cwd=nh.project, env=env,
                capture_output=True, text=True, timeout=60, check=False))
    return nh


def test_vertex_config_label_and_background_task_through_the_gateway(tmp_path: Path) -> None:
    require_tmux()
    fake = FakeVertex(tmp_path / "fake", project=PROJECT, region=REGION, sa_project=SA_EMBEDDED_PROJECT,
                      script=_respond, aux=lambda _r: Say("Vertex title"))
    fake.start()
    try:
        nh = _drive(fake, tmp_path)
    finally:
        fake.stop()  # independent of the CLI cleanup inside _drive, which may itself raise

    bg = [r for r in fake.requests if BG_MARK in _user_text(r.get("body") or {})]
    assert bg and all(r.get("status") == 200 and not r.get("rejected") for r in bg), bg
    minted = set(fake.minted_tokens())
    assert all(r["auth"].removeprefix("Bearer ") in minted for r in bg), "bg request carried an unminted bearer"
    assert bg[0]["body"]["model"] == "google/gemini-3-flash-preview"
    # The side task never writes into the foreground session's transcript.
    foreground = [m for m in messages(nh) if BG_MARK in str(m.get("content") or "")]
    assert all(not str(m.get("session_id") or "").startswith("local-") for m in foreground), foreground
