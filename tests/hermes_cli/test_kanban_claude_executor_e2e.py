"""End-to-end: a disposable board driven through the real dispatcher path.

Everything here is real except the model. A disposable board is created on
disk, a real card is created on it, and a real `dispatch_once` tick claims the
card and calls the real `_default_spawn`, which resolves the real
`claude_cli` executor and launches a real detached subprocess.

The subprocess is a stand-in for the host Claude Code CLI: it accepts the same
argv shape, then walks the lifecycle protocol out of the prompt it was handed
using real `hermes kanban …` subcommands against the real board DB. That is
the part worth proving. Whether an LLM chooses to follow the protocol is a
model-quality question; whether the protocol *works* — whether the env pins
reach the child, whether `hermes kanban complete` from inside that child lands
on the right card of the right board — is what breaks in production, and it is
exercised for real here.

Covers the full claim -> show -> workspace operation -> heartbeat/comment ->
complete round trip named in the board-recovery acceptance criteria.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb


BOARD = "e2e-claude-exec"

# Provider credentials seeded into the dispatcher's env. The worker asserts it
# cannot see any of them; if the strip regresses, the E2E fails on a real
# subprocess rather than on a mocked env dict.
LEAK_CANARIES = {
    "CLAUDE_CONFIG_DIR": "/tmp/hermes-managed-claude",
    "ANTHROPIC_API_KEY": "sk-ant-e2e-canary",
    "ANTHROPIC_BASE_URL": "https://proxy.invalid",
    "OPENAI_API_KEY": "sk-oai-e2e-canary",
    "CODEX_HOME": "/tmp/codex-e2e",
}


# The fake host CLI. Mirrors `claude -p [--model …] [--effort …]
# [--permission-mode …] "<prompt>"`: flags first, prompt as the trailing
# positional. It then executes the lifecycle the prompt describes.
FAKE_CLAUDE = '''#!{python}
"""Stand-in for the host Claude Code CLI: runs the kanban worker protocol."""
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

RECEIPT = Path(os.environ["E2E_RECEIPT"])
receipt = {{"argv": sys.argv[1:], "steps": [], "errors": []}}


def fail(msg):
    receipt["errors"].append(msg)
    RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    sys.exit(3)


# --- argv contract -----------------------------------------------------
if "-p" not in sys.argv:
    fail("missing -p (print/non-interactive mode)")
prompt = sys.argv[-1]
if prompt.startswith("-"):
    fail("prompt is not the trailing positional: %r" % prompt)
receipt["prompt"] = prompt

# --- environment contract ----------------------------------------------
for canary in {canaries!r}:
    if canary in os.environ:
        fail("provider credential leaked into the worker env: %s" % canary)

task_id = os.environ.get("HERMES_KANBAN_TASK")
if not task_id:
    fail("HERMES_KANBAN_TASK not pinned")
receipt["env"] = {{
    key: os.environ.get(key)
    for key in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_PROFILE", "HERMES_TENANT", "HERMES_SESSION_SOURCE",
        "HERMES_KANBAN_EXECUTOR", "GIT_TERMINAL_PROMPT", "TERMINAL_CWD",
    )
}}
receipt["cwd"] = os.getcwd()

# The prompt tells the worker which Hermes invocation to drive the board with.
# Parsed out of the prompt rather than assumed, because that resolution is
# exactly what breaks when `hermes` is not on a detached worker's PATH.
match = re.search(r"`(.*?) kanban show " + re.escape(task_id) + r"`", prompt)
if not match:
    fail("prompt did not carry a `kanban show` command")
hermes_cmd = shlex.split(match.group(1).strip())
receipt["hermes_cmd"] = hermes_cmd


def kanban(*args, check=True):
    proc = subprocess.run(
        hermes_cmd + ["kanban"] + list(args),
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    receipt["steps"].append({{
        "args": list(args), "rc": proc.returncode,
        "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:],
    }})
    if check and proc.returncode != 0:
        fail("`kanban %s` exited %d: %s" % (" ".join(args), proc.returncode, proc.stderr[-500:]))
    return proc


# --- protocol ----------------------------------------------------------
shown = kanban("show", task_id)
if task_id not in shown.stdout:
    fail("`kanban show` did not return this task")
receipt["show_ok"] = True

# Workspace operation: the claimed workspace is the cwd.
artifact = Path(os.environ["HERMES_KANBAN_WORKSPACE"]) / "worker-artifact.txt"
artifact.write_text("written by the direct Claude CLI worker\\n", encoding="utf-8")
receipt["artifact"] = str(artifact)

kanban("heartbeat", task_id, "--note", "e2e worker alive")
kanban("comment", task_id, "direct-claude-cli worker did the thing")
kanban("complete", task_id, "--result", "e2e complete via direct Claude CLI")

receipt["ok"] = True
RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("worker finished task %s" % task_id)
'''


@pytest.fixture
def disposable_board(tmp_path, monkeypatch):
    """A real board on disk under an isolated HERMES_HOME."""
    home = tmp_path / "hermes_home"
    (home / "profiles" / "integrator").mkdir(parents=True)
    (home / "profiles" / "integrator" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    home.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKER_EXECUTOR",
    ):
        monkeypatch.delenv(var, raising=False)
    kb._reset_path_cache_for_tests() if hasattr(kb, "_reset_path_cache_for_tests") else None

    kb.create_board(BOARD, name="E2E direct Claude executor")
    return home


@pytest.fixture
def fake_claude_on_path(tmp_path, monkeypatch):
    """Install the stand-in host CLI as the only `claude` on PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "claude"
    exe.write_text(
        FAKE_CLAUDE.format(python=sys.executable, canaries=sorted(LEAK_CANARIES)),
        encoding="utf-8",
    )
    exe.chmod(0o755)
    # Only the fake bin dir plus the system essentials: a real `claude` or
    # `hermes` on the developer's PATH must not be reachable from this test.
    monkeypatch.setenv("PATH", os.pathsep.join([str(bindir), "/usr/bin", "/bin"]))
    return exe


def test_disposable_board_roundtrip_via_direct_claude_cli(
    disposable_board, fake_claude_on_path, tmp_path, monkeypatch
):
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setenv("E2E_RECEIPT", str(receipt_path))
    # The child re-enters the CLI as `python -m hermes_cli.main`; make sure it
    # imports *this* worktree rather than an installed copy.
    monkeypatch.setenv("PYTHONPATH", str(_WORKTREE))
    for key, value in LEAK_CANARIES.items():
        monkeypatch.setenv(key, value)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    conn = kb.connect(board=BOARD)
    try:
        task_id = kb.create_task(
            conn,
            title="E2E: direct Claude CLI worker",
            body="Write the artifact, comment, and complete.",
            assignee="integrator",
            created_by="e2e",
            workspace_kind="dir",
            workspace_path=str(workspace),
            tenant="e2e-tenant",
        )

        # --- the real dispatcher tick -----------------------------------
        result = kb.dispatch_once(conn, board=BOARD, max_spawn=1)
        # spawned is a list of (task_id, assignee, workspace) triples.
        assert task_id in [row[0] for row in result.spawned], (
            f"dispatcher did not spawn the card: {result!r}"
        )

        claimed = kb.get_task(conn, task_id)
        assert claimed.status == "running"
        assert claimed.worker_pid, "no worker PID recorded — crash detection would be blind"

        # --- wait for the worker to finish the protocol ------------------
        deadline = time.time() + 120
        while time.time() < deadline:
            if receipt_path.exists():
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    receipt = None
                if receipt is not None and (receipt.get("ok") or receipt.get("errors")):
                    break
            time.sleep(0.5)
        else:
            pytest.fail("worker did not finish within 120s")

        assert receipt["errors"] == [], f"worker protocol failures: {receipt['errors']}"
        assert receipt["ok"] is True

        # --- worker-side observations ------------------------------------
        env_seen = receipt["env"]
        assert env_seen["HERMES_KANBAN_TASK"] == task_id
        assert env_seen["HERMES_KANBAN_BOARD"] == BOARD
        assert env_seen["HERMES_PROFILE"] == "integrator"
        assert env_seen["HERMES_TENANT"] == "e2e-tenant"
        assert env_seen["HERMES_SESSION_SOURCE"] == "kanban"
        assert env_seen["HERMES_KANBAN_EXECUTOR"] == "claude_cli"
        assert env_seen["GIT_TERMINAL_PROMPT"] == "0"
        assert env_seen["HERMES_KANBAN_WORKSPACE"] == str(workspace)
        assert Path(env_seen["HERMES_KANBAN_DB"]).name == "kanban.db"
        assert BOARD in env_seen["HERMES_KANBAN_DB"]
        assert receipt["cwd"] == str(workspace)
        assert receipt["show_ok"] is True

        # It really was the direct CLI, with a non-interactive permission mode.
        argv = receipt["argv"]
        assert "-p" in argv
        assert "--permission-mode" in argv

        # --- workspace operation ------------------------------------------
        assert (workspace / "worker-artifact.txt").exists()

        # --- board-side lifecycle -----------------------------------------
        final = kb.get_task(conn, task_id)
        assert final.status == "done", f"card not completed: {final.status}"
        assert "direct Claude CLI" in (final.result or "")

        bodies = " ".join(c.body for c in kb.list_comments(conn, task_id))
        assert "direct-claude-cli worker did the thing" in bodies

        kinds = {e.kind for e in kb.list_events(conn, task_id)}
        assert "heartbeat" in kinds, f"no heartbeat recorded; kinds={sorted(kinds)}"
        assert "completed" in kinds, f"no completion event; kinds={sorted(kinds)}"

        # --- log observation ------------------------------------------------
        log_path = kb.worker_logs_dir(board=BOARD) / f"{task_id}.log"
        assert log_path.exists(), "no per-task worker log — `hermes kanban log` would be empty"
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert f"worker finished task {task_id}" in log_text
        for secret in LEAK_CANARIES.values():
            assert secret not in log_text, "a credential reached the task log"
    finally:
        conn.close()


def test_dispatch_fails_loudly_without_the_host_cli(
    disposable_board, tmp_path, monkeypatch
):
    """No `claude` on PATH must not silently fall back to a native worker."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    workspace = tmp_path / "workspace2"
    workspace.mkdir()

    conn = kb.connect(board=BOARD)
    try:
        task_id = kb.create_task(
            conn,
            title="E2E: missing host CLI",
            assignee="integrator",
            created_by="e2e",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        result = kb.dispatch_once(conn, board=BOARD, max_spawn=1)

        assert not result.spawned, "a worker was spawned without the host CLI"
        task = kb.get_task(conn, task_id)
        assert task.status != "running"
        # The failure is recorded as a spawn failure, not masked.
        assert task.consecutive_failures >= 1
        assert task.last_failure_error
    finally:
        conn.close()
