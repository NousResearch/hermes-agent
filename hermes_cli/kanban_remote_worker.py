"""Polling supervisor for a Hermes worker attached to a remote coordinator."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_remote import RemoteKanban


log = logging.getLogger(__name__)

def _worker_workspace(task_id: str) -> Path:
    path = kb.kanban_home() / "remote-workspaces" / task_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_task(client: RemoteKanban, task, *, profile: str, machine_id: str) -> None:
    workspace = _worker_workspace(task.id)
    context = client.build_worker_context(None, task.id)
    prompt = (
        f"Work the assigned remote Kanban task. Your local workspace is {workspace}.\n\n"
        f"{context}"
    )
    env = os.environ.copy()
    env.update({
        "HERMES_KANBAN_TASK": task.id,
        "HERMES_KANBAN_RUN_ID": str(task.current_run_id or ""),
        "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock or "",
        "HERMES_KANBAN_COORDINATOR_URL": client.base_url,
        "HERMES_KANBAN_COORDINATOR_TOKEN": client.token,
        "HERMES_MACHINE_ID": machine_id,
        "TERMINAL_CWD": str(workspace),
        "HERMES_PROFILE": profile,
    })
    env.pop("HERMES_TUI", None)
    proc = subprocess.Popen(
        [
            *kb._resolve_hermes_argv(),
            "-p", profile,
            "--cli",
            "--accept-hooks",
            "chat",
            "-q", prompt,
        ],
        cwd=str(workspace),
        env=env,
        stdin=subprocess.DEVNULL,
    )
    reported_started = False
    while proc.poll() is None:
        try:
            if not reported_started:
                reported_started = client.record_worker_started(
                    task.id,
                    claim_lock=task.claim_lock or "",
                    worker_pid=proc.pid,
                )
            client.heartbeat_claim(None, task.id, claimer=task.claim_lock)
        except Exception:
            # The coordinator lease remains the authority; retry next poll.
            pass
        time.sleep(20)
    # A normal worker closes its task through kanban_complete or kanban_block.
    # Do not leave a terminated remote process holding a lease until its TTL.
    try:
        current = client.get_task(None, task.id)
        if current is not None and current.status == "running":
            client.block_task(
                None,
                task.id,
                reason=f"remote worker exited with status {proc.returncode}",
                kind="transient",
                claim_lock=task.claim_lock,
            )
    except Exception as exc:
        # The coordinator may be briefly unreachable exactly as the child
        # exits. Do not crash the supervisor: the lease/dispatcher recovery
        # path will make the unfinished task visible again.
        log.warning("could not finalize remote task %s: %s", task.id, exc)


def run_worker_loop(
    client: RemoteKanban,
    *,
    profile: str,
    capabilities: list[str],
    poll_seconds: float,
    max_retry_seconds: float,
    sleep_fn=time.sleep,
) -> None:
    """Register, claim and run work forever with bounded outage backoff."""
    machine_id = kb.get_machine_id()
    delay = max(1.0, poll_seconds)
    maximum_delay = max(delay, max_retry_seconds)
    while True:
        try:
            client.register_machine(
                machine_id,
                hostname=socket.gethostname(),
                profiles=[profile],
                capabilities=capabilities,
            )
            task = client.claim_next(machine_id)
            delay = max(1.0, poll_seconds)
        except (OSError, RuntimeError) as exc:
            log.warning(
                "Kanban coordinator unavailable; retrying in %.1fs: %s",
                delay,
                exc,
            )
            sleep_fn(delay)
            delay = min(maximum_delay, delay * 2)
            continue
        if task is None:
            sleep_fn(max(1.0, poll_seconds))
            continue
        _run_task(client, task, profile=profile, machine_id=machine_id)


def run(
    *,
    url: str,
    token_env: str,
    profile: str,
    capabilities: list[str],
    poll_seconds: float,
    max_retry_seconds: float,
) -> int:
    """Run a remote worker using the same configuration surface as the CLI."""
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise RuntimeError(f"set {token_env} before starting the worker")

    merged_capabilities = sorted(set(kb.local_machine_capabilities()) | set(capabilities))
    client = RemoteKanban(url, token)
    try:
        run_worker_loop(
            client,
            profile=profile,
            capabilities=merged_capabilities,
            poll_seconds=poll_seconds,
            max_retry_seconds=max_retry_seconds,
        )
    except KeyboardInterrupt:
        log.info("remote Kanban worker stopped")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a remote Hermes Kanban worker")
    parser.add_argument("--url", required=True, help="Coordinator base URL")
    parser.add_argument("--token-env", default="HERMES_KANBAN_COORDINATOR_TOKEN")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--max-retry-seconds",
        type=float,
        default=60.0,
        help="Maximum coordinator reconnect delay after an outage (default: 60)",
    )
    args = parser.parse_args(argv)
    try:
        return run(
            url=args.url,
            token_env=args.token_env,
            profile=args.profile,
            capabilities=args.capability,
            poll_seconds=args.poll_seconds,
            max_retry_seconds=args.max_retry_seconds,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
