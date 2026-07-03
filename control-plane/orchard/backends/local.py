"""Local backend: one worker daemon subprocess per employee on the shared host.

Isolation = filesystem permissions (+ optional `run_as_user` privilege drop).
This is for development and small, trusted deployments. For untrusted / at-scale
use the docker (or future microvm) backend where the OS kernel is the boundary.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import subprocess
import sys
import time

from ..config import Settings
from ..models import Employee
from ..security import drop_privileges_cmd, runtime_read_paths, seatbelt_wrap
from .base import WorkerBackend


class LocalBackend(WorkerBackend):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._procs: dict[str, subprocess.Popen] = {}
        self._tokens: dict[str, str] = {}

    # --- lifecycle -----------------------------------------------------------
    async def is_ready(self, employee: Employee) -> bool:
        try:
            resp = await self._rpc(employee, {"type": "ping"}, timeout=2.0)
            return bool(resp.get("pong"))
        except Exception:
            return False

    async def ensure_ready(self, employee: Employee) -> None:
        if await self.is_ready(employee):
            return
        self._spawn(employee)
        deadline = time.monotonic() + self.settings.supervisor.wake_timeout_seconds
        while time.monotonic() < deadline:
            if await self.is_ready(employee):
                return
            # daemon died during startup?
            proc = self._procs.get(employee.id)
            if proc and proc.poll() is not None:
                raise RuntimeError(f"worker {employee.id} exited during wake (code {proc.returncode})")
            await asyncio.sleep(0.25)
        raise TimeoutError(f"worker {employee.id} did not become ready in time")

    def _spawn(self, employee: Employee) -> None:
        s = self.settings
        home = s.paths.home_for(employee.id)
        # Refresh on every wake: base skills (Hermes only scans HERMES_HOME/skills),
        # operating rules (SOUL.md), and config (model + disabled bundled skills).
        from ..provisioner import (
            install_fetch_helper, sync_shared_skills, write_hermes_config, write_soul,
        )
        sync_shared_skills(s, employee.id)
        write_soul(s, employee.id)
        write_hermes_config(s, home)
        install_fetch_helper(s, employee.id)
        sock = s.paths.socket_for(employee.id)
        sock.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(16)
        self._tokens[employee.id] = token
        env = dict(os.environ)
        env.update(
            HERMES_HOME=str(home),
            ORCHARD_SOCKET=str(sock),
            ORCHARD_HERMES_BIN=s.hermes_bin,
            ORCHARD_WORKSPACE=str(home / "workspace"),
            ORCHARD_EMPLOYEE_ID=employee.id,
            ORCHARD_IDLE_TTL="0",  # supervisor owns idle policy
        )
        # Shared org config (URLs etc.) — common to all workers, non-secret, so
        # env is fine. Per-employee TOKENS are injected separately by the daemon
        # (from secrets.json, per turn) — never here.
        from .. import integrations as _integrations
        env.update(_integrations.shared_config(s))
        # Let a skill mint a secure secret-entry link for THIS employee when a
        # token is missing (the agent hands the user the link instead of asking
        # for the value in chat).
        env["ORCHARD_API"] = s.secrets.form_base_url.rstrip("/")
        # NOTE: the token is intentionally NOT in env (a same-UID sibling could
        # read it via `ps eww`). It's piped over stdin instead.
        cmd = drop_privileges_cmd(
            [sys.executable, "-m", "orchard.worker_daemon"], s.security.run_as_user
        )
        if s.security.sandbox == "seatbelt":
            # Confine to this tenant's own home; deny the entire human home root
            # (other tenants, ~/.hermes secrets, ~/.ssh, docs, other repos...).
            cmd = seatbelt_wrap(cmd, home, runtime_read_paths(s.hermes_bin))
        # New session so we can signal the whole group on sleep.
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, start_new_session=True)
        try:
            proc.stdin.write((token + "\n").encode())
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass
        self._procs[employee.id] = proc

    async def sleep(self, employee: Employee) -> None:
        self._tokens.pop(employee.id, None)
        proc = self._procs.pop(employee.id, None)
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.get_event_loop().run_in_executor(None, proc.wait, 10)
            except Exception:
                proc.kill()
        sock = self.settings.paths.socket_for(employee.id)
        if sock.exists():
            sock.unlink()

    async def shutdown_all(self) -> None:
        for emp_id in list(self._procs):
            await self.sleep(Employee(emp_id, emp_id, emp_id, 0.0))

    # --- messaging -----------------------------------------------------------
    async def send(self, employee: Employee, session: str, message: str) -> str:
        resp = await self._rpc(
            employee, {"type": "chat", "session": session, "message": message}, timeout=600.0
        )
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "worker error"))
        return resp["reply"]

    async def _rpc(self, employee: Employee, req: dict, timeout: float) -> dict:
        sock = self.settings.paths.socket_for(employee.id)
        req = {**req, "token": self._tokens.get(employee.id, "")}

        async def _do() -> dict:
            reader, writer = await asyncio.open_unix_connection(path=str(sock))
            try:
                writer.write((json.dumps(req) + "\n").encode("utf-8"))
                await writer.drain()
                line = await reader.readline()
                return json.loads(line.decode("utf-8"))
            finally:
                writer.close()

        return await asyncio.wait_for(_do(), timeout=timeout)
