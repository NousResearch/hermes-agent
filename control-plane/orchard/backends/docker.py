"""Docker backend: one container per employee — the recommended prod baseline.

The container is the OS boundary (mount/PID/net namespaces). Inside it runs the
same worker_daemon as the local backend, so the router protocol is unchanged.

This is a scaffold with the real `docker` invocations sketched out. It is not
exercised by the local test loop (no Docker in the dev sandbox). Fill in image
build + registry push for your environment, then flip `backend: docker`.

Key hardening (mirrors Hermes' own docker terminal backend):
  --cap-drop ALL --security-opt no-new-privileges --pids-limit ...
  --read-only rootfs + tmpfs for scratch, employee HERMES_HOME as the only
  writable bind mount, per-container memory/cpu limits, no inter-container net.
"""
from __future__ import annotations

import asyncio
import json

from ..config import Settings
from ..models import Employee
from .base import WorkerBackend


class DockerBackend(WorkerBackend):
    """Container-per-employee. See module docstring; TODOs marked inline."""

    IMAGE = "hermes-orchard-worker:latest"

    def _container(self, employee: Employee) -> str:
        return f"orchard-{employee.id}"

    async def is_ready(self, employee: Employee) -> bool:
        rc, out, _ = await self._run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self._container(employee)]
        )
        return rc == 0 and out.strip() == "true"

    async def ensure_ready(self, employee: Employee) -> None:
        if await self.is_ready(employee):
            return
        s = self.settings
        home = s.paths.home_for(employee.id)
        sock_dir = s.paths.runtime / employee.id
        sock_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", self._container(employee),
            # --- isolation / hardening ---
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "512",
            "--network", "none",            # worker reaches LLM via a proxy sidecar, not open net
            "--memory", "2g", "--cpus", "1.5",
            "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,size=256m",
            # --- the ONLY writable tenant data: this employee's home ---
            "-v", f"{home}:/data:rw",
            "-v", f"{sock_dir}:/run/orchard:rw",
            "-e", "HERMES_HOME=/data",
            "-e", "ORCHARD_SOCKET=/run/orchard/worker.sock",
            # Per-worker token (store it in self._tokens and echo it in send()).
            "-e", f"ORCHARD_WORKER_TOKEN={__import__('secrets').token_hex(16)}",
            "-e", f"ORCHARD_HERMES_BIN={s.hermes_bin}",
            "-e", "ORCHARD_WORKSPACE=/data/workspace",
            self.IMAGE,
            "python", "-m", "orchard.worker_daemon",
        ]
        rc, _, err = await self._run(cmd)
        if rc != 0:
            raise RuntimeError(f"docker run failed: {err}")
        # TODO: poll the container's socket for readiness like LocalBackend does.

    async def send(self, employee: Employee, session: str, message: str) -> str:
        # TODO: connect to sock_dir/worker.sock (bind-mounted out of the container)
        # and speak the same JSON protocol as LocalBackend._rpc.
        raise NotImplementedError("wire up unix-socket RPC to the container's daemon")

    async def sleep(self, employee: Employee) -> None:
        await self._run(["docker", "rm", "-f", self._container(employee)])

    async def _run(self, cmd: list[str]) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")
