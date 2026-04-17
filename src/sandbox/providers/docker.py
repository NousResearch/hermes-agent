"""Docker-backed sandbox provider."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from sandbox.base import SandboxProvider
from sandbox.docker_cmd import build_docker_run_argv
from sandbox.errors import SandboxConfigError, SandboxNotSupportedError
from sandbox.types import IsolationProfile, SandboxExecResult, isolation_profile_from_config


class DockerSandboxProvider(SandboxProvider):
    """Runs commands inside ``docker run`` with profile-derived limits."""

    def __init__(self, config: dict[str, Any], *, docker_runtime: str | None = None):
        super().__init__(config)
        self._docker_runtime = docker_runtime

    def profile(self) -> IsolationProfile:
        name = str(self._config.get("profile") or "default")
        return isolation_profile_from_config(self._config, name)

    def build_popen_argv(
        self,
        *,
        workdir: str,
        inner_cmd: list[str],
        child_env: dict[str, str],
    ) -> list[str]:
        """Synchronous argv for :class:`subprocess.Popen` on the host."""
        if sys.platform == "win32":
            raise SandboxNotSupportedError(
                "Docker sandbox provider for execute_code is not supported on Windows hosts"
            )
        image = str(self._config.get("image") or "").strip()
        if not image:
            raise SandboxConfigError("sandbox.image is required for docker provider")
        return build_docker_run_argv(
            image=image,
            workdir=workdir,
            inner_cmd=inner_cmd,
            child_env=child_env,
            profile=self.profile(),
            docker_runtime=self._docker_runtime,
        )

    async def exec_cmd(
        self,
        argv: list[str],
        cwd: str,
        env: dict[str, str],
        timeout_sec: float,
    ) -> SandboxExecResult:
        host_argv = self.build_popen_argv(workdir=cwd, inner_cmd=argv, child_env=env)
        proc = await asyncio.create_subprocess_exec(
            *host_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxExecResult("", "timeout", -1)
        stdout = (out or b"").decode("utf-8", errors="replace")
        stderr = (err or b"").decode("utf-8", errors="replace")
        code = proc.returncode if proc.returncode is not None else -1
        return SandboxExecResult(stdout, stderr, code)
