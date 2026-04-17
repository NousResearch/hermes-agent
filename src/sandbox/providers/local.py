"""Host-process sandbox provider (no extra isolation)."""

from __future__ import annotations

import asyncio

from sandbox.base import SandboxProvider
from sandbox.types import SandboxExecResult


class LocalSandboxProvider(SandboxProvider):
    """Runs commands directly on the host (parity with legacy ``execute_code``)."""

    async def exec_cmd(
        self,
        argv: list[str],
        cwd: str,
        env: dict[str, str],
        timeout_sec: float,
    ) -> SandboxExecResult:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxExecResult("", "timeout", -1)
        stdout = (out or b"").decode("utf-8", errors="replace")
        stderr = (err or b"").decode("utf-8", errors="replace")
        code = proc.returncode if proc.returncode is not None else -1
        return SandboxExecResult(stdout, stderr, code)
