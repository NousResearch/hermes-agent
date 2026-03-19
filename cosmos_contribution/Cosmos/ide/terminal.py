"""
Terminal Session for IDE.

Provides PTY-based terminal for Windows and Unix.
"""

import os
import sys
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TerminalSession:
    """
    Terminal session for xterm.js integration.

    Uses:
    - pywinpty on Windows
    - pty on Unix
    """

    def __init__(self, cwd: str = None):
        self.cwd = cwd or os.getcwd()
        self._process = None
        self._reader = None

    async def start(self):
        """Start the terminal session."""
        if sys.platform == "win32":
            await self._start_windows()
        else:
            await self._start_unix()

    async def _start_windows(self):
        """Start terminal on Windows using pywinpty."""
        try:
            import winpty

            self._pty = winpty.PtyProcess.spawn(
                "cmd.exe",
                cwd=self.cwd,
                env=os.environ.copy(),
            )

        except ImportError:
            # Fallback to subprocess
            self._process = await asyncio.create_subprocess_shell(
                "cmd.exe",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.cwd,
            )

    async def _start_unix(self):
        """Start terminal on Unix using pty."""
        import pty
        import subprocess

        shell = os.environ.get("SHELL", "/bin/bash")
        master_fd, slave_fd = pty.openpty()

        self._process = subprocess.Popen(
            [shell],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=self.cwd,
            preexec_fn=os.setsid,
        )

        os.close(slave_fd)
        self._master_fd = master_fd

    async def read(self) -> bytes:
        """Read output from terminal."""
        if hasattr(self, "_pty"):
            try:
                return self._pty.read(4096).encode()
            except Exception:
                return b""

        elif hasattr(self, "_master_fd"):
            import select
            if select.select([self._master_fd], [], [], 0.1)[0]:
                return os.read(self._master_fd, 4096)

        elif self._process and self._process.stdout:
            return await self._process.stdout.read(4096)

        return b""

    def write(self, data: bytes):
        """Write input to terminal."""
        if hasattr(self, "_pty"):
            self._pty.write(data.decode())

        elif hasattr(self, "_master_fd"):
            os.write(self._master_fd, data)

        elif self._process and self._process.stdin:
            self._process.stdin.write(data)

    async def stop(self):
        """Stop the terminal session."""
        if hasattr(self, "_pty"):
            self._pty.close()

        elif self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
