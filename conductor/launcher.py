from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


class TmuxLauncher:
    """Launch only validated worker specs through a fixed exec wrapper.

    The conductor never receives a shell or general terminal method. The sole
    mutation boundary is a configured worker executable, run in one named tmux
    session with a hash-pinned launch specification.
    """

    def __init__(self, tmux: str = "tmux", socket_path: str | Path | None = None):
        self.tmux = tmux
        self.socket_path = str(Path(socket_path).resolve()) if socket_path else None
        self._wrapper = str(Path(__file__).with_name("worker_exec.py").resolve())

    def _command(self, *args: str) -> list[str]:
        prefix = [self.tmux]
        if self.socket_path:
            prefix.extend(["-S", self.socket_path])
        return [*prefix, *args]

    def launch(self, spec) -> dict:
        if not spec.command or not Path(spec.cwd).is_absolute():
            raise ValueError("worker command and absolute cwd are required")
        run_dir = Path(spec.receipt_path).parent
        launch_path = run_dir / "launch.json"
        log_path = run_dir / "worker.log"
        launch_path.write_text(
            json.dumps(asdict(spec), sort_keys=True), encoding="utf-8"
        )
        fixed_command = shlex.join([sys.executable, self._wrapper, str(launch_path)])
        fixed_command += f" > {shlex.quote(str(log_path))} 2>&1"
        completed = subprocess.run(
            self._command(
                "new-session",
                "-d",
                "-s",
                spec.tmux_session,
                "-c",
                spec.cwd,
                fixed_command,
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"tmux worker launch failed: {completed.stderr.strip()}")
        metadata = subprocess.run(
            self._command(
                "display-message",
                "-p",
                "-t",
                spec.tmux_session,
                "#{pane_pid} #{session_created}",
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        parts = metadata.stdout.strip().split()
        return {
            "pid": int(parts[0]) if parts and parts[0].isdigit() else None,
            "start_marker": parts[1] if len(parts) > 1 else None,
        }

    def is_running(self, session: str) -> bool:
        return (
            subprocess.run(
                self._command("has-session", "-t", f"={session}"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )

    def cleanup(self, session: str) -> None:
        if self.is_running(session):
            subprocess.run(
                self._command("kill-session", "-t", f"={session}"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
