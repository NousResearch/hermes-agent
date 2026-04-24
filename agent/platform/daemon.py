"""
Daemon management abstraction.
"""
import os
import subprocess
import tempfile
import sys
from agent.platform.platform_info import platform_info
from pathlib import Path

class DaemonManager:
    @staticmethod
    def spawn_daemon(args: list[str], log_path: str, cwd: str) -> None:
        """
        Spawns a background process disconnected from the terminal.
        """
        with open(log_path, 'a') as log_file:
            if platform_info.is_windows:
                # Use creationflags for DETACHED_PROCESS so it survives parent exit.
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

                subprocess.Popen(
                    args,
                    cwd=cwd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=flags
                )
            elif platform_info.is_termux:
                import shlex
                import uuid
                # Termux doesn't support setsid surviving terminal closes natively, wrap in tmux
                escaped_args = " ".join(shlex.quote(arg) for arg in args)
                session_name = f"hermes-daemon-{uuid.uuid4().hex[:8]}"
                tmux_cmd = ["tmux", "new-session", "-d", "-s", session_name, f"{escaped_args}"]
                subprocess.Popen(
                    tmux_cmd,
                    cwd=cwd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL
                )
            else:
                # Use standard setsid via preexec_fn for Unix
                subprocess.Popen(
                    args,
                    cwd=cwd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True, # Modern Python equivalent to preexec_fn=os.setsid
                )