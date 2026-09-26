"""Bounded execution of operator-authored classic CLI quick commands."""
from __future__ import annotations

import os
import subprocess
import threading
import time

from hermes_cli._subprocess_compat import IS_WINDOWS, kill_process_tree, windows_hide_flags

_OUTPUT_LIMIT = 64 * 1024  # per stream, before decoding
_READ_CHUNK = 4096


def _capture_stream(stream, result):
    """Drain continuously while retaining at most the display budget."""
    try:
        with stream:
            while chunk := stream.read(_READ_CHUNK):
                remaining = _OUTPUT_LIMIT - len(result["data"])
                result["data"].extend(chunk[:remaining])
                result["truncated"] |= len(chunk) > remaining
    except (OSError, ValueError) as exc:
        result["error"] = exc


def _render_stream(result):
    text = result["data"].decode("utf-8", errors="replace").strip()
    if result["truncated"]:
        text += f"\n[output truncated after {_OUTPUT_LIMIT} bytes from this stream]"
    return text.strip()


def run_quick_command(command, timeout=30):
    """Keep the configured shell semantics with bounded output and process lifetime."""
    from agent.redact import redact_sensitive_text
    from tools.environments.local import build_subprocess_env

    if not isinstance(command, str):
        return {"ok": False, "message": "quick command must be a string"}
    if not command.strip():
        return {"ok": False, "message": "empty command"}
    proc = None
    readers = []
    completed = False
    outputs = [{"data": bytearray(), "truncated": False} for _ in range(2)]
    try:
        # Own the POSIX process group. Keep the shell unreaped until its pipes reach
        # EOF so tree cleanup can still identify it if a descendant holds them open.
        flags = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"start_new_session": True}
        proc = subprocess.Popen(
            command.strip(), shell=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=os.getenv("TERMINAL_CWD") or os.getcwd(),
            env=build_subprocess_env(), **flags)
        deadline = time.monotonic() + timeout
        for stream, output in zip((proc.stdout, proc.stderr), outputs):
            reader = threading.Thread(target=_capture_stream, args=(stream, output),
                                      daemon=True, name="quick-command-output")
            reader.start()
            readers.append(reader)
        for reader in readers:
            reader.join(timeout=max(0, deadline - time.monotonic()))
        if any(reader.is_alive() for reader in readers):
            raise subprocess.TimeoutExpired(command, timeout)
        returncode = proc.wait(timeout=max(0, deadline - time.monotonic()))
        completed = True
        for output in outputs:
            if "error" in output:
                raise output["error"]
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": f"command timed out ({timeout}s)"}
    except Exception as exc:
        return {"ok": False, "message": redact_sensitive_text(str(exc))}
    finally:
        if proc is not None and not completed:
            # Also runs on KeyboardInterrupt; cancellation must not orphan the command.
            kill_process_tree(proc)
            try:
                proc.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
            drain_deadline = time.monotonic() + 1
            for reader in readers:
                reader.join(timeout=max(0, drain_deadline - time.monotonic()))
            # A reader owns its pipe; closing a pipe under a blocked read can hang.
            for stream in (proc.stdout, proc.stderr)[len(readers):]:
                stream.close()
    output = redact_sensitive_text("\n".join(filter(None, map(_render_stream, outputs))))
    if returncode:
        return {"ok": False, "message": output or f"command failed with exit code {returncode}",
                "returncode": returncode}
    return {"ok": True, "output": output, "returncode": returncode}
