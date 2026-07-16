"""Fixed tmux entrypoint for configured governed workers."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


REQUIRED = {
    "worker_id",
    "campaign_id",
    "step_index",
    "role",
    "command",
    "cwd",
    "tmux_session",
    "provider",
    "model",
    "prompt_path",
    "prompt_hash",
    "mutable_manifest",
    "output_path",
    "receipt_path",
    "heartbeat_path",
    "nonce",
    "read_only",
    "protected_roots",
}


def _seatbelt_profile(protected_roots: list[str], writable_root: Path) -> str:
    lines = ["(version 1)", "(allow default)"]
    for root in protected_roots:
        lines.append(f"(deny file-write* (subpath {json.dumps(root)}))")
    lines.append(f"(allow file-write* (subpath {json.dumps(str(writable_root))}))")
    return "\n".join(lines)


def _reviewer_command(command: list[str], launch_path: Path, spec: dict) -> list[str]:
    protected = spec["protected_roots"]
    if (
        not isinstance(protected, list)
        or not protected
        or not all(
            isinstance(root, str) and Path(root).is_absolute() for root in protected
        )
        or str(Path(spec["cwd"]).resolve())
        not in {str(Path(root).resolve()) for root in protected}
    ):
        raise SystemExit("reviewer launch lacks protected product roots")
    writable_root = launch_path.parent.resolve()
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
        return [
            "/usr/bin/sandbox-exec",
            "-p",
            _seatbelt_profile(protected, writable_root),
            *command,
            str(launch_path),
        ]
    bubblewrap = shutil.which("bwrap")
    if sys.platform.startswith("linux") and bubblewrap:
        return [
            bubblewrap,
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(writable_root),
            str(writable_root),
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            spec["cwd"],
            "--",
            *command,
            str(launch_path),
        ]
    raise SystemExit("no supported hard read-only reviewer sandbox is available")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("worker_exec requires exactly one launch spec")
    launch_path = Path(sys.argv[1]).resolve()
    spec = json.loads(launch_path.read_text(encoding="utf-8"))
    if set(spec) != REQUIRED or spec["role"] not in {"writer", "reviewer"}:
        raise SystemExit("invalid governed worker launch spec")
    command = spec["command"]
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(p, str) and p for p in command)
    ):
        raise SystemExit("invalid governed worker command")
    if bool(spec["read_only"]) != (spec["role"] == "reviewer"):
        raise SystemExit("role/capability mismatch")
    env = os.environ.copy()
    env["HERMES_CONDUCTOR_ROLE"] = spec["role"]
    env["HERMES_CONDUCTOR_READ_ONLY"] = "1" if spec["read_only"] else "0"
    os.chdir(spec["cwd"])
    argv = (
        _reviewer_command(command, launch_path, spec)
        if spec["read_only"]
        else [*command, str(launch_path)]
    )
    os.execvpe(argv[0], argv, env)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
