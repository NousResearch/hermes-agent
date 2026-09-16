"""Native-test owner process; never run against a real Hermes profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil


KEEPALIVE = "trap 'exit 0' TERM INT; while IFS= read -r _hermes_keepalive; do :; done"


def write_record(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("raw", "backend"))
    parser.add_argument("executable")
    parser.add_argument("root", type=Path)
    parser.add_argument("home", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    args.home.mkdir(parents=True, exist_ok=True)
    task_id = args.root.name
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    os.environ["HERMES_HOME"] = str(args.home)
    os.environ["HERMES_TEST_ISOLATION"] = str(args.home)
    os.environ["TERMINAL_SANDBOX_DIR"] = str(args.home / "sandboxes")
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"

    apple = None
    if args.mode == "backend":
        import tools.credential_files as credential_files
        from tools.environments import apple_container as apple

        # Suppress real user mounts, not lifecycle or guest execution.
        credential_files.get_credential_file_mounts = lambda: []
        credential_files.get_skills_directory_mount = lambda: []
        credential_files.get_cache_directory_mounts = lambda: []
        apple._container_executable = args.executable

    real_popen = subprocess.Popen
    started = {}

    def recording_popen(command, *positional, **kwargs):
        process = real_popen(command, *positional, **kwargs)
        argv = list(command)
        if (
            len(argv) > 1 and argv[0] == args.executable
            and argv[1] == "run" and "--name" in argv
        ):
            try:
                created = psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                created = None
            started.update(
                name=argv[argv.index("--name") + 1],
                client_pid=process.pid,
                client_created=created,
            )
            write_record(args.root / "started.json", started)
        return process

    # Observe real spawns, including subprocess.run on the unfixed backend.
    # Record exact fixture identities for teardown without replacing the CLI.
    subprocess.Popen = recording_popen

    persistent_file = None
    if args.mode == "raw":
        name = "hermes-eof-test-" + uuid.uuid4().hex
        command = [
            args.executable, "run", "--name", name,
            "--interactive", "--rm", "--entrypoint", "bash",
            "--cpus", "1", "--memory", "1024M", "--network", "none",
            "--read-only", "--tmpfs", "/tmp", "--tmpfs", "/root",
            "python:3.11-slim-bookworm", "-c", KEEPALIVE,
        ]
        lifetime = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=sys.stderr, bufsize=0, close_fds=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + 300
        while True:
            if lifetime.poll() is not None:
                raise RuntimeError("native lifetime client exited before readiness")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("native lifetime client did not become ready")
            try:
                result = subprocess.run(
                    [args.executable, "exec", name, "bash", "-c", ":"],
                    capture_output=True, text=True, timeout=min(5, remaining),
                )
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0 and lifetime.poll() is None:
                break
            time.sleep(0.2)
        assert lifetime.stdin is not None
        pipe_inheritable = os.get_inheritable(lifetime.stdin.fileno())
    else:
        assert apple is not None
        environment = apple.AppleContainerEnvironment(
            cpu=1, memory=1024, persistent_filesystem=True, task_id=task_id,
            extra_args=["--network", "none"],
        )
        name = environment._container_name
        result = environment.execute("printf preserved > /workspace/owner.txt")
        assert result.get("returncode") == 0, result
        persistent_file = str(
            args.home / "sandboxes" / "apple_container" / task_id
            / "workspace" / "owner.txt"
        )
        lifetime = getattr(environment, "_run_process", None)
        pipe_inheritable = (
            os.get_inheritable(lifetime.stdin.fileno())
            if lifetime is not None and lifetime.stdin is not None else None
        )

    # CLOEXEC must prevent this unrelated exec'd child from retaining the
    # lifetime writer even when the caller requests close_fds=False.
    spectator = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, close_fds=False, start_new_session=True,
    )
    write_record(
        args.root / "ready.json",
        {
            **started,
            "name": name,
            "spectator_pid": spectator.pid,
            "spectator_created": psutil.Process(spectator.pid).create_time(),
            "pipe_inheritable": pipe_inheritable,
            "persistent_file": persistent_file,
        },
    )
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
