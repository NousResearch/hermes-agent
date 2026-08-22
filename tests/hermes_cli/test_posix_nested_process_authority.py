from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import posix_process_authority as authority

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_FILE = PROJECT_ROOT / "hermes_cli" / "desktop_bootstrap" / "sitecustomize.py"


def authority_env(generation: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            authority.AUTHORITY_MODE_ENV: authority.AUTHORITY_MODE,
            authority.GENERATION_ENV: generation,
            authority.PARENT_PID_ENV: str(os.getpid()),
            authority.PARENT_STARTED_AT_ENV: "1700000000000",
            "PYTHONPATH": os.pathsep.join(
                [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        }
    )
    return env


def bootstrapped(code: str) -> str:
    return f"import runpy\nrunpy.run_path({str(BOOTSTRAP_FILE)!r})\n" + textwrap.dedent(code)


def spawn_authority(script: str, generation: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", bootstrapped(script)],
        env=authority_env(generation),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def read_json_line(proc: subprocess.Popen[str], timeout: float = 10.0) -> dict[str, object]:
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        stderr = proc.stderr.read() if proc.stderr and proc.poll() is not None else ""
        raise AssertionError(f"authority process produced no output: {stderr}")
    line = proc.stdout.readline()
    assert line, proc.stderr.read() if proc.stderr else "authority process produced no output"
    return json.loads(line)


def process_is_live(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def wait_not_live(pid: int, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_is_live(pid):
            return True
        time.sleep(0.05)
    return not process_is_live(pid)


def stop_supervisor(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=12)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_nested_owned_popen_preserves_local_scope_and_outer_reap():
    target_code = textwrap.dedent(
        """
        import json, os, time
        print(json.dumps({
            "target": os.getpid(),
            "target_sid": os.getsid(0),
            "target_pgid": os.getpgrp(),
        }), flush=True)
        time.sleep(60)
        """
    )
    proc = spawn_authority(
        f"""
        import json, os, subprocess, sys, time

        control = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
        )
        nested = subprocess.Popen(
            [sys.executable, "-c", {target_code!r}],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        target = json.loads(nested.stdout.readline())
        print(json.dumps({{
            "backend": os.getpid(),
            "backend_pgid": os.getpgrp(),
            "control": control.pid,
            "owner": nested.pid,
            "owner_pgid": os.getpgid(nested.pid),
            "nested_owned": bool(getattr(nested, "__hermes_nested_owned__", False)),
            **target,
        }}), flush=True)

        nested.terminate()
        nested.wait(timeout=8)
        print(json.dumps({{
            "control_alive": control.poll() is None,
            "backend_alive": True,
        }}), flush=True)
        time.sleep(60)
        """,
        "generation-posix-nested-owned-01",
    )
    topology = read_json_line(proc)
    local_cleanup = read_json_line(proc)
    try:
        backend_pgid = int(topology["backend_pgid"])
        target = int(topology["target"])
        assert topology["nested_owned"] is True
        assert int(topology["owner_pgid"]) == backend_pgid
        assert int(topology["target_sid"]) == target
        assert int(topology["target_pgid"]) == target
        assert target != backend_pgid
        assert local_cleanup == {"control_alive": True, "backend_alive": True}
        assert wait_not_live(target)
        assert process_is_live(int(topology["control"]))
    finally:
        stop_supervisor(proc)
    assert wait_not_live(int(topology["control"]))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_mcp_watchdog_cleanup_cannot_widen_to_desktop_root():
    target_code = textwrap.dedent(
        """
        import json, os, time
        print(json.dumps({
            "target": os.getpid(),
            "target_pgid": os.getpgrp(),
        }), flush=True)
        time.sleep(60)
        """
    )
    proc = spawn_authority(
        f"""
        import json, os, subprocess, sys, time
        from tools.mcp_stdio_watchdog import _terminate_process_group

        control = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
        )
        child = subprocess.Popen(
            [sys.executable, "-c", {target_code!r}],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        target = json.loads(child.stdout.readline())
        _terminate_process_group(child)
        print(json.dumps({{
            "backend": os.getpid(),
            "backend_pgid": os.getpgrp(),
            "control": control.pid,
            "control_alive": control.poll() is None,
            "nested_owned": bool(getattr(child, "__hermes_nested_owned__", False)),
            **target,
        }}), flush=True)
        time.sleep(60)
        """,
        "generation-posix-mcp-nested-01",
    )
    topology = read_json_line(proc, timeout=12)
    try:
        assert topology["nested_owned"] is True
        assert topology["control_alive"] is True
        assert int(topology["target_pgid"]) == int(topology["target"])
        assert int(topology["target_pgid"]) != int(topology["backend_pgid"])
        assert wait_not_live(int(topology["target"]))
        assert process_is_live(int(topology["backend"]))
        assert process_is_live(int(topology["control"]))
    finally:
        stop_supervisor(proc)
    assert wait_not_live(int(topology["control"]))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_asyncio_lsp_shape_retains_private_scope_under_guard():
    target_code = textwrap.dedent(
        """
        import json, os, time
        print(json.dumps({
            "target": os.getpid(),
            "target_sid": os.getsid(0),
            "target_pgid": os.getpgrp(),
        }), flush=True)
        time.sleep(60)
        """
    )
    proc = spawn_authority(
        f"""
        import asyncio, json, os, sys

        async def main():
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                {target_code!r},
                stdout=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            line = await child.stdout.readline()
            target = json.loads(line.decode())
            print(json.dumps({{
                "backend": os.getpid(),
                "backend_pgid": os.getpgrp(),
                "owner": child.pid,
                "owner_pgid": os.getpgid(child.pid),
                **target,
            }}), flush=True)
            child.terminate()
            await asyncio.wait_for(child.wait(), timeout=8)

        asyncio.run(main())
        """,
        "generation-posix-lsp-shape-01",
    )
    topology = read_json_line(proc, timeout=12)
    proc.wait(timeout=12)
    target = int(topology["target"])
    assert int(topology["owner_pgid"]) == int(topology["backend_pgid"])
    assert int(topology["target_sid"]) == target
    assert int(topology["target_pgid"]) == target
    assert target != int(topology["backend_pgid"])
    assert wait_not_live(target)
