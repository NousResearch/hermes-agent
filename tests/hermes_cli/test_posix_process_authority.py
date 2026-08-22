from __future__ import annotations

import json
import os
import select
import signal
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


def read_json_line(proc: subprocess.Popen[str], timeout: float = 10.0) -> dict[str, object]:
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        stderr = proc.stderr.read() if proc.stderr and proc.poll() is not None else ""
        raise AssertionError(f"authority process produced no output: {stderr}")
    line = proc.stdout.readline()
    assert line, proc.stderr.read() if proc.stderr else "authority process produced no output"
    return json.loads(line)


def spawn_authority(script: str, generation: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", bootstrapped(script)],
        env=authority_env(generation),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def stop_supervisor(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        force_signal = getattr(signal, "SIGUSR2", signal.SIGTERM)
        proc.send_signal(force_signal)
        proc.wait(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_nested_containment_propagates_and_root_teardown_reaps_scope():
    control = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    child_code = bootstrapped(
        """
        import json, os, subprocess, sys, time
        grandchild = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        print(json.dumps({
            "child": os.getpid(),
            "child_pgid": os.getpgrp(),
            "grandchild_owner": grandchild.pid,
        }), flush=True)
        time.sleep(60)
        """
    )
    proc = spawn_authority(
        f"""
        import json, os, subprocess, sys, time
        child = subprocess.Popen(
            [sys.executable, "-c", {child_code!r}],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        nested = json.loads(child.stdout.readline())
        print(json.dumps({{
            "backend": os.getpid(),
            "backend_pgid": os.getpgrp(),
            "child_owner": child.pid,
            "child_owner_pgid": os.getpgid(child.pid),
            **nested,
        }}), flush=True)
        time.sleep(60)
        """,
        "generation-posix-contained-01",
    )
    topology = read_json_line(proc)
    try:
        backend_pgid = int(topology["backend_pgid"])
        child = int(topology["child"])
        child_pgid = int(topology["child_pgid"])
        child_owner = int(topology["child_owner"])
        grandchild_owner = int(topology["grandchild_owner"])
        assert int(topology["backend"]) != proc.pid
        assert int(topology["child_owner_pgid"]) == backend_pgid
        assert child_pgid == child
        assert child_pgid != backend_pgid
        assert os.getpgid(grandchild_owner) == child_pgid

        proc.terminate()
        proc.wait(timeout=12)
        assert wait_not_live(int(topology["backend"]))
        assert wait_not_live(child_owner)
        assert wait_not_live(child)
        assert wait_not_live(grandchild_owner)
        assert process_is_live(control.pid), "unrelated process must not be mutated"
    finally:
        stop_supervisor(proc)
        if control.poll() is None:
            control.terminate()
            control.wait(timeout=8)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_positive_process_group_and_posix_spawn_setpgroup_cannot_escape(tmp_path: Path):
    spawn_result = tmp_path / "spawn-pgid.txt"
    proc = spawn_authority(
        f"""
        import json, os, subprocess, sys, time
        popen_child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            process_group=987654,
        )
        target = {str(spawn_result)!r}
        code = (
            "import os; open("
            + repr(target)
            + ", 'w', encoding='utf-8').write(str(os.getpgrp()))"
        )
        spawned = os.posix_spawn(
            sys.executable,
            [sys.executable, "-c", code],
            os.environ.copy(),
            setpgroup=987654,
        )
        os.waitpid(spawned, 0)
        print(json.dumps({{
            "backend": os.getpid(),
            "backend_pgid": os.getpgrp(),
            "popen_child": popen_child.pid,
            "spawn_pgid": int(open(target, encoding="utf-8").read()),
        }}), flush=True)
        time.sleep(60)
        """,
        "generation-posix-positive-pgroup-01",
    )
    topology = read_json_line(proc)
    try:
        backend_pgid = int(topology["backend_pgid"])
        assert os.getpgid(int(topology["popen_child"])) == backend_pgid
        assert int(topology["spawn_pgid"]) == backend_pgid
    finally:
        stop_supervisor(proc)
        assert wait_not_live(int(topology["popen_child"]))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_intentional_detach_helper_completes_child_side_handoff_and_preserves_env(tmp_path: Path):
    stop_file = tmp_path / "stop-transferred"
    transferred_code = textwrap.dedent(
        f"""
        import json, os, time
        print(json.dumps({{
            "pid": os.getpid(),
            "sid": os.getsid(0),
            "pgid": os.getpgrp(),
            "marker": os.environ.get("HERMES_TRANSFER_TEST_MARKER"),
            "receipt": os.environ.get("HERMES_DESKTOP_PROCESS_TRANSFER_RECEIPT"),
        }}), flush=True)
        stop = {str(stop_file)!r}
        while not os.path.exists(stop):
            time.sleep(0.05)
        """
    )
    proc = spawn_authority(
        f"""
        import json, os, subprocess, sys, time
        from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

        env = os.environ.copy()
        env["HERMES_TRANSFER_TEST_MARKER"] = "preserved"
        child = subprocess.Popen(
            [sys.executable, "-c", {transferred_code!r}],
            env=env,
            stdout=subprocess.PIPE,
            text=True,
            **windows_detach_popen_kwargs(),
        )
        accepted = json.loads(child.stdout.readline())
        print(json.dumps({{
            "backend": os.getpid(),
            "backend_pgid": os.getpgrp(),
            "owner": child.pid,
            **accepted,
        }}), flush=True)
        time.sleep(60)
        """,
        "generation-posix-transfer-01",
    )
    topology = read_json_line(proc)
    transferred = int(topology["pid"])
    owner = int(topology["owner"])
    try:
        assert int(topology["sid"]) == owner
        assert int(topology["pgid"]) == transferred
        assert transferred != owner
        assert owner != int(topology["backend_pgid"])
        assert topology["marker"] == "preserved"
        assert str(topology["receipt"]).startswith(
            "desktop-posix-transfer-v1:hermes-intentional-detached-child"
        )

        proc.terminate()
        proc.wait(timeout=12)
        assert wait_not_live(int(topology["backend"]))
        assert process_is_live(owner)
        assert process_is_live(transferred)

        stop_file.touch()
        assert wait_not_live(transferred), "the receiving process must own and finish its own lifecycle"
        assert wait_not_live(owner), "the receiving supervisor must reap and follow its target"
    finally:
        stop_supervisor(proc)
        stop_file.touch(exist_ok=True)
        assert wait_not_live(transferred)
        assert wait_not_live(owner)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_transferred_popen_kill_routes_through_receiving_owner():
    target_code = textwrap.dedent(
        """
        import json, os, time
        print(json.dumps({"pid": os.getpid()}), flush=True)
        time.sleep(60)
        """
    )
    proc = spawn_authority(
        f"""
        import json, subprocess, sys
        from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

        child = subprocess.Popen(
            [sys.executable, "-c", {target_code!r}],
            stdout=subprocess.PIPE,
            text=True,
            **windows_detach_popen_kwargs(),
        )
        target = json.loads(child.stdout.readline())
        owner = child.pid
        child.kill()
        status = child.wait(timeout=8)
        print(json.dumps({{
            "owner": owner,
            "target": target["pid"],
            "status": status,
        }}), flush=True)
        """,
        "generation-posix-transfer-kill-01",
    )
    topology = read_json_line(proc, timeout=12)
    proc.wait(timeout=12)
    assert int(topology["status"]) != 0
    assert wait_not_live(int(topology["target"]))
    assert wait_not_live(int(topology["owner"]))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_transfer_without_a_runnable_receiving_target_is_rejected():
    proc = spawn_authority(
        """
        import json, subprocess
        from hermes_cli.posix_process_authority import (
            ProcessAuthorityError,
            begin_process_transfer,
            desktop_child_env,
        )

        grant = begin_process_transfer("test-missing-target")
        env = desktop_child_env(lifetime="transferred", transfer=grant)
        try:
            subprocess.Popen(
                ["/definitely/missing/hermes-transfer-target"],
                env=env,
                start_new_session=True,
            )
        except ProcessAuthorityError as exc:
            print(json.dumps({"error": str(exc)}), flush=True)
        else:
            raise AssertionError("transfer to a missing target unexpectedly succeeded")
        """,
        "generation-posix-transfer-fail-01",
    )
    result = read_json_line(proc, timeout=8)
    proc.wait(timeout=8)
    assert "could not exec transferred target" in str(result["error"])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_generation_n_teardown_cannot_touch_generation_n_plus_one():
    script = """
        import json, os, subprocess, sys, time
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        print(json.dumps({
            "backend": os.getpid(),
            "child": child.pid,
            "pgid": os.getpgrp(),
        }), flush=True)
        time.sleep(60)
    """
    first = spawn_authority(script, "generation-posix-fence-0001")
    second = spawn_authority(script, "generation-posix-fence-0002")
    first_topology = read_json_line(first)
    second_topology = read_json_line(second)
    try:
        assert int(first_topology["pgid"]) != int(second_topology["pgid"])
        first.send_signal(getattr(signal, "SIGUSR2", signal.SIGTERM))
        first.wait(timeout=10)
        assert wait_not_live(int(first_topology["backend"]))
        assert wait_not_live(int(first_topology["child"]))
        assert process_is_live(int(second_topology["backend"]))
        assert process_is_live(int(second_topology["child"]))
    finally:
        stop_supervisor(first)
        stop_supervisor(second)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX authority")
def test_natural_backend_exit_reaps_contained_residue():
    proc = spawn_authority(
        """
        import json, os, subprocess, sys
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        print(json.dumps({"backend": os.getpid(), "child": child.pid}), flush=True)
        """,
        "generation-posix-natural-01",
    )
    topology = read_json_line(proc)
    proc.wait(timeout=10)
    assert wait_not_live(int(topology["backend"]))
    assert wait_not_live(int(topology["child"]))


def test_caller_supplied_receipt_is_rejected_as_false_authority():
    with pytest.raises(authority.ProcessAuthorityError, match="caller-supplied transfer receipts"):
        authority.desktop_child_env(
            lifetime="transferred",
            transfer_receipt="service-manager:unit-42",
        )
    with pytest.raises(authority.ProcessAuthorityError, match="requires a transfer grant"):
        authority.desktop_child_env(lifetime="transferred")
    with pytest.raises(authority.ProcessAuthorityError, match="requires a transfer grant"):
        authority.desktop_child_env(lifetime="foreign")
