"""Process-tree cleanup for timed-out shell hooks (port of openai/codex#37527).

Timing out a shell hook must not leave descendant processes running after the
hook itself is stopped. ``_spawn`` places the hook in its own process group on
POSIX (``process_group=0``) and, on timeout, ``kill_process_tree`` signals the
whole group — gated on the child actually leading its own group so a
shared-group spawn can never take down unrelated processes. Hooks that
complete within their timeout keep their descendants (intentionally detached
helpers survive a successful run).

These tests use REAL subprocesses (no mocks): group membership and survival
semantics cannot be mocked.
"""

import os
import sys
import textwrap
import time

import pytest

from agent.shell_hooks import ShellHookSpec, _spawn

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX process-group semantics"
)


def _spec(command: str, timeout: int = 2) -> ShellHookSpec:
    return ShellHookSpec(
        event="post_tool_call",
        command=command,
        matcher=None,
        timeout=timeout,
        fail_closed=False,
    )


def _write_forking_script(tmp_path, stall_after: bool):
    """A hook that forks a long-lived descendant, then stalls or exits."""
    marker = tmp_path / "child.pid"
    script = tmp_path / "hook.sh"
    tail = "sleep 300" if stall_after else "exit 0"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            sleep 300 &
            echo $! > {marker}
            {tail}
            """
        )
    )
    script.chmod(0o755)
    return script, marker


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _read_marker(marker, timeout=5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text().strip():
            return int(marker.read_text().strip())
        time.sleep(0.05)
    raise AssertionError("hook script never wrote its descendant pid")


def test_timeout_kills_descendants(tmp_path):
    """A hook timeout must take the forked descendant down with the hook."""
    script, marker = _write_forking_script(tmp_path, stall_after=True)

    t0 = time.monotonic()
    r = _spawn(_spec(str(script), timeout=1), "{}")
    elapsed = time.monotonic() - t0

    assert r["timed_out"] is True
    assert elapsed < 30, "timeout cleanup must not hang on held pipes"

    child_pid = _read_marker(marker)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and _pid_alive(child_pid):
        time.sleep(0.05)
    alive = _pid_alive(child_pid)
    if alive:  # cleanup so a failure doesn't leak a 300s sleeper
        os.kill(child_pid, 9)
    assert not alive, f"descendant {child_pid} survived hook timeout"


@pytest.mark.live_system_guard_bypass  # cleanup-kills a helper reparented to init
def test_successful_hook_preserves_detached_helpers(tmp_path):
    """A hook that completes in time keeps its intentionally detached helpers.

    Mirrors codex#37527's "preserve descendants on success" semantics: tree
    cleanup fires only on the timeout/error path.
    """
    script, marker = _write_forking_script(tmp_path, stall_after=False)
    # Detach the helper's stdio so the pipes reach EOF despite the survivor.
    script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            sleep 300 > /dev/null 2>&1 < /dev/null &
            echo $! > {marker}
            exit 0
            """
        )
    )

    r = _spawn(_spec(str(script), timeout=10), "{}")
    assert r["timed_out"] is False
    assert r["returncode"] == 0

    child_pid = _read_marker(marker)
    time.sleep(0.3)
    alive = _pid_alive(child_pid)
    if alive:
        os.kill(child_pid, 9)
    assert alive, "successful hook's detached helper must survive"


def test_hook_child_leads_own_process_group(tmp_path):
    """The hook child must lead its own group (killpg ownership precondition)."""
    script = tmp_path / "pgid.sh"
    script.write_text("#!/bin/bash\necho \"$$ $(ps -o pgid= -p $$ | tr -d ' ')\"\n")
    script.chmod(0o755)

    r = _spawn(_spec(str(script), timeout=10), "{}")
    assert r["returncode"] == 0
    pid, pgid = r["stdout"].split()
    assert pid == pgid, f"hook child pid={pid} does not lead its group pgid={pgid}"
    assert int(pgid) != os.getpgid(0), "hook child must not share our group"


def test_fast_path_contract_unchanged(tmp_path):
    """stdin JSON delivery, stdout/stderr capture, and exit codes still work."""
    script = tmp_path / "echoer.sh"
    script.write_text(
        "#!/bin/bash\ncat\necho errline >&2\nexit 3\n"
    )
    script.chmod(0o755)

    r = _spawn(_spec(str(script), timeout=10), '{"tool_name": "terminal"}')
    assert r["returncode"] == 3
    assert r["stdout"] == '{"tool_name": "terminal"}'
    assert "errline" in r["stderr"]
    assert r["error"] is None
    assert r["timed_out"] is False


def test_missing_command_still_fails_open():
    r = _spawn(_spec("/nonexistent/hook-command-xyz", timeout=2), "{}")
    assert r["error"] == "command not found"
    assert r["returncode"] is None
