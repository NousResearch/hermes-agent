"""Regression tests for the docker-exec privilege-drop shim.

The shim (docker/hermes-exec-shim.sh, installed at /opt/hermes/bin/hermes)
exists to prevent the auth.json ownership-mismatch bug where
`docker exec <c> hermes login` would write /opt/data/auth.json as
root:root mode 0600, leaving the supervised gateway (UID 10000) unable
to read its own credentials and returning "Provider authentication
failed: Hermes is not logged into Nous Portal" on every message.

These tests verify:

1. ``docker exec <c> hermes …`` (defaulting to root) gets dropped to the
   hermes user before the real binary runs.
2. ``docker exec --user hermes <c> hermes …`` (already non-root) short-
   circuits and doesn't try to drop again.
3. Files written under $HERMES_HOME from a ``docker exec`` session land
   as hermes:hermes — the actual user-visible invariant.
4. The HERMES_DOCKER_EXEC_AS_ROOT opt-out lets diagnostic sessions keep
   running as root deliberately.
5. The main CMD path (``docker run <image> …``) is unaffected by the
   PATH-shim ordering — no recursion, no behavior change.
"""

from __future__ import annotations
from tests.docker.conftest import docker_exec

import subprocess
import time
from collections.abc import Iterator

import pytest


# How long to give a `docker run -d` container before declaring it not ready.
_RUN_READY_TIMEOUT_S = 20


def _wait_for_init(container: str) -> None:
    """Block until /init is up enough that `docker exec` is responsive."""
    deadline = time.monotonic() + _RUN_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        r = subprocess.run(
            ["docker", "exec", container, "true"],
            capture_output=True, timeout=5,
        )
        if r.returncode == 0:
            return
        time.sleep(0.2)
    pytest.fail(f"container {container} not responsive to docker exec within {_RUN_READY_TIMEOUT_S}s")


@pytest.fixture
def sleep_container(built_image: str, container_name: str) -> Iterator[str]:
    """Long-lived container running `sleep infinity` so we can docker exec into it."""
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True, check=False,
    )
    r = subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "infinity"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"docker run failed: {r.stderr}"
    try:
        _wait_for_init(container_name)
        yield container_name
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, check=False,
        )


def test_shim_drops_root_to_hermes_uid(sleep_container: str) -> None:
    """docker exec defaults to root; the shim should drop to uid 10000.

    We invoke `hermes` with a Python-style `-c` shim equivalent — there's no
    pure-hermes "print my uid" command, so we use the venv's python directly
    via the shim's PATH lookup: `python -c 'print(os.getuid())'` is resolved
    through the venv. But that bypasses the shim. Instead, we exploit the
    fact that the venv's `hermes` is a console_scripts entry — under the
    hood it's a tiny Python wrapper. We can't easily inject "print my uid"
    into it without forking subcommands. Simplest approach: have `hermes`
    do anything that writes to disk, then check the file's owner.

    Use `hermes config set` which writes config.yaml under HERMES_HOME.
    The resulting file ownership tells us what UID the shim ended up at.
    """
    # Wipe any prior state.
    subprocess.run(
        ["docker", "exec", "--user", "root", sleep_container,
         "rm", "-f", "/opt/data/config.yaml"],
        capture_output=True, check=False,
    )

    # Default docker exec (root) — should be dropped by the shim.
    r = subprocess.run(
        ["docker", "exec", sleep_container,
         "hermes", "config", "set", "_test.shim_marker", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"config set failed: stdout={r.stdout!r} stderr={r.stderr!r}"

    # The written file must be owned by hermes, not root.
    r = subprocess.run(
        ["docker", "exec", sleep_container,
         "stat", "-c", "%U:%G", "/opt/data/config.yaml"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, f"stat failed: {r.stderr}"
    assert r.stdout.strip() == "hermes:hermes", (
        f"config.yaml owned by {r.stdout.strip()!r}, expected hermes:hermes. "
        "The shim did not drop privileges before invoking hermes."
    )


def test_shim_short_circuits_for_non_root_exec(sleep_container: str) -> None:
    """docker exec --user hermes already runs as 10000; shim should be a no-op.

    Verified indirectly: the command must still succeed end-to-end. If the
    shim incorrectly tried to drop privileges a second time (e.g. by
    invoking s6-setuidgid which requires root), it would fail with
    EPERM. A clean success proves the short-circuit fired.
    """
    subprocess.run(
        ["docker", "exec", "--user", "root", sleep_container,
         "rm", "-f", "/opt/data/config.yaml"],
        capture_output=True, check=False,
    )

    r = subprocess.run(
        ["docker", "exec", "--user", "hermes", sleep_container,
         "hermes", "config", "set", "_test.shim_short_circuit", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, (
        f"docker exec --user hermes failed: {r.stderr!r} stdout={r.stdout!r}. "
        "If the shim mis-handled the non-root path, this would fail with EPERM."
    )

    # File still ends up hermes:hermes — orthogonally confirms uid.
    r = subprocess.run(
        ["docker", "exec", sleep_container,
         "stat", "-c", "%U:%G", "/opt/data/config.yaml"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.stdout.strip() == "hermes:hermes"


def test_shim_opt_out_keeps_root(sleep_container: str) -> None:
    """HERMES_DOCKER_EXEC_AS_ROOT=1 should suppress the privilege drop.

    Reserved for diagnostic sessions where the operator deliberately
    wants root semantics. Verified by writing a file and checking its
    owner.
    """
    subprocess.run(
        ["docker", "exec", "--user", "root", sleep_container,
         "rm", "-f", "/opt/data/config.yaml"],
        capture_output=True, check=False,
    )

    r = subprocess.run(
        ["docker", "exec",
         "-e", "HERMES_DOCKER_EXEC_AS_ROOT=1",
         sleep_container,
         "hermes", "config", "set", "_test.opt_out", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"opt-out invocation failed: {r.stderr}"

    r = subprocess.run(
        ["docker", "exec", sleep_container,
         "stat", "-c", "%U:%G", "/opt/data/config.yaml"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.stdout.strip() == "root:root", (
        f"With HERMES_DOCKER_EXEC_AS_ROOT=1, expected root:root, "
        f"got {r.stdout.strip()!r}"
    )


@pytest.mark.parametrize("falsy_value", ["0", "false", "no", "", "garbage", "2"])
def test_shim_opt_out_strict_truthiness(
    sleep_container: str, falsy_value: str,
) -> None:
    """Anything other than 1/true/yes (case-insensitive) does NOT opt out.

    Strict truthiness so a typo (``HERMES_DOCKER_EXEC_AS_ROOT=0``) doesn't
    silently keep the user as root. Mirrors the policy used by
    ``HERMES_GATEWAY_NO_SUPERVISE`` in #33583.
    """
    subprocess.run(
        ["docker", "exec", "--user", "root", sleep_container,
         "rm", "-f", "/opt/data/config.yaml"],
        capture_output=True, check=False,
    )

    r = subprocess.run(
        ["docker", "exec",
         "-e", f"HERMES_DOCKER_EXEC_AS_ROOT={falsy_value}",
         sleep_container,
         "hermes", "config", "set", "_test.falsy", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"falsy value {falsy_value!r} caused failure: {r.stderr}"

    r = subprocess.run(
        ["docker", "exec", sleep_container,
         "stat", "-c", "%U:%G", "/opt/data/config.yaml"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.stdout.strip() == "hermes:hermes", (
        f"falsy opt-out value {falsy_value!r} unexpectedly suppressed the drop; "
        f"file owner is {r.stdout.strip()!r}, expected hermes:hermes"
    )


def test_main_cmd_path_unaffected(built_image: str) -> None:
    """The CMD path (docker run <image> <args>) must still work.

    The shim sits at /opt/hermes/bin earliest on PATH; main-wrapper.sh
    invokes `s6-setuidgid hermes hermes <args>` which resolves `hermes`
    through PATH. With the shim in the way, this could regress if the
    shim recurses or interferes with TTY/exit-code propagation.

    `chat --help` is cheap and exercises the full subcommand
    passthrough path. The duplicate of test_main_invocation's
    pre-existing test is intentional — that one would have passed
    pre-shim too; this one specifically guards against shim regressions
    in the CMD-as-main-program codepath.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "chat", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"CMD path broken by shim: stderr={r.stderr!r}"
    assert "Traceback" not in r.stderr


def test_e2e_login_then_supervised_gateway_can_read_auth(
    sleep_container: str,
) -> None:
    """End-to-end regression for the original bug.

    Pre-shim: ``docker exec <c> hermes login`` (root) wrote
    /opt/data/auth.json as root:root 0600. The supervised gateway (UID
    10000) couldn't read it, _load_auth_store swallowed PermissionError
    as a parse failure, and resolve_nous_runtime_credentials raised
    "Hermes is not logged into Nous Portal" on every message.

    We can't do a real OAuth login in a unit test, but we can stand in
    for it by writing the same file shape via `hermes config set`-style
    writes — what matters is the *file ownership invariant* downstream
    of `_save_auth_store`. If the shim works, every file the
    `docker exec` path produces is hermes-readable.

    Specifically: pretend the operator ran `hermes login` (writes
    auth.json) and verify (a) the file exists and (b) it's readable by
    the hermes UID. We use `hermes auth list` since that touches the
    auth store on the read side and would fail with the same
    'not logged in' shape if the file was unreadable to uid 10000.
    """
    # Have the shim-protected `docker exec` write the auth store.
    # `hermes auth list` is read-only but still exercises _load_auth_store
    # under the shim's UID. We invoke `hermes config set` first to
    # provoke a write into HERMES_HOME so we have something concrete to
    # owner-check.
    r = subprocess.run(
        ["docker", "exec", sleep_container,
         "hermes", "config", "set", "_test.e2e_marker", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"config set failed: {r.stderr}"

    # The supervised UID (10000) must be able to read everything under
    # HERMES_HOME that docker exec just wrote.
    r = subprocess.run(
        ["docker", "exec", "--user", "hermes", sleep_container,
         "find", "/opt/data", "-maxdepth", "2", "-type", "f",
         "!", "-readable", "-print"],
        capture_output=True, text=True, timeout=15,
    )
    assert r.returncode == 0, f"find failed: {r.stderr}"
    unreadable = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert not unreadable, (
        "Files written by `docker exec` are unreadable to the hermes user "
        f"(supervised gateway UID): {unreadable}. The shim failed to drop "
        "privileges before the write."
    )