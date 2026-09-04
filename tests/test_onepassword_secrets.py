"""Hermetic tests for the 1Password (`op` CLI) secret source.

We never invoke the real ``op`` binary: the process helper is mocked so the
suite stays fast and offline-safe.  A live resolve is exercised manually via
``hermes secrets onepassword sync`` outside of pytest.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

import pytest


# Make the worktree importable without depending on the installed wheel.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import _op_subreaper as op_helper  # noqa: E402
from agent.secret_sources import onepassword as op  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    with tempfile.TemporaryDirectory(prefix=".op-") as root_text:
        root = Path(root_text)
        hermes_home = root / "home"
        runtime_dir = root / "run"
        hermes_home.mkdir(mode=0o700)
        runtime_dir.mkdir(mode=0o700)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
        op._reset_cache_for_tests(hermes_home)
        yield
        op._reset_cache_for_tests(hermes_home)


@pytest.fixture(autouse=True)
def _clean_op_env(monkeypatch):
    """Start every test from a known 1Password auth state."""
    for key in list(os.environ):
        if key.startswith("OP_SESSION_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    monkeypatch.delenv("OP_ACCOUNT", raising=False)
    monkeypatch.delenv("OP_CONNECT_HOST", raising=False)
    monkeypatch.delenv("OP_CONNECT_TOKEN", raising=False)
    yield


def _ok(value: str):
    return mock.Mock(returncode=0, stdout=value, stderr="")


def _err(code: int, stderr: str):
    return mock.Mock(returncode=code, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------


def test_validate_references_filters_bad_names_and_refs():
    refs = {
        "OPENAI_API_KEY": "op://Private/OpenAI/api key",
        "1BAD_NAME": "op://Private/x/y",  # bad env name
        "HAS SPACE": "op://Private/x/y",  # bad env name
        "NOT_A_REF": "https://example.com",  # not op://
        "WHITESPACE": "  op://Private/z/field  ",  # stripped + kept
    }
    valid, warnings = op._validate_references(refs)
    assert valid == {
        "OPENAI_API_KEY": "op://Private/OpenAI/api key",
        "WHITESPACE": "op://Private/z/field",
    }
    assert len(warnings) == 3


# ---------------------------------------------------------------------------
# fetch_onepassword_secrets
# ---------------------------------------------------------------------------


def test_op_child_env_disables_cli_daemon_cache(monkeypatch):
    """Gateway-owned op reads must not leave 24h `op daemon` cgroup children."""
    monkeypatch.setenv("OP_CACHE", "true")
    monkeypatch.setenv("OP_LOAD_DESKTOP_APP_SETTINGS", "true")
    monkeypatch.setenv("OP_BIOMETRIC_UNLOCK_ENABLED", "true")

    child_env = op._op_child_env("service-account-token")

    assert child_env["OP_CACHE"] == "false"
    assert child_env["OP_LOAD_DESKTOP_APP_SETTINGS"] == "false"
    assert child_env["OP_BIOMETRIC_UNLOCK_ENABLED"] == "false"
    assert child_env["OP_SERVICE_ACCOUNT_TOKEN"] == "service-account-token"


def test_non_linux_posix_preserves_native_runtime_behavior(monkeypatch):
    monkeypatch.setattr(op.sys, "platform", "darwin")
    monkeypatch.setattr(
        op,
        "_safe_op_runtime_root",
        lambda source_env=None: pytest.fail(
            "Darwin must not require the Linux private socket path"
        ),
    )

    child_env = op._op_child_env("service-account-token")

    assert child_env["OP_CACHE"] == "false"
    assert child_env["OP_SERVICE_ACCOUNT_TOKEN"] == "service-account-token"
    assert "OP_SOCK" not in child_env


def test_windows_preserves_native_runtime_behavior(monkeypatch):
    monkeypatch.setattr(op.sys, "platform", "win32")
    monkeypatch.setattr(
        op,
        "_safe_op_runtime_root",
        lambda source_env=None: pytest.fail(
            "Windows must not require the Linux private socket path"
        ),
    )

    child_env = op._op_child_env("service-account-token")

    assert child_env["OP_CACHE"] == "false"
    assert child_env["OP_SERVICE_ACCOUNT_TOKEN"] == "service-account-token"
    assert "OP_SOCK" not in child_env


@pytest.mark.skipif(sys.platform != "linux", reason="Linux runtime fallback only")
def test_linux_runtime_root_falls_back_to_private_hermes_home(monkeypatch):
    hermes_home = Path(os.environ["HERMES_HOME"])
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(op, "_runtime_root_candidates", lambda source_env, uid: [])

    runtime_root = op._safe_op_runtime_root()

    assert runtime_root == (hermes_home / ".runtime").resolve()
    assert runtime_root.is_dir()
    assert stat.S_IMODE(runtime_root.stat().st_mode) == 0o700


@pytest.mark.skipif(sys.platform != "linux", reason="Linux runtime fallback only")
def test_runtime_root_rejects_nonsticky_writable_ancestry(tmp_path):
    writable_parent = tmp_path / "writable-parent"
    writable_parent.mkdir(mode=0o777)
    writable_parent.chmod(0o777)
    candidate = writable_parent / "runtime"
    candidate.mkdir(mode=0o700)

    assert op._runtime_root_is_safe_and_short(candidate, os.getuid()) is None


@pytest.mark.skipif(sys.platform != "linux", reason="Linux runtime fallback only")
def test_symlinked_fallback_runtime_is_rejected(monkeypatch, tmp_path):
    hermes_home = Path(os.environ["HERMES_HOME"])
    target = tmp_path / "runtime-target"
    target.mkdir(mode=0o700)
    (hermes_home / ".runtime").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(op, "_runtime_root_candidates", lambda source_env, uid: [])

    with pytest.raises(RuntimeError, match="setup HOLD"):
        op._create_op_runtime_namespace()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux runtime binding only")
def test_open_runtime_root_rejects_fstat_identity_mismatch(monkeypatch):
    runtime_root = Path(os.environ["XDG_RUNTIME_DIR"]).resolve()
    original_fstat = os.fstat

    def mismatched_fstat(fd):
        actual = original_fstat(fd)
        values = list(actual)
        values[1] = actual.st_ino + 1
        return os.stat_result(values)

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(op.os, "fstat", mismatched_fstat)
        with pytest.raises(
            RuntimeError, match="runtime root identity drift.*setup HOLD"
        ):
            op._open_bound_runtime_root(runtime_root, os.getuid())


@pytest.mark.skipif(sys.platform != "linux", reason="Linux runtime binding only")
def test_runtime_procfd_binding_survives_path_swap_and_restore():
    namespace = op._create_op_runtime_namespace()
    replacement = namespace.runtime_dir
    moved = replacement.with_name(replacement.name + "-moved")
    os.rename(replacement, moved)
    replacement.mkdir(mode=0o700)
    try:
        assert namespace.child_runtime_dir.resolve(strict=True) == moved
        assert namespace.socket_path.parent.resolve(strict=True) == moved
        with pytest.raises(RuntimeError, match="runtime identity drift.*cleanup HOLD"):
            op._validate_runtime_namespace(namespace)
    finally:
        replacement.rmdir()
        os.rename(moved, replacement)

    op._validate_runtime_namespace(namespace)
    op._remove_op_runtime_namespace(namespace)
    os.close(namespace.dir_fd)


@pytest.mark.skipif(sys.platform != "linux", reason="private socket is Linux-only")
def test_op_read_pins_private_socket_and_explicitly_disables_cache(
    monkeypatch, tmp_path
):
    """A shared global op socket must not resurrect a cgroup-resident daemon."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    namespace = mock.Mock(
        runtime_dir=runtime_dir,
        socket_path=runtime_dir / "hermes-op-test.sock",
    )
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _ok("secret-value\n")

    monkeypatch.setattr(op, "_run_op_process", fake_run)
    child_env = op._op_child_env("token", namespace)

    assert (
        op._run_op_read(
            fake_op,
            "op://V/I/F",
            token_value="token",
            child_env=child_env,
        )
        == "secret-value"
    )
    assert captured["cmd"][:3] == [str(fake_op), "--cache=false", "read"]
    assert captured["env"]["OP_CACHE"] == "false"
    socket_path = Path(captured["env"]["OP_SOCK"])
    assert socket_path.is_absolute()
    assert socket_path.parent == socket_path.parent.resolve()
    assert socket_path != Path("/run/user") / str(os.getuid()) / "op-daemon.sock"
    assert socket_path.name.startswith("hermes-op-")
    assert socket_path.suffix == ".sock"


@pytest.mark.skipif(sys.platform != "linux", reason="managed namespaces are Linux-only")
def test_runtime_namespaces_are_unique(monkeypatch):
    first = op._create_op_runtime_namespace()
    second = op._create_op_runtime_namespace()
    try:
        assert first.runtime_dir != second.runtime_dir
        assert first.socket_path != second.socket_path
        assert first.dir_ino != second.dir_ino
    finally:
        for namespace in (first, second):
            op._remove_op_runtime_namespace(namespace)
            os.close(namespace.dir_fd)


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_fetch_uses_one_private_xdg_namespace_and_always_cleans(monkeypatch, tmp_path):
    """All reads share one isolated runtime and cleanup runs after failures too."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    runtime_dir = tmp_path / "managed-runtime"
    runtime_dir.mkdir(mode=0o700)
    namespace = mock.Mock(
        runtime_dir=runtime_dir,
        socket_path=runtime_dir / "op.sock",
    )
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp/foreign-runtime-alias")
    monkeypatch.setattr(op, "_create_op_runtime_namespace", lambda: namespace)
    cleanup_calls = []
    monkeypatch.setattr(
        op,
        "_cleanup_op_runtime_namespace",
        lambda ns, binary: cleanup_calls.append((ns, binary)),
    )
    child_envs = []

    def fake_run(cmd, **kwargs):
        child_envs.append(dict(kwargs["env"]))
        ref = cmd[cmd.index("--") + 1]
        if ref.endswith("/bad"):
            return _err(1, "read failed")
        return _ok("secret-value")

    monkeypatch.setattr(op, "_run_op_process", fake_run)

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"GOOD": "op://V/I/good", "BAD": "op://V/I/bad"},
        binary=fake_op,
        use_cache=False,
    )

    assert secrets == {"GOOD": "secret-value"}
    assert len(warnings) == 1
    assert len(child_envs) == 2
    assert {env["XDG_RUNTIME_DIR"] for env in child_envs} == {str(runtime_dir)}
    assert {env["OP_SOCK"] for env in child_envs} == {str(namespace.socket_path)}
    assert {env["OP_CACHE"] for env in child_envs} == {"false"}
    assert cleanup_calls == [(namespace, fake_op)]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux op lifecycle serialization")
def test_concurrent_fetch_batches_are_process_serialized(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "token")
    monkeypatch.setattr(op, "_create_op_runtime_namespace", mock.Mock)
    monkeypatch.setattr(op, "_op_child_env", lambda *_args: {})
    monkeypatch.setattr(op, "_cleanup_op_runtime_namespace", lambda *_args: None)

    first_entered = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    state = {"active": 0, "maximum": 0, "calls": 0}

    def fake_read(*_args, **_kwargs):
        with state_lock:
            state["calls"] += 1
            call = state["calls"]
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        try:
            if call == 1:
                first_entered.set()
                assert release_first.wait(timeout=2.0)
            return "secret"
        finally:
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(op, "_run_op_read", fake_read)
    results = []

    def fetch():
        results.append(
            op.fetch_onepassword_secrets(
                references={"KEY": "op://V/I/F"},
                binary=fake_op,
                use_cache=False,
                home_path=tmp_path,
            )
        )

    first = threading.Thread(target=fetch)
    second = threading.Thread(target=fetch)
    first.start()
    assert first_entered.wait(timeout=2.0)
    second.start()
    time.sleep(0.05)
    release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert state == {"active": 0, "maximum": 1, "calls": 2}
    assert results == [({"KEY": "secret"}, []), ({"KEY": "secret"}, [])]


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_fetch_fails_closed_when_private_daemon_cleanup_fails(monkeypatch, tmp_path):
    """Resolved values must not escape when exact daemon cleanup cannot prove exit."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    runtime_dir = tmp_path / "managed-runtime"
    runtime_dir.mkdir(mode=0o700)
    namespace = mock.Mock(
        runtime_dir=runtime_dir,
        socket_path=runtime_dir / "op.sock",
    )
    monkeypatch.setattr(op, "_create_op_runtime_namespace", lambda: namespace)
    monkeypatch.setattr(op, "_run_op_process", lambda *a, **k: _ok("secret-value"))

    def cleanup_hold(ns, binary):
        raise RuntimeError("ambiguous 1Password daemon identity; cleanup HOLD")

    monkeypatch.setattr(op, "_cleanup_op_runtime_namespace", cleanup_hold)

    with pytest.raises(RuntimeError, match="cleanup HOLD"):
        op.fetch_onepassword_secrets(
            references={"K": "op://V/I/F"}, binary=fake_op, use_cache=False
        )


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_cleanup_signals_only_exact_singleton_after_pidfd_revalidation(
    monkeypatch, tmp_path
):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    pid_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(dir_fd=namespace_fd)
    identity = op._OpDaemonProcess(pid=4242, start_ticks=9001)
    sent = []

    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: 4242)
    scans = iter([([identity], []), ([], []), ([], []), ([], []), ([], [])])
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: next(scans)
    )
    monkeypatch.setattr(
        op, "_inspect_op_daemon", lambda pid, ns, binary_stat: ("exact", identity)
    )
    monkeypatch.setattr(op, "_pidfd_open", lambda pid: pid_fd)
    monkeypatch.setattr(op, "_pidfd_send_sigterm", lambda fd: sent.append(fd))
    monkeypatch.setattr(op, "_wait_pidfd_exit", lambda fd: True)
    monkeypatch.setattr(op, "_remove_op_runtime_namespace", lambda ns: None)

    op._cleanup_op_runtime_namespace(namespace, fake_op)

    assert sent == [pid_fd]
    with pytest.raises(OSError):
        os.fstat(pid_fd)
    with pytest.raises(OSError):
        os.fstat(namespace_fd)


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_cleanup_holds_late_respawn_after_pidfd_exit(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    pid_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(dir_fd=namespace_fd)
    original = op._OpDaemonProcess(pid=4301, start_ticks=100)
    respawn = op._OpDaemonProcess(pid=4302, start_ticks=101)
    scans = iter([([original], []), ([respawn], [])])
    sent = []
    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: original.pid)
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: next(scans)
    )
    monkeypatch.setattr(
        op, "_inspect_op_daemon", lambda pid, ns, binary_stat: ("exact", original)
    )
    monkeypatch.setattr(op, "_pidfd_open", lambda pid: pid_fd)
    monkeypatch.setattr(op, "_pidfd_send_sigterm", lambda fd: sent.append(fd))
    monkeypatch.setattr(op, "_wait_pidfd_exit", lambda fd: True)

    with pytest.raises(RuntimeError, match="not quiescent.*cleanup HOLD"):
        op._cleanup_op_runtime_namespace(namespace, fake_op)

    assert sent == [pid_fd]


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_cleanup_holds_late_foreign_process_without_signal(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(dir_fd=namespace_fd)
    scans = iter([([], []), ([], [4401])])
    sent = []
    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: None)
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: next(scans)
    )
    monkeypatch.setattr(op, "_pidfd_send_sigterm", lambda fd: sent.append(fd))

    with pytest.raises(RuntimeError, match="not quiescent.*cleanup HOLD"):
        op._cleanup_op_runtime_namespace(namespace, fake_op)

    assert sent == []


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_cleanup_identity_drift_after_pidfd_pin_never_signals(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    pid_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(dir_fd=namespace_fd)
    original = op._OpDaemonProcess(pid=4501, start_ticks=100)
    reused = op._OpDaemonProcess(pid=4501, start_ticks=101)
    sent = []
    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: original.pid)
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: ([original], [])
    )
    monkeypatch.setattr(
        op, "_inspect_op_daemon", lambda pid, ns, binary_stat: ("exact", reused)
    )
    monkeypatch.setattr(op, "_pidfd_open", lambda pid: pid_fd)
    monkeypatch.setattr(op, "_pidfd_send_sigterm", lambda fd: sent.append(fd))

    with pytest.raises(RuntimeError, match="identity changed.*cleanup HOLD"):
        op._cleanup_op_runtime_namespace(namespace, fake_op)

    assert sent == []


@pytest.mark.parametrize(
    "mismatch",
    [
        "cgroup",
        "executable_inode",
        "argv",
        "runtime_environment",
        "rehomed_environment",
        "rehomed_executable_swap",
        "unreadable_environment",
        "uid",
        "ppid",
        "start_time",
    ],
)
@pytest.mark.skipif(sys.platform != "linux", reason="proc identity is Linux-only")
def test_each_daemon_identity_mismatch_holds_without_signal(
    mismatch, monkeypatch, tmp_path
):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    binary_stat = fake_op.stat()
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    runtime_dir = tmp_path / "runtime"
    namespace = mock.Mock(
        dir_fd=namespace_fd,
        runtime_dir=runtime_dir,
        child_runtime_dir=runtime_dir,
        socket_path=runtime_dir / "op.sock",
        uid=os.getuid(),
        start_ticks_floor=100,
        cgroup=b"0::/expected\n",
    )
    pid = 4601
    env = {
        b"OP_SOCK": os.fsencode(namespace.socket_path),
        b"XDG_RUNTIME_DIR": os.fsencode(namespace.child_runtime_dir),
    }
    evidence_values = {
        "ppid": 1,
        "start_ticks": 100,
        "uid": namespace.uid,
        "executable_dev": binary_stat.st_dev,
        "executable_ino": binary_stat.st_ino,
        "cmdline": b"op\0daemon\0",
        "cgroup": namespace.cgroup,
    }
    if mismatch == "runtime_environment":
        env[b"XDG_RUNTIME_DIR"] = b"/foreign/runtime"
    elif mismatch == "rehomed_environment":
        env[b"OP_SOCK"] = b"/run/user/0/op.sock"
        env[b"XDG_RUNTIME_DIR"] = b"/run/user/0"
    elif mismatch == "rehomed_executable_swap":
        # Re-homed to the global runtime AND op upgraded mid-batch (exe
        # dev/ino changed). Every other identity axis still matches, so this
        # must surface as foreign, not silently drop as unrelated.
        env[b"OP_SOCK"] = b"/run/user/0/op.sock"
        env[b"XDG_RUNTIME_DIR"] = b"/run/user/0"
        evidence_values["executable_ino"] = binary_stat.st_ino + 1
    elif mismatch == "cgroup":
        evidence_values["cgroup"] = b"0::/foreign\n"
    elif mismatch == "executable_inode":
        evidence_values["executable_ino"] = binary_stat.st_ino + 1
    elif mismatch == "argv":
        evidence_values["cmdline"] = b"op\0read\0"
    elif mismatch == "uid":
        evidence_values["uid"] = namespace.uid + 1
    elif mismatch == "ppid":
        evidence_values["ppid"] = 2
    elif mismatch == "start_time":
        evidence_values["start_ticks"] = 99
    evidence = op._OpDaemonEvidence(**evidence_values)
    if mismatch == "unreadable_environment":
        monkeypatch.setattr(
            op,
            "_read_proc_environment",
            lambda candidate: (_ for _ in ()).throw(PermissionError("proc race")),
        )
    else:
        monkeypatch.setattr(op, "_read_proc_environment", lambda candidate: env)
    monkeypatch.setattr(op, "_read_op_daemon_evidence", lambda candidate: evidence)

    status, identity = op._inspect_op_daemon(pid, namespace, binary_stat)
    assert status in {"foreign", "unrelated"}
    if mismatch in {"rehomed_environment", "rehomed_executable_swap"}:
        assert status == "foreign"
    assert identity is None

    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: pid)
    monkeypatch.setattr(
        op,
        "_scan_op_runtime_namespace",
        lambda ns, bs: ([], [pid]) if status == "foreign" else ([], []),
    )
    signaled = []
    monkeypatch.setattr(op, "_pidfd_send_sigterm", lambda fd: signaled.append(fd))
    with pytest.raises(RuntimeError, match="cleanup HOLD"):
        op._cleanup_op_runtime_namespace(namespace, fake_op)
    assert signaled == []


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_cleanup_holds_foreign_namespace_process_without_signal(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(dir_fd=namespace_fd)
    signaled = []

    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: None)
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: ([], [7331])
    )
    monkeypatch.setattr(op, "_pidfd_send_sigterm", lambda fd: signaled.append(fd))

    with pytest.raises(RuntimeError, match="ambiguous.*cleanup HOLD"):
        op._cleanup_op_runtime_namespace(namespace, fake_op)

    assert signaled == []
    with pytest.raises(OSError):
        os.fstat(namespace_fd)


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_cleanup_holds_multiple_exact_daemons_without_signal(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(dir_fd=namespace_fd)
    identities = [
        op._OpDaemonProcess(pid=8101, start_ticks=10),
        op._OpDaemonProcess(pid=8102, start_ticks=11),
    ]
    signaled = []
    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: 8101)
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: (identities, [])
    )
    monkeypatch.setattr(op, "_pidfd_send_sigterm", lambda fd: signaled.append(fd))

    with pytest.raises(RuntimeError, match="ambiguous.*cleanup HOLD"):
        op._cleanup_op_runtime_namespace(namespace, fake_op)

    assert signaled == []


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd child lifecycle is Linux-only"
)
def test_op_child_timeout_uses_pidfd_sigterm_and_never_sigkill(monkeypatch):
    class StubbornChild:
        pid = 91234
        returncode = None
        stdout = mock.Mock()
        stderr = mock.Mock()

        def __init__(self):
            self.kill_calls = 0
            self.terminate_calls = 0

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(["op", "read"], timeout)

        def kill(self):
            self.kill_calls += 1

        def terminate(self):
            self.terminate_calls += 1

    child = StubbornChild()
    sent = []
    closed = []
    monkeypatch.setattr(op.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(op, "_pidfd_open", lambda pid: 77)
    monkeypatch.setattr(op, "_pidfd_send_sigterm", lambda fd: sent.append(fd))
    monkeypatch.setattr(op, "_wait_pidfd_exit", lambda fd: False)
    monkeypatch.setattr(op.os, "close", lambda fd: closed.append(fd))

    with pytest.raises(RuntimeError, match="timed out.*SIGTERM.*still running"):
        op._run_op_process(["/usr/bin/op", "read"], env={})

    assert sent == [77]
    assert closed[-1] == 77
    assert len(closed) == 2  # inode-bound helper FD, then child pidfd
    assert child.kill_calls == 0
    assert child.terminate_calls == 0


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_cleanup_timeout_never_escalates_to_sigkill(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    pid_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(dir_fd=namespace_fd)
    identity = op._OpDaemonProcess(pid=8201, start_ticks=12)
    sent = []
    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: identity.pid)
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: ([identity], [])
    )
    monkeypatch.setattr(
        op, "_inspect_op_daemon", lambda pid, ns, binary_stat: ("exact", identity)
    )
    monkeypatch.setattr(op, "_pidfd_open", lambda pid: pid_fd)
    monkeypatch.setattr(
        op, "_pidfd_send_sigterm", lambda fd: sent.append(signal.SIGTERM)
    )
    monkeypatch.setattr(op, "_wait_pidfd_exit", lambda fd: False)

    with pytest.raises(RuntimeError, match="did not exit.*cleanup HOLD"):
        op._cleanup_op_runtime_namespace(namespace, fake_op)

    assert sent == [signal.SIGTERM]


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_runtime_removal_requires_bound_inode_unlink_proof(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "bound-runtime"
    runtime_dir.mkdir(mode=0o700)
    dir_fd = os.open(runtime_dir, os.O_RDONLY | os.O_DIRECTORY)
    directory_stat = os.fstat(dir_fd)
    namespace = op._OpRuntimeNamespace(
        runtime_dir=runtime_dir,
        child_runtime_dir=runtime_dir,
        socket_path=runtime_dir / "op.sock",
        dir_fd=dir_fd,
        dir_dev=directory_stat.st_dev,
        dir_ino=directory_stat.st_ino,
        uid=os.getuid(),
        start_ticks_floor=0,
        cgroup=b"test",
    )
    original_rmdir = os.rmdir
    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(op.os, "rmdir", lambda path: None)
        with pytest.raises(RuntimeError, match="inode unlink proof.*cleanup HOLD"):
            op._remove_op_runtime_namespace(namespace)

    os.close(dir_fd)
    original_rmdir(runtime_dir)


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_cleanup_normalizes_filesystem_error_to_explicit_hold(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(dir_fd=namespace_fd)

    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: None)
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: ([], [])
    )
    monkeypatch.setattr(
        op,
        "_remove_op_runtime_namespace",
        lambda ns: (_ for _ in ()).throw(OSError("path race")),
    )

    with pytest.raises(RuntimeError, match="cleanup HOLD"):
        op._cleanup_op_runtime_namespace(namespace, fake_op)

    with pytest.raises(OSError):
        os.fstat(namespace_fd)


def test_close_failure_does_not_overwrite_primary_cleanup_hold(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(dir_fd=namespace_fd)
    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: None)
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: ([], [])
    )
    monkeypatch.setattr(op.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        op,
        "_remove_op_runtime_namespace",
        lambda ns: (_ for _ in ()).throw(RuntimeError("primary cleanup HOLD")),
    )
    monkeypatch.setattr(
        op,
        "_close_op_runtime_namespace_fd",
        lambda fd: (_ for _ in ()).throw(OSError("secondary close failure")),
    )
    try:
        with pytest.raises(RuntimeError, match="primary cleanup HOLD") as error:
            op._cleanup_op_runtime_namespace(namespace, fake_op)
    finally:
        os.close(namespace_fd)

    assert "secondary close failure" in " ".join(getattr(error.value, "__notes__", []))


def _install_namespace_close_failure(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace_fd = os.open("/dev/null", os.O_RDONLY)
    namespace = mock.Mock(
        dir_fd=namespace_fd,
        runtime_dir=tmp_path / "runtime",
        socket_path=tmp_path / "runtime" / "op.sock",
    )
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setattr(op, "_create_op_runtime_namespace", lambda: namespace)
    monkeypatch.setattr(op, "_run_op_process", lambda *args, **kwargs: _ok("secret"))
    monkeypatch.setattr(op, "_validate_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op, "_read_op_daemon_pidfile", lambda ns: None)
    monkeypatch.setattr(
        op, "_scan_op_runtime_namespace", lambda ns, binary_stat: ([], [])
    )
    monkeypatch.setattr(op, "_remove_op_runtime_namespace", lambda ns: None)
    monkeypatch.setattr(op.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        op,
        "_close_op_runtime_namespace_fd",
        lambda fd: (_ for _ in ()).throw(OSError("close race")),
    )
    return fake_op, namespace_fd


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_namespace_fd_close_failure_normalizes_to_hold(monkeypatch, tmp_path):
    fake_op, namespace_fd = _install_namespace_close_failure(monkeypatch, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="namespace close failed.*cleanup HOLD"):
            op.fetch_onepassword_secrets(
                references={"K": "op://V/I/F"},
                binary=fake_op,
                use_cache=False,
            )
    finally:
        os.close(namespace_fd)


def test_apply_namespace_fd_close_failure_returns_error(monkeypatch, tmp_path):
    _, namespace_fd = _install_namespace_close_failure(monkeypatch, tmp_path)
    monkeypatch.delenv("CLOSE_ERROR_KEY", raising=False)
    try:
        result = op.apply_onepassword_secrets(
            enabled=True,
            env={"CLOSE_ERROR_KEY": "op://V/I/F"},
            cache_ttl_seconds=0,
        )
    finally:
        os.close(namespace_fd)

    assert not result.ok
    assert result.error is not None
    assert "namespace close failed" in result.error
    assert "CLOSE_ERROR_KEY" not in os.environ


def test_source_namespace_fd_close_failure_returns_error(monkeypatch, tmp_path):
    _, namespace_fd = _install_namespace_close_failure(monkeypatch, tmp_path)
    try:
        result = op.OnePasswordSource().fetch(
            {
                "env": {"CLOSE_ERROR_KEY": "op://V/I/F"},
                "cache_ttl_seconds": 0,
            },
            tmp_path,
        )
    finally:
        os.close(namespace_fd)

    assert not result.ok
    assert result.error is not None
    assert "namespace close failed" in result.error
    assert result.secrets == {}


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_direct_run_op_read_manages_namespace_when_env_is_omitted(
    monkeypatch, tmp_path
):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace = mock.Mock(
        runtime_dir=tmp_path / "runtime",
        socket_path=tmp_path / "runtime" / "op.sock",
    )
    cleaned = []
    captured = {}
    monkeypatch.setattr(op, "_create_op_runtime_namespace", lambda: namespace)
    monkeypatch.setattr(
        op,
        "_cleanup_op_runtime_namespace",
        lambda ns, binary: cleaned.append((ns, binary)),
    )

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _ok("secret\n")

    monkeypatch.setattr(op, "_run_op_process", fake_run)

    assert op._run_op_read(fake_op, "op://V/I/F") == "secret"
    assert captured["env"]["XDG_RUNTIME_DIR"] == str(namespace.runtime_dir)
    assert cleaned == [(namespace, fake_op)]


@pytest.mark.skipif(
    sys.platform != "linux", reason="pidfd daemon cleanup is Linux-only"
)
def test_direct_run_op_read_cleanup_failure_discards_value(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace = mock.Mock(
        runtime_dir=tmp_path / "runtime",
        socket_path=tmp_path / "runtime" / "op.sock",
    )
    monkeypatch.setattr(op, "_create_op_runtime_namespace", lambda: namespace)
    monkeypatch.setattr(op, "_run_op_process", lambda *args, **kwargs: _ok("secret\n"))
    monkeypatch.setattr(
        op,
        "_cleanup_op_runtime_namespace",
        lambda ns, binary: (_ for _ in ()).throw(RuntimeError("cleanup HOLD")),
    )

    with pytest.raises(RuntimeError, match="cleanup HOLD"):
        op._run_op_read(fake_op, "op://V/I/F")


def test_apply_cleanup_failure_never_mutates_environment(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    namespace = mock.Mock(
        runtime_dir=tmp_path / "runtime",
        socket_path=tmp_path / "runtime" / "op.sock",
    )
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setattr(op, "_create_op_runtime_namespace", lambda: namespace)
    monkeypatch.setattr(op, "_run_op_process", lambda *args, **kwargs: _ok("secret\n"))
    monkeypatch.setattr(
        op,
        "_cleanup_op_runtime_namespace",
        lambda ns, binary: (_ for _ in ()).throw(RuntimeError("cleanup HOLD")),
    )
    monkeypatch.delenv("MY_OP_KEY", raising=False)

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"MY_OP_KEY": "op://V/I/F"},
        cache_ttl_seconds=0,
    )

    assert not result.ok
    assert result.error is not None
    assert "cleanup HOLD" in result.error
    assert "MY_OP_KEY" not in os.environ


def test_apply_namespace_creation_oserror_becomes_error_result(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setattr(
        op,
        "_create_op_runtime_namespace_inner",
        lambda: (_ for _ in ()).throw(OSError("read-only runtime")),
    )
    monkeypatch.delenv("CREATE_ERROR_KEY", raising=False)

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"CREATE_ERROR_KEY": "op://V/I/F"},
        cache_ttl_seconds=0,
        home_path=tmp_path,
    )

    assert not result.ok
    assert result.error is not None
    assert "runtime setup" in result.error
    assert "HOLD" in result.error
    assert "CREATE_ERROR_KEY" not in os.environ


def test_source_fetch_namespace_creation_oserror_becomes_error_result(
    monkeypatch, tmp_path
):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setattr(
        op,
        "_create_op_runtime_namespace_inner",
        lambda: (_ for _ in ()).throw(OSError("read-only runtime")),
    )

    result = op.OnePasswordSource().fetch(
        {
            "env": {"CREATE_ERROR_KEY": "op://V/I/F"},
            "cache_ttl_seconds": 0,
        },
        tmp_path,
    )

    assert not result.ok
    assert result.error is not None
    assert "runtime setup" in result.error
    assert "HOLD" in result.error
    assert result.secrets == {}


def test_fetch_happy_path(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    values = {
        "op://Private/OpenAI/api key": "sk-abc\n",
        "op://Private/Anthropic/credential": "sk-ant-xyz",
    }

    def fake_run(cmd, **kwargs):
        # argv list, never shell=True; reference passed after `--`.
        assert "--" in cmd
        ref = cmd[cmd.index("--") + 1]
        return _ok(values[ref])

    monkeypatch.setattr(op, "_run_op_process", fake_run)

    secrets, warnings = op.fetch_onepassword_secrets(
        references={
            "OPENAI_API_KEY": "op://Private/OpenAI/api key",
            "ANTHROPIC_API_KEY": "op://Private/Anthropic/credential",
        },
        binary=fake_op,
        use_cache=False,
    )
    assert secrets == {"OPENAI_API_KEY": "sk-abc", "ANTHROPIC_API_KEY": "sk-ant-xyz"}
    assert warnings == []


def test_fetch_read_failure_becomes_warning(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(
        op,
        "_run_op_process",
        lambda *a, **k: _err(1, "\x1b[31m[ERROR] not signed in\x1b[0m"),
    )

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, binary=fake_op, use_cache=False
    )
    assert secrets == {}
    assert len(warnings) == 1
    # ANSI control sequences are fully scrubbed from the surfaced message.
    assert "\x1b" not in warnings[0]
    assert "[31m" not in warnings[0]
    assert "not signed in" in warnings[0]


def test_partial_reference_failure_is_not_cached_and_retries(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"op://V/I/A": 0, "op://V/I/B": 0}

    def fake_run(cmd, **kwargs):
        ref = cmd[cmd.index("--") + 1]
        calls[ref] += 1
        if ref == "op://V/I/B" and calls[ref] == 1:
            return _err(1, "temporary auth failure")
        return _ok(ref.rsplit("/", 1)[-1].lower())

    monkeypatch.setattr(op, "_run_op_process", fake_run)
    refs = {"A": "op://V/I/A", "B": "op://V/I/B"}

    first, first_warnings = op.fetch_onepassword_secrets(
        references=refs, binary=fake_op, home_path=tmp_path
    )
    second, second_warnings = op.fetch_onepassword_secrets(
        references=refs, binary=fake_op, home_path=tmp_path
    )

    assert first == {"A": "a"}
    assert first_warnings
    assert second == {"A": "a", "B": "b"}
    assert second_warnings == []
    assert calls == {"op://V/I/A": 2, "op://V/I/B": 2}


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_inprocess_cache_hit(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(op, "_run_op_process", fake_run)
    op._reset_cache_for_tests(tmp_path)
    for _ in range(2):
        op.fetch_onepassword_secrets(
            references={"K": "op://V/I/F"},
            cache_ttl_seconds=60,
            binary=fake_op,
            home_path=tmp_path,
        )
    assert calls["n"] == 1  # second call served from L1 cache


def test_disk_cache_hit_promotes_l1_without_binary_or_runtime(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return _ok("disk-value")

    monkeypatch.setattr(op, "_run_op_process", fake_run)
    first, _ = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"},
        cache_ttl_seconds=60,
        binary=fake_op,
        home_path=tmp_path,
    )
    assert first == {"K": "disk-value"}
    op._CACHE.clear()
    monkeypatch.setattr(
        op, "find_op", lambda *args, **kwargs: pytest.fail("disk hit must skip find_op")
    )
    monkeypatch.setattr(
        op,
        "_create_op_runtime_namespace_inner",
        lambda: pytest.fail("disk hit must skip runtime creation"),
    )

    second, _ = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"},
        cache_ttl_seconds=60,
        home_path=tmp_path,
    )

    assert second == {"K": "disk-value"}
    assert calls["n"] == 1
    assert op._CACHE


def test_connect_credential_change_invalidates_cache(monkeypatch, tmp_path):
    """A different 1Password Connect identity must not reuse a cached value."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(op, "_run_op_process", fake_run)
    op._reset_cache_for_tests(tmp_path)

    monkeypatch.setenv("OP_CONNECT_HOST", "https://connect.example.com")
    monkeypatch.setenv("OP_CONNECT_TOKEN", "tokenA")
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"},
        cache_ttl_seconds=300,
        binary=fake_op,
        home_path=tmp_path,
    )
    # Rotate the Connect token → new identity.
    monkeypatch.setenv("OP_CONNECT_TOKEN", "tokenB")
    op._CACHE.clear()
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"},
        cache_ttl_seconds=300,
        binary=fake_op,
        home_path=tmp_path,
    )
    assert calls["n"] == 2  # cache key changed → refetch


# ---------------------------------------------------------------------------
# find_op
# ---------------------------------------------------------------------------


def test_find_op_pinned_path_not_on_path(tmp_path, monkeypatch):
    pinned = tmp_path / "op"
    pinned.write_text("")
    pinned.chmod(0o755)
    # PATH lookup must NOT be consulted when a binary_path is pinned.
    monkeypatch.setattr(op.shutil, "which", lambda name: "/usr/bin/op")
    assert op.find_op(str(pinned)) == pinned


# ---------------------------------------------------------------------------
# apply_onepassword_secrets
# ---------------------------------------------------------------------------


def test_apply_disabled_returns_empty():
    result = op.apply_onepassword_secrets(enabled=False, env={"K": "op://V/I/F"})
    assert result.ok
    assert not result.applied


def test_apply_missing_binary_sets_error(monkeypatch):
    monkeypatch.setattr(op, "find_op", lambda binary_path="": None)
    result = op.apply_onepassword_secrets(enabled=True, env={"K": "op://V/I/F"})
    assert not result.ok
    assert "op CLI" in result.error


def test_apply_sets_env(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setattr(op, "_run_op_process", lambda *a, **k: _ok("resolved-val"))
    monkeypatch.delenv("MY_OP_KEY", raising=False)

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"MY_OP_KEY": "op://V/I/F"},
        cache_ttl_seconds=0,
    )
    assert result.ok
    assert result.applied == ["MY_OP_KEY"]
    assert os.environ["MY_OP_KEY"] == "resolved-val"


def test_apply_skips_before_fetch_when_not_overriding(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setenv("MY_OP_KEY", "from-env")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("from-1password")

    monkeypatch.setattr(op, "_run_op_process", fake_run)

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"MY_OP_KEY": "op://V/I/F"},
        override_existing=False,
        cache_ttl_seconds=0,
    )
    assert "MY_OP_KEY" in result.skipped
    assert os.environ["MY_OP_KEY"] == "from-env"
    assert calls["n"] == 0  # never even called op for a value we'd discard


def test_apply_never_overrides_token_var(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "original")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("malicious")

    monkeypatch.setattr(op, "_run_op_process", fake_run)

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"OP_SERVICE_ACCOUNT_TOKEN": "op://V/I/F"},
        override_existing=True,
        cache_ttl_seconds=0,
    )
    assert "OP_SERVICE_ACCOUNT_TOKEN" in result.skipped
    assert os.environ["OP_SERVICE_ACCOUNT_TOKEN"] == "original"
    assert calls["n"] == 0


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper helper is Linux-only")
def test_subreaper_helper_sigterms_exact_adopted_daemon(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    binary_stat = fake_op.stat()
    expected_cgroup = Path("/proc/self/cgroup").read_bytes()
    pid = 4701
    identity = {
        "pid": pid,
        "ppid": os.getpid(),
        "start_ticks": 100,
        "uid": os.getuid(),
        "cmdline": op_helper.OP_DAEMON_CMDLINE,
        "cgroup": expected_cgroup,
        "exe": (binary_stat.st_dev, binary_stat.st_ino),
    }
    scans = iter([[pid], [], [], [], [], []])
    pid_fd = os.open("/dev/null", os.O_RDONLY)
    sent = []
    monkeypatch.setattr(op_helper, "_reap_exited_children", lambda: None)
    monkeypatch.setattr(op_helper, "_child_pids", lambda: next(scans))
    monkeypatch.setattr(op_helper, "_proc_identity", lambda candidate: identity)
    monkeypatch.setattr(op_helper, "_pidfd_open", lambda candidate: pid_fd)
    monkeypatch.setattr(
        op_helper, "_pidfd_send_sigterm", lambda descriptor: sent.append(descriptor)
    )
    monkeypatch.setattr(op_helper, "_wait_pidfd_exit", lambda descriptor: True)
    monkeypatch.setattr(op_helper.time, "sleep", lambda seconds: None)

    op_helper._cleanup_adopted_children(fake_op)

    assert sent == [pid_fd]


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper helper is Linux-only")
def test_subreaper_helper_holds_unknown_adopted_child_without_signal(
    monkeypatch, tmp_path
):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    pid = 4702
    unknown = {
        "pid": pid,
        "ppid": os.getpid(),
        "start_ticks": 100,
        "uid": os.getuid(),
        "cmdline": b"op\0read\0",
        "cgroup": Path("/proc/self/cgroup").read_bytes(),
        "exe": (fake_op.stat().st_dev, fake_op.stat().st_ino),
    }
    sent = []
    monkeypatch.setattr(op_helper, "ADOPTION_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(op_helper, "_reap_exited_children", lambda: None)
    monkeypatch.setattr(op_helper, "_child_pids", lambda: [pid])
    monkeypatch.setattr(op_helper, "_proc_identity", lambda candidate: unknown)
    monkeypatch.setattr(
        op_helper, "_pidfd_send_sigterm", lambda descriptor: sent.append(descriptor)
    )

    with pytest.raises(RuntimeError, match="unknown adopted op child; helper HOLD"):
        op_helper._cleanup_adopted_children(fake_op)

    assert sent == []


@pytest.mark.skipif(sys.platform != "linux", reason="child subreaper is Linux-only")
def test_set_child_subreaper_adopts_double_forked_grandchild():
    """Un-mocked integration check of the core mechanism.

    After ``_set_child_subreaper()`` a double-forked grandchild must reparent
    to THIS process (an attestable, reapable descendant) instead of escaping
    to init (pid 1) the way the un-hardened ``op`` daemon does. Exercises the
    real ``prctl(PR_SET_CHILD_SUBREAPER)`` syscall and a genuine double fork,
    which the mocked helper tests never run.
    """
    import select

    report_r, report_w = os.pipe()
    subreaper_pid = os.fork()  # windows-footgun: ok
    if subreaper_pid == 0:  # ---- subreaper process ----
        os.close(report_r)
        rc = 0
        try:
            op_helper._set_child_subreaper()
            intermediate = os.fork()  # windows-footgun: ok
            if intermediate == 0:  # ---- intermediate ----
                intermediate_pid = os.getpid()
                if os.fork() == 0:  # windows-footgun: ok
                    deadline = time.monotonic() + 5.0
                    while (
                        os.getppid() == intermediate_pid
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.01)
                    os.write(report_w, str(os.getppid()).encode())
                    os._exit(0)
                os._exit(0)  # intermediate exits -> grandchild reparents upward
            os.waitpid(intermediate, 0)
            try:
                while True:
                    os.waitpid(-1, 0)  # reap the adopted grandchild
            except ChildProcessError:
                pass
        except BaseException:
            rc = 1
        os._exit(rc)

    # ---- parent (test process) ----
    os.close(report_w)
    try:
        ready, _, _ = select.select([report_r], [], [], 10.0)
        assert ready, "grandchild never reported its reparented ppid"
        reported = os.read(report_r, 64).decode()
    finally:
        os.close(report_r)
        os.waitpid(subreaper_pid, 0)

    assert reported == str(subreaper_pid), (
        f"grandchild reparented to ppid {reported!r}; expected the subreaper "
        f"{subreaper_pid}, not init (pid 1)"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX group/owner semantics")
def test_writable_by_others_tolerates_private_group_but_rejects_shared(monkeypatch):
    # Regression for a umask-002 checkout: a bundled helper left mode 0664 with
    # a private per-user group (no secondary members) is NOT writable by others
    # and must be accepted — the pre-fix `& 0o022` check rejected it and HELD
    # every Linux op fetch. World- and shared-group-writable stay unsafe.
    import grp
    import pwd

    def st(mode, uid=4242, gid=4242):
        return mock.Mock(st_mode=mode, st_uid=uid, st_gid=gid)

    def set_group(name, members):
        monkeypatch.setattr(
            grp, "getgrgid", lambda gid: mock.Mock(gr_name=name, gr_mem=members)
        )

    monkeypatch.setattr(
        pwd, "getpwuid", lambda uid: mock.Mock(pw_name="alice", pw_gid=4242)
    )
    monkeypatch.setattr(os, "listxattr", lambda fd: [])  # no extended ACL by default

    def w(mode, **kw):
        return op._writable_by_others(st(mode, **kw), fd=7)

    # Owner's own private per-user group: gid == owner's login gid (4242),
    # named after the owner, no secondary members, no ACL -> safe.
    set_group("alice", [])
    assert w(0o100644) is False
    assert w(0o100664) is False

    # World-writable is always unsafe.
    assert w(0o100666) is True
    assert w(0o100646) is True

    # An extended POSIX ACL (the group-write bit may be a mask hiding a
    # u:other:w grant) is unsafe even for the owner's own private group.
    monkeypatch.setattr(os, "listxattr", lambda fd: ["system.posix_acl_access"])
    assert w(0o100664) is True
    monkeypatch.setattr(os, "listxattr", lambda fd: [])

    # Group-writable but the file's group is NOT the owner's primary gid — some
    # other account could hold gid 9999 as its primary group -> unsafe.
    assert w(0o100664, gid=9999) is True

    # Group-writable via a shared group not named after the owner -> unsafe.
    set_group("developers", [])
    assert w(0o100664) is True

    # Owner-named primary group with a secondary member -> unsafe.
    set_group("alice", ["bob"])
    assert w(0o100664) is True

    # Owner's own private group listing only the owner -> safe.
    set_group("alice", ["alice"])
    assert w(0o100664) is False

    # Unresolvable identity fails closed (unsafe).
    def _boom(_gid):
        raise KeyError(_gid)

    monkeypatch.setattr(grp, "getgrgid", _boom)
    assert w(0o100664) is True
