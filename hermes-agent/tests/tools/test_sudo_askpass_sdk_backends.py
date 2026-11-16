"""Askpass sudo delivery on the SDK backends (direct Modal, Managed Modal, Daytona, Vercel).

These backends run no capability probe. They select the askpass form directly and the prologue
checks sudo ``-A`` support and its helper programs inside the command itself, falling back to the
stdin form there. Each backend is driven through its real ``execute`` with only the SDK / HTTP
layer replaced by a local runner: real bash runs the exact command string and stdin the backend
would send, with a stand-in sudo first on PATH.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path

import pytest

from tests.tools.test_sudo_askpass_delivery import PW, _write_fake_sudo
from tests.waiting import wait_until
from tools import terminal_tool, terminal_tool_sudo
from tools import terminal_tool_sudo_askpass as askpass
from tools.environments.base import BaseEnvironment

pytestmark = [
    pytest.mark.platforms("posix"),
    pytest.mark.skipif(sys.platform == "win32" or not all(shutil.which(t) for t in ("bash", "mkfifo", "mktemp")),
                       reason="needs a POSIX bash with mkfifo and mktemp"),
]

BACKENDS = ("modal", "managed_modal", "daytona", "vercel")


def setup_function():
    terminal_tool_sudo._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool_sudo._reset_cached_sudo_passwords()


class _Runner:
    """Runs a command with local bash in a fixed environment: fake sudo first on PATH, own TMPDIR."""

    def __init__(self, tmp_path: Path, bin_dir: Path):
        self.temp_dir = tmp_path / "tmp"
        self.temp_dir.mkdir(exist_ok=True)
        self.cwd = str(tmp_path)
        self.env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "TMPDIR": str(self.temp_dir), "HOME": str(tmp_path)}
        self.commands: list[str] = []

    def run(self, argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
        self.commands.append(argv[-1])
        return subprocess.run(argv, input=stdin, capture_output=True, cwd=self.cwd, env=self.env,
                              timeout=30, check=False,
                              stdin=None if stdin is not None else subprocess.DEVNULL)


def _aio(fn):
    async def _call(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)
    return types.SimpleNamespace(aio=_call)


class _FakeModalProcess:
    """Modal ``ContainerProcess``: stdin is collected until EOF, then the argv runs."""

    def __init__(self, runner, argv):
        self._runner, self._argv, self._stdin, self._done = runner, argv, bytearray(), None
        self.stdin = types.SimpleNamespace(write=self._stdin.extend, write_eof=lambda: None, drain=_aio(lambda: None))
        self.stdout = types.SimpleNamespace(read=_aio(lambda: self._result().stdout))
        self.stderr = types.SimpleNamespace(read=_aio(lambda: self._result().stderr))
        self.wait = _aio(lambda: self._result().returncode)

    def _result(self):
        if self._done is None:
            self._done = self._runner.run(self._argv, bytes(self._stdin))
        return self._done


def _bare(cls, runner):
    """An instance of *cls* without its SDK constructor: base state only, sandbox temp dir local."""
    env = cls.__new__(cls)
    env.get_temp_dir = lambda: str(runner.temp_dir)
    BaseEnvironment.__init__(env, cwd=runner.cwd, timeout=10)
    env._prefer_nonlogin = True  # no snapshot: plain ``bash -c`` per command
    env._before_execute = lambda: None  # sandbox refresh and file sync talk to the SDK
    return env


def _modal_env(runner, _monkeypatch):
    from tools.environments.modal import ModalEnvironment, _AsyncWorker

    env = _bare(ModalEnvironment, runner)
    env._worker = _AsyncWorker()
    env._worker.start()
    env._sandbox = types.SimpleNamespace(
        exec=_aio(lambda *argv, timeout=None: _FakeModalProcess(runner, list(argv))),
        terminate=_aio(lambda: None))
    return env, env._worker.stop


def _daytona_env(runner, _monkeypatch):
    from tools.environments.daytona import DaytonaEnvironment

    def upload(content, path):
        Path(path).write_bytes(content)

    def process_exec(shell_cmd, timeout=None):
        done = runner.run(["sh", "-c", shell_cmd], None)
        return types.SimpleNamespace(result=(done.stdout + done.stderr).decode(), exit_code=done.returncode)

    env = _bare(DaytonaEnvironment, runner)
    env._lock = threading.Lock()
    env._sandbox = types.SimpleNamespace(
        fs=types.SimpleNamespace(upload_file=upload, delete_file=lambda path: Path(path).unlink(),
                                 set_file_permissions=lambda path, mode: os.chmod(path, int(mode, 8))),
        process=types.SimpleNamespace(exec=process_exec), stop=lambda: None)
    return env, None


def _vercel_env(runner, _monkeypatch):
    from tools.environments.vercel_sandbox import VercelSandboxEnvironment

    def write_files(files):
        for item in files:
            Path(item["path"]).write_bytes(item["content"])
            os.chmod(item["path"], item.get("mode", 0o644))

    def run_command(program, args, cwd=None):
        done = runner.run([program, *args], None)
        return types.SimpleNamespace(output=lambda: (done.stdout + done.stderr).decode(), exit_code=done.returncode)

    env = _bare(VercelSandboxEnvironment, runner)
    env._lock = threading.Lock()
    env._workspace_root = runner.cwd
    env._sandbox = types.SimpleNamespace(write_files=write_files, run_command=run_command)
    return env, None


class _GatewayResponse:
    def __init__(self, payload):
        self.status_code, self._payload, self.text = 200, payload, ""

    def json(self):
        return self._payload


def _managed_modal_env(runner, monkeypatch):
    from tools import credential_files
    from tools.environments import managed_modal

    monkeypatch.setattr(credential_files, "get_credential_file_mounts", list)
    monkeypatch.setattr(managed_modal, "resolve_managed_tool_gateway", lambda vendor: types.SimpleNamespace(
        vendor=vendor, gateway_origin="https://modal-gateway.example.com", nous_user_token="user-token",
        managed_mode=True))

    def fake_request(method, url, headers=None, json=None, timeout=None):
        if url.endswith("/v1/sandboxes"):
            return _GatewayResponse({"id": "sandbox-1"})
        if url.endswith("/execs"):
            stdin = json["stdinData"].encode() if "stdinData" in json else None
            done = runner.run(["bash", "-c", json["command"]], stdin)
            return _GatewayResponse({"execId": json["execId"], "status": "completed",
                                     "output": (done.stdout + done.stderr).decode(), "returncode": done.returncode})
        if url.endswith("/terminate"):
            return _GatewayResponse({"status": "terminated"})
        raise AssertionError(f"unexpected request {method} {url}")

    monkeypatch.setattr(managed_modal.requests, "request", fake_request)
    env = managed_modal.ManagedModalEnvironment(image="python:3.11", cwd=runner.cwd, timeout=10)
    return env, env.cleanup


_FACTORIES = {"modal": _modal_env, "managed_modal": _managed_modal_env, "daytona": _daytona_env,
              "vercel": _vercel_env}


@pytest.fixture
def make_backend(tmp_path, monkeypatch):
    """``make_backend(name, accepts_askpass=True, password=True)`` -> (env, runner)."""
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_sudo_password_callback(None)
    stops = []

    def _make(name, accepts_askpass=True, password=True):
        bin_dir = tmp_path / "bin"
        _write_fake_sudo(bin_dir, accepts_askpass=accepts_askpass)
        runner = _Runner(tmp_path, bin_dir)
        env, stop = _FACTORIES[name](runner, monkeypatch)
        if stop is not None:
            stops.append(stop)
        if password:
            terminal_tool_sudo._set_cached_sudo_password(PW)
        return env, runner

    yield _make
    for stop in stops:
        stop()


def _probe_commands(runner):
    return [c for c in runner.commands if c in (askpass.CAPABILITY_PROBE, "sudo -n true")]


# --- the spare password line on the SDK backends -------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_spare_password_line_never_reaches_a_later_reader(make_backend, backend):
    env, runner = make_backend(backend)
    result = env.execute("sudo true && sudo true; cat", stdin_data="rest\n")
    assert result["returncode"] == 0, result
    assert PW not in result["output"]
    assert result["output"].strip() == "rest"
    assert _probe_commands(runner) == []
    _wait_gone(runner.temp_dir)


def _wait_gone(temp_dir):
    wait_until(lambda: not list(temp_dir.glob("hermes-sudo.*")), timeout=5, what="askpass dir removed")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_sudo_rejecting_askpass_falls_back_to_stdin_form_in_the_same_command(make_backend, backend):
    env, runner = make_backend(backend, accepts_askpass=False)
    result = env.execute("sudo id -u && echo authed; cat", stdin_data="rest\n")
    assert result["returncode"] == 0, result
    assert result["output"].split() == [str(os.getuid()), "authed", "rest"]
    assert _probe_commands(runner) == []
    assert len(runner.commands) == 1  # one command: the check and the fallback run inside it
    _wait_gone(runner.temp_dir)


_REFUSING_SUDO_FUNCTION = 'sudo() { if [ "$1" = -A ]; then echo "wrapper refuses -A" >&2; return 42; fi; command sudo "$@"; }\n'


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("command", ["sudo id -u && echo authed; cat", '"sudo" id -u && echo authed; cat'])
def test_sdk_backend_sudo_function_rejecting_askpass_falls_back_to_stdin_form(make_backend, tmp_path, backend,
                                                                              command):
    """A snapshot function shadows the binary: the check runs the function, as the command does."""
    env, runner = make_backend(backend)
    (tmp_path / "functions.sh").write_text(_REFUSING_SUDO_FUNCTION, encoding="utf-8")
    runner.env["BASH_ENV"] = str(tmp_path / "functions.sh")
    result = env.execute(command, stdin_data="rest\n")
    _wait_gone(runner.temp_dir)
    assert result["returncode"] == 0, result
    assert result["output"].split() == [str(os.getuid()), "authed", "rest"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_sudo_wrapper_accepting_askpass_keeps_askpass_form(make_backend, tmp_path, backend):
    env, runner = make_backend(backend)
    (tmp_path / "functions.sh").write_text('sudo() { command sudo "$@"; }\n', encoding="utf-8")
    runner.env["BASH_ENV"] = str(tmp_path / "functions.sh")
    result = env.execute("sudo true && sudo true; cat", stdin_data="rest\n")
    _wait_gone(runner.temp_dir)
    assert result["returncode"] == 0, result
    assert result["output"].strip() == "rest"


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_sudo_wrapper_that_exits_cannot_end_the_command(make_backend, tmp_path, backend):
    """The check runs the wrapper in a subshell: an ``exit`` in it ends only the check."""
    env, runner = make_backend(backend)
    wrapper = 'sudo() { if [ "$1" = -A ] && [ "$2" = -V ]; then exit 0; fi; command sudo "$@"; }\n'
    (tmp_path / "functions.sh").write_text(wrapper, encoding="utf-8")
    runner.env["BASH_ENV"] = str(tmp_path / "functions.sh")
    marker = tmp_path / "before"
    result = env.execute(f"touch {marker}; echo before; sudo id -u && echo authed; cat", stdin_data="rest\n")
    _wait_gone(runner.temp_dir)
    assert marker.exists()
    assert result["returncode"] == 0, result
    assert result["output"].split() == ["before", str(os.getuid()), "authed", "rest"]
    assert PW not in result["output"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_sudo_wrapper_that_execs_cannot_end_the_command_early(make_backend, tmp_path, backend):
    """An ``exec`` wrapper replaces only the check's subshell; the command still runs up to its own
    sudo, which the wrapper then replaces the shell with, as it does on the stdin form."""
    env, runner = make_backend(backend)
    (tmp_path / "functions.sh").write_text('sudo() { exec sudo "$@"; }\n', encoding="utf-8")
    runner.env["BASH_ENV"] = str(tmp_path / "functions.sh")
    marker = tmp_path / "before"
    result = env.execute(f"touch {marker}; echo before; sudo id -u", stdin_data="rest\n")
    assert marker.exists()
    assert result["returncode"] == 0, result
    assert result["output"].split() == ["before", str(os.getuid())]
    assert PW not in result["output"]


def test_local_backend_sudo_rejecting_askpass_falls_back_to_stdin_form(tmp_path, monkeypatch):
    """LocalEnvironment with a probe answer of True: the prologue's own check still falls back."""
    from tools.environments.local import LocalEnvironment

    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_sudo_password_callback(None)
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    fake = _write_fake_sudo(tmp_path / "bin", accepts_askpass=False)
    monkeypatch.setenv("TERMINAL_TEMP_DIR", str(temp_dir))
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10, env={"TMPDIR": str(temp_dir)})
    try:
        assert env.execute(f"export PATH={fake.parent}:$PATH")["returncode"] == 0
        terminal_tool_sudo._set_cached_sudo_password(PW)
        env._sudo_askpass_ok = True  # as if the capability probe had passed
        result = env.execute(f"{fake} id -u && echo authed; cat", stdin_data="rest\n")
    finally:
        env.cleanup()
    assert result["returncode"] == 0, result
    assert result["output"].split() == [str(os.getuid()), "authed", "rest"]
    _wait_gone(temp_dir)


# --- the helper must run, and every sudo word the command runs must accept -A --------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_helper_that_cannot_run_falls_back_to_stdin_form(make_backend, tmp_path, backend):
    """A chmod that reports success but leaves the helper non-executable: setup falls back."""
    env, runner = make_backend(backend)
    shim = tmp_path / "bin" / "chmod"
    shim.write_text('#!/bin/sh\n/bin/chmod 600 "$2"\n')
    shim.chmod(0o700)
    result = env.execute("sudo id -u && echo authed; cat", stdin_data="rest\n")
    _wait_gone(runner.temp_dir)
    assert result["returncode"] == 0, result
    assert result["output"].split() == [str(os.getuid()), "authed", "rest"]


def _noexec_dir_or_skip():
    """A fresh directory on a writable mount that refuses to execute files, or skip."""
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        pytest.skip("no /proc/mounts to find a noexec mount")
    for line in mounts:
        fields = line.split()
        if len(fields) < 4 or "noexec" not in fields[3].split(",") or "rw" not in fields[3].split(","):
            continue
        mount = fields[1]
        if not os.path.isdir(mount) or not os.access(mount, os.W_OK | os.X_OK):
            continue
        try:
            directory = Path(tempfile.mkdtemp(prefix="hermes-noexec-test.", dir=mount))
        except OSError:
            continue
        script = directory / "x"
        try:
            script.write_text("#!/bin/sh\nexit 0\n")
            script.chmod(0o700)
            subprocess.run([str(script)], check=False, timeout=5)
        except PermissionError:
            script.unlink()
            return directory
        except OSError:
            pass
        shutil.rmtree(directory, ignore_errors=True)
    pytest.skip("no writable noexec mount available to this user (mounting one needs root)")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_noexec_temp_dir_falls_back_to_stdin_form(make_backend, backend):
    """TMPDIR on a real noexec mount: every setup step but running the helper succeeds."""
    env, runner = make_backend(backend)
    noexec = _noexec_dir_or_skip()
    try:
        runner.temp_dir = noexec
        runner.env["TMPDIR"] = str(noexec)
        result = env.execute("sudo id -u && echo authed; cat", stdin_data="rest\n")
        _wait_gone(noexec)
    finally:
        shutil.rmtree(noexec, ignore_errors=True)
    assert result["returncode"] == 0, result
    assert result["output"].split() == [str(os.getuid()), "authed", "rest"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_absolute_sudo_rejecting_askpass_uses_stdin_form(make_backend, tmp_path, backend):
    """PATH sudo accepts -A, the sudo the command names does not: the stdin form is used."""
    env, runner = make_backend(backend)
    target = _write_fake_sudo(tmp_path / "alternate", accepts_askpass=False)
    result = env.execute(f"{target} id -u && echo authed; cat", stdin_data="rest\n")
    _wait_gone(runner.temp_dir)
    assert result["returncode"] == 0, result
    assert result["output"].split() == [str(os.getuid()), "authed", "rest"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_absolute_sudo_accepting_askpass_uses_askpass_form(make_backend, tmp_path, backend):
    env, runner = make_backend(backend)
    target = _write_fake_sudo(tmp_path / "alternate")
    result = env.execute(f"{target} true && sudo true; cat", stdin_data="rest\n")
    _wait_gone(runner.temp_dir)
    assert result["returncode"] == 0, result
    assert PW not in result["output"]
    assert result["output"].strip() == "rest"


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_prologue_checks_each_distinct_sudo_word_as_written(make_backend, backend):
    env, _ = make_backend(backend)
    command = "/opt/x/sudo true && sudo id && env A=1 sudo true; '/opt/y z/sudo' true"
    exec_command, sudo_stdin = env._prepare_command(command)
    assert sudo_stdin == (PW + "\n") * 4  # one line per counted sudo, as for the stdin form
    checks = [line.strip() for line in exec_command.splitlines() if "-A -V" in line]
    assert checks == [f"( SUDO_ASKPASS=/bin/false {word} -A -V ) </dev/null >/dev/null 2>&1 &&"
                      for word in ("/opt/x/sudo", "sudo", "'/opt/y z/sudo'")]


def test_build_prologue_default_checks_bare_sudo():
    checks = [line.strip() for line in askpass.build_prologue(1, "10").splitlines() if "-A -V" in line]
    assert checks == ["( SUDO_ASKPASS=/bin/false sudo -A -V ) </dev/null >/dev/null 2>&1 &&"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_sudo_words_that_cannot_be_collected_keep_stdin_form(make_backend, backend):
    """Marker text inside the command makes the word lists outnumber the counted sudos."""
    env, _ = make_backend(backend)
    command = f"echo '{askpass.SUDO_ASKPASS_PREFIX}'; sudo id"
    expected = terminal_tool_sudo._rewrite_real_sudo_invocations(command)[0]
    assert env._prepare_command(command) == (expected, PW + "\n")


# --- controls: unchanged behavior ----------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_command_without_sudo_is_unchanged(make_backend, backend):
    env, runner = make_backend(backend)
    assert env._prepare_command("printf '%s\\n' hi; cat") == ("printf '%s\\n' hi; cat", None)
    result = env.execute("printf '%s\\n' hi; cat", stdin_data="rest\n")
    assert result["returncode"] == 0, result
    assert result["output"].split() == ["hi", "rest"]
    assert all("__hermes_sudo" not in command for command in runner.commands)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_sudo_with_own_stdin_option_keeps_the_stdin_form(make_backend, backend):
    env, _ = make_backend(backend)
    command = "sudo -S true"
    expected = terminal_tool_sudo._rewrite_real_sudo_invocations(command)[0]
    assert env._prepare_command(command) == (expected, PW + "\n")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdk_backend_without_password_leaves_sudo_command_alone(make_backend, backend):
    env, runner = make_backend(backend, password=False)
    assert env._prepare_command("sudo true && sudo true") == ("sudo true && sudo true", None)
    assert _probe_commands(runner) == []


def _probe_backends():
    from tools.environments.docker import DockerEnvironment
    from tools.environments.local import LocalEnvironment
    from tools.environments.ssh import SSHEnvironment
    return [LocalEnvironment, DockerEnvironment, SSHEnvironment]


@pytest.mark.parametrize("index", range(3))
def test_local_docker_ssh_still_run_the_capability_probe_once(index, monkeypatch):
    cls = _probe_backends()[index]
    monkeypatch.setattr("tools.terminal_tool_sudo_askpass.platform.system", lambda: "Linux")
    env = cls.__new__(cls)
    calls = []

    def run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
        calls.append(cmd_string)
        return cmd_string

    env._run_bash = run_bash
    env._wait_for_process = lambda proc, timeout=120, **_kw: {"output": "", "returncode": 0}
    assert askpass.sudo_askpass_supported_once(env) is True
    assert askpass.sudo_askpass_supported_once(env) is True
    assert calls == [askpass.CAPABILITY_PROBE]
