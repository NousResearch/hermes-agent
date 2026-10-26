import base64
import io
import threading

import pytest

from hermes_cli import ssh_workspace_fs
from hermes_cli.ssh_workspace_fs import SshWorkspaceFs, SshWorkspaceFsError


class FakeEnv:
    cwd = "/srv/repos"
    timeout = 30
    _remote_home = "/home/dev"

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def execute(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return self.results.pop(0)


def _entry(name: str, path: str, kind: str) -> str:
    return "\t".join(
        (
            base64.b64encode(name.encode()).decode(),
            base64.b64encode(path.encode()).decode(),
            kind,
        )
    )


def test_ssh_workspace_fs_lists_and_reads_remote_paths():
    env = FakeEnv(
        [
            {
                "returncode": 0,
                "output": "\n".join(
                    [
                        _entry("z.txt", "/srv/repos/z.txt", "f"),
                        _entry("src", "/srv/repos/src", "d"),
                        _entry("node_modules", "/srv/repos/node_modules", "d"),
                    ]
                ),
            },
            {
                "returncode": 0,
                "output": "__HERMES_FS_SIZE__:5\naGVsbG8=\n",
            },
        ]
    )
    backend = SshWorkspaceFs(env)

    listing = backend.list_dir("/srv/repos", {"node_modules"})
    data, size, path = backend.read_bytes("README.md", max_bytes=100, read_limit=50)

    assert listing == {
        "entries": [
            {"name": "src", "path": "/srv/repos/src", "isDirectory": True},
            {"name": "z.txt", "path": "/srv/repos/z.txt", "isDirectory": False},
        ]
    }
    assert (data, size, path) == (b"hello", 5, "/srv/repos/README.md")
    assert 'p=/srv/repos' in env.calls[0][0]
    assert 'p=/srv/repos/README.md' in env.calls[1][0]
    assert 'if [ "$size" -gt 100 ]' in env.calls[1][0]
    assert 'set -o pipefail; head -c 50 < "$p" | base64' in env.calls[1][0]


def test_ssh_workspace_fs_maps_remote_size_failure():
    env = FakeEnv([{"returncode": 47, "output": "__HERMES_FS_ERROR__:EFBIG\n"}])
    backend = SshWorkspaceFs(env)

    with pytest.raises(SshWorkspaceFsError, match="EFBIG") as raised:
        backend.read_bytes("/srv/repos/large.bin", max_bytes=3)

    assert raised.value.code == "EFBIG"


def test_ssh_workspace_fs_inspects_resolved_file_before_streaming(monkeypatch):
    resolved = "/srv/repos/.env"
    encoded = base64.b64encode(resolved.encode()).decode()
    env = FakeEnv(
        [{"returncode": 0, "output": f"__HERMES_FS_TARGET__:{encoded}\n__HERMES_FS_SIZE__:6\n"}]
    )
    backend = SshWorkspaceFs(env)

    target, size = backend.inspect_file("safe-link")

    assert (target, size) == (resolved, 6)
    assert "realpath \"$p\"" in env.calls[0][0]

    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(b"streamed")

        def poll(self):
            return 0

        def wait(self):
            return 0

    process = FakeProcess()
    env._build_ssh_command = lambda: ["ssh", "dev@example.com"]
    monkeypatch.setattr(ssh_workspace_fs.subprocess, "Popen", lambda *args, **kwargs: process)

    assert b"".join(backend.stream_file(target, chunk_size=3)) == b"streamed"


def test_ssh_workspace_factory_disables_agent_file_sync(monkeypatch):
    created = []

    class FakeSshEnvironment:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.cwd = kwargs["cwd"]
            self.timeout = kwargs["timeout"]
            self._remote_home = "/home/dev"

        def cleanup(self):
            pass

    monkeypatch.setattr(ssh_workspace_fs, "SSHEnvironment", FakeSshEnvironment)
    monkeypatch.setattr(ssh_workspace_fs, "_BACKENDS", {})

    backend = ssh_workspace_fs.get_ssh_workspace_fs(
        "remote-dev",
        {
            "backend": "ssh",
            "cwd": "/srv/repos",
            "ssh_host": "ssh.example",
            "ssh_user": "dev",
            "ssh_port": 2222,
            "ssh_key": "/keys/id_ed25519",
            "timeout": 45,
        },
    )

    assert backend is not None
    assert backend.cwd == "/srv/repos"
    assert created == [
        {
            "host": "ssh.example",
            "user": "dev",
            "cwd": "/srv/repos",
            "timeout": 45,
            "port": 2222,
            "key_path": "/keys/id_ed25519",
            "sync_files": False,
        }
    ]


def _ssh_terminal(host: str) -> dict:
    return {"backend": "ssh", "cwd": "/srv/repos", "ssh_host": host, "ssh_user": "dev"}


@pytest.fixture
def gated_ssh(monkeypatch):
    """A fake SSHEnvironment that records each dial; a dial to a ``hanging`` host blocks until ``release``."""

    class GatedSshEnvironment:
        created: list = []
        cleaned: list = []
        hanging: set = set()
        failing: set = set()
        dialing = threading.Event()
        release = threading.Event()

        def __init__(self, **kwargs):
            self.created.append(kwargs["host"])
            if kwargs["host"] in self.hanging:
                self.dialing.set()
                self.release.wait(5)
            if kwargs["host"] in self.failing:
                raise OSError("connection timed out")
            self.host = kwargs["host"]
            self.cwd = kwargs["cwd"]
            self.timeout = kwargs["timeout"]
            self._remote_home = "/home/dev"

        def cleanup(self):
            self.cleaned.append(self.host)

    monkeypatch.setattr(ssh_workspace_fs, "SSHEnvironment", GatedSshEnvironment)
    monkeypatch.setattr(ssh_workspace_fs, "_BACKENDS", {})
    yield GatedSshEnvironment
    GatedSshEnvironment.release.set()


def test_ssh_workspace_factory_serves_other_profiles_while_one_host_hangs(gated_ssh):
    gated_ssh.hanging.add("down.example")
    results: dict = {}

    def open_profile(profile_key: str, host: str) -> None:
        results[profile_key] = ssh_workspace_fs.get_ssh_workspace_fs(profile_key, _ssh_terminal(host))

    down = threading.Thread(target=open_profile, args=("down", "down.example"), daemon=True)
    up = threading.Thread(target=open_profile, args=("up", "up.example"), daemon=True)
    down.start()
    try:
        assert gated_ssh.dialing.wait(5)
        up.start()
        up.join(2)
        assert not up.is_alive(), "a hanging dial on one profile blocked another profile"
        assert results["up"].cwd == "/srv/repos"
        assert "down" not in results
    finally:
        gated_ssh.release.set()
        down.join(5)
        if up.ident is not None:
            up.join(5)
    assert results["down"] is not None


def test_ssh_workspace_factory_dials_one_profile_once_for_concurrent_requests(gated_ssh):
    gated_ssh.hanging.add("ssh.example")
    backends: list = []

    def open_profile() -> None:
        backends.append(ssh_workspace_fs.get_ssh_workspace_fs("remote-dev", _ssh_terminal("ssh.example")))

    threads = [threading.Thread(target=open_profile, daemon=True) for _ in range(2)]
    threads[0].start()
    assert gated_ssh.dialing.wait(5)
    threads[1].start()
    gated_ssh.release.set()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()

    assert gated_ssh.created == ["ssh.example"]
    assert len(backends) == 2 and backends[0] is backends[1]


def test_ssh_workspace_factory_replaces_changed_config_and_drops_failed_redial(gated_ssh):
    first = ssh_workspace_fs.get_ssh_workspace_fs("remote-dev", _ssh_terminal("old.example"))
    assert ssh_workspace_fs.get_ssh_workspace_fs("remote-dev", _ssh_terminal("old.example")) is first

    second = ssh_workspace_fs.get_ssh_workspace_fs("remote-dev", _ssh_terminal("new.example"))
    assert second is not first
    assert gated_ssh.cleaned == ["old.example"]

    gated_ssh.failing.add("down.example")
    with pytest.raises(SshWorkspaceFsError, match="Could not connect") as raised:
        ssh_workspace_fs.get_ssh_workspace_fs("remote-dev", _ssh_terminal("down.example"))

    assert raised.value.code == "ECONN"
    assert gated_ssh.cleaned == ["old.example", "new.example"]
    assert "remote-dev" not in ssh_workspace_fs._BACKENDS


def test_ssh_workspace_factory_closes_replaced_connection_before_dialing_new_key(monkeypatch):
    """A key change on the same endpoint must not validate through the old key's control master."""
    events: list = []
    open_masters: set = set()

    class SharedSocketSsh:
        # Like SSHEnvironment, the control socket is keyed on user@host:port only, so a dial with
        # a new key reuses an open master and cleanup() closes it for every env on that endpoint.
        def __init__(self, **kwargs):
            self.key_path = kwargs["key_path"]
            self.socket = f"{kwargs['user']}@{kwargs['host']}:{kwargs['port']}"
            events.append(("dial", self.key_path))
            if self.socket not in open_masters:
                if self.key_path == "/keys/revoked":
                    raise OSError("Permission denied (publickey)")
                open_masters.add(self.socket)
            self.cwd = kwargs["cwd"]
            self.timeout = kwargs["timeout"]
            self._remote_home = "/home/dev"

        def cleanup(self):
            events.append(("cleanup", self.key_path))
            open_masters.discard(self.socket)

    monkeypatch.setattr(ssh_workspace_fs, "SSHEnvironment", SharedSocketSsh)
    monkeypatch.setattr(ssh_workspace_fs, "_BACKENDS", {})
    monkeypatch.setattr(ssh_workspace_fs, "_DIAL_LOCKS", {})

    def open_with(key: str):
        return ssh_workspace_fs.get_ssh_workspace_fs("remote-dev", {**_ssh_terminal("ssh.example"), "ssh_key": key})

    open_with("/keys/old")
    replacement = open_with("/keys/new")
    assert events == [("dial", "/keys/old"), ("cleanup", "/keys/old"), ("dial", "/keys/new")]
    assert open_masters == {"dev@ssh.example:22"}
    assert ssh_workspace_fs._BACKENDS["remote-dev"][1] is replacement

    with pytest.raises(SshWorkspaceFsError, match="Could not connect") as raised:
        open_with("/keys/revoked")

    assert raised.value.code == "ECONN"
    assert events[3:] == [("cleanup", "/keys/new"), ("dial", "/keys/revoked")]
    assert "remote-dev" not in ssh_workspace_fs._BACKENDS


def test_ssh_workspace_factory_failed_cleanup_does_not_wedge_the_profile(gated_ssh, monkeypatch):
    first = ssh_workspace_fs.get_ssh_workspace_fs("remote-dev", _ssh_terminal("old.example"))
    assert first is not None

    def broken_cleanup():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(first._env, "cleanup", broken_cleanup)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        ssh_workspace_fs.get_ssh_workspace_fs("remote-dev", _ssh_terminal("new.example"))

    assert not ssh_workspace_fs._DIAL_LOCKS["remote-dev"].locked()
    assert "remote-dev" not in ssh_workspace_fs._BACKENDS
    second = ssh_workspace_fs.get_ssh_workspace_fs("remote-dev", _ssh_terminal("new.example"))
    assert second is not first
    assert gated_ssh.created == ["old.example", "new.example"]
