import base64
import io

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


def test_ssh_workspace_fs_inspects_resolved_remote_file():
    resolved = "/srv/repos/.env"
    encoded = base64.b64encode(resolved.encode()).decode()
    env = FakeEnv(
        [
            {
                "returncode": 0,
                "output": f"__HERMES_FS_SIZE__:8\n__HERMES_FS_PATH__:{encoded}\n",
            }
        ]
    )
    backend = SshWorkspaceFs(env)

    assert backend.inspect_file("safe-link") == (resolved, 8)
    assert "realpath \"$p\"" in env.calls[0][0]


def test_ssh_workspace_fs_streams_remote_file_without_buffering(monkeypatch):
    env = FakeEnv([])
    env._build_ssh_command = lambda: ["ssh", "dev@example"]
    captured = {}

    class FakeProcess:
        stdout = io.BytesIO(b"hello")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(ssh_workspace_fs.subprocess, "Popen", fake_popen)
    backend = SshWorkspaceFs(env)

    assert list(backend.stream_file("README.md", chunk_size=2)) == [b"he", b"ll", b"o"]
    assert captured["command"][:2] == ["ssh", "dev@example"]
    assert "/srv/repos/README.md" in captured["command"][-1]


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
