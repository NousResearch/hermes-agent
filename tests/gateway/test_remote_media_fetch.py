"""MEDIA tags pointing into a remote terminal sandbox are fetched for delivery (#466).

The host cannot see files an agent writes inside an ssh/modal/daytona sandbox, so
``filter_media_delivery_paths`` used to drop them silently. With a remote backend active the path is
pulled through the environment's ``fetch_file`` into the document cache — after the SAME denylist
the host applies, so a sandbox path (or a symlink) at ``~/.ssh/...`` never crosses.
"""

from pathlib import Path

import pytest

from gateway import media_fetch
from gateway.platforms.base import BasePlatformAdapter
from tools.environments.base import BaseEnvironment, FileFetchError


class _FakeRemoteEnv:
    """Stands in for a live SSHEnvironment: a tiny remote filesystem + symlink table."""

    _remote_home = "/home/agent"

    def __init__(self):
        self.files = {"/home/agent/out/report.txt": b"hello from sandbox",
                      "/home/agent/.ssh/id_rsa": b"SECRET", "/home/agent/app/.ENV": b"API_KEY=SECRET"}
        self.links = {"/home/agent/out/innocent.txt": "/home/agent/.ssh/id_rsa",
                      "/home/agent/out/notes.txt": "/home/agent/app/.ENV"}
        self.fetched: list = []

    def fetch_realpath(self, remote_path):
        return self.links.get(remote_path, remote_path)

    def fetch_file(self, remote_path, local_dest, *, max_bytes):
        self.fetched.append(remote_path)
        if remote_path not in self.files:
            raise FileFetchError("missing")
        Path(local_dest).write_bytes(self.files[remote_path])


@pytest.fixture
def remote_env(monkeypatch, tmp_path):
    env = _FakeRemoteEnv()
    monkeypatch.setattr(media_fetch, "_active_remote_env", lambda: env)
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "cache" / "documents")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path / "cache" / "documents",))
    return env


def test_sandbox_artifact_is_fetched_but_credentials_and_symlinks_to_them_are_not(remote_env):
    media = [("/home/agent/out/report.txt", False), ("/home/agent/out/innocent.txt", False),
             ("/home/agent/.ssh/id_rsa", False), ("~/out/report.txt", False)]
    delivered = BasePlatformAdapter.filter_media_delivery_paths(media)

    assert [Path(p).read_bytes() for p, _ in delivered] == [b"hello from sandbox", b"hello from sandbox"]
    assert all(Path(p).name.endswith("_report.txt") for p, _ in delivered)
    # The credential file and the symlink that resolves to it were refused BEFORE any bytes moved.
    assert remote_env.fetched == ["/home/agent/out/report.txt", "/home/agent/out/report.txt"]


def test_sandbox_project_env_file_is_not_fetched(remote_env):
    """The fetched copy is renamed, so the host basename rule must be applied to the sandbox path
    (and its symlink target) before any bytes move (#59044)."""
    media = [("/home/agent/app/.ENV", False), ("/home/agent/out/notes.txt", False)]
    assert BasePlatformAdapter.filter_media_delivery_paths(media) == []
    assert remote_env.fetched == []


@pytest.mark.parametrize("name", [".netrc", ".pgpass", ".npmrc", ".pypirc", ".git-credentials"])
@pytest.mark.parametrize("home", ["/root", "/home/agent", None])
def test_remote_home_credentials_and_symlinks_are_not_fetched(remote_env, name, home):
    remote_env._remote_home = home
    prefix = home or "/home/agent"
    secret = prefix + "/" + name
    ordinary = secret + "backup"
    alias = prefix + "/out/alias.txt"
    remote_env.files = {secret: b"placeholder", ordinary: b"artifact"}
    remote_env.links = {alias: secret}
    delivered = BasePlatformAdapter.filter_media_delivery_paths([(secret, False), (alias, False), (ordinary, True)])
    assert [(Path(path).read_bytes(), voice) for path, voice in delivered] == [(b"artifact", True)]
    assert remote_env.fetched == [ordinary]


@pytest.mark.parametrize("spelling", ["{}", '"{}"', "'{}'", "`{}`", "{},"])
def test_rejected_local_file_is_not_retried_remotely(remote_env, tmp_path, monkeypatch, spelling):
    import os
    from gateway.platforms import base

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    secret = home / ".netrc"
    secret.write_bytes(b"placeholder")
    alias = tmp_path / "cache" / "documents" / "alias.txt"
    alias.parent.mkdir(parents=True)
    os.link(secret, alias)
    remote_env._remote_home = str(tmp_path)
    remote_env.files = {str(alias): b"placeholder", str(tmp_path / "remote.txt"): b"artifact"}
    delivered = base.BasePlatformAdapter.filter_media_delivery_paths([
        (spelling.format(alias), False), (str(tmp_path / "remote.txt"), True)])
    assert [(Path(path).read_bytes(), voice) for path, voice in delivered] == [(b"artifact", True)]
    assert remote_env.fetched == [str(tmp_path / "remote.txt")]


def test_local_backend_and_strict_mode_do_not_fetch(monkeypatch, tmp_path, remote_env):
    """Strict mode keeps its recency gate: a fetched copy would land in an allowlisted root and skip it."""
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    assert BasePlatformAdapter.filter_media_delivery_paths([("/home/agent/out/report.txt", False)]) == []
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT")
    monkeypatch.setattr(media_fetch, "_active_remote_env", lambda: None)
    assert BasePlatformAdapter.filter_media_delivery_paths([(str(tmp_path / "nope.txt"), False)]) == []
    assert remote_env.fetched == []


class _ScriptedEnv(BaseEnvironment):
    """Runs the fetch command through a real shell so the transport (marker fencing, in-sandbox
    size bound, base64 round-trip) is exercised end to end."""

    def __init__(self):
        pass

    def cleanup(self):
        pass

    def execute(self, command, cwd="", **kwargs):
        import subprocess
        proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=False)
        return {"output": "echo login-noise\n" + proc.stdout + proc.stderr, "returncode": proc.returncode}


def test_fetch_file_round_trips_bytes_and_enforces_the_in_sandbox_size_cap(tmp_path):
    src = tmp_path / "artifact.bin"
    src.write_bytes(bytes(range(256)) * 40)
    dest = tmp_path / "copy.bin"
    env = _ScriptedEnv()

    env.fetch_file(str(src), dest, max_bytes=len(src.read_bytes()))
    assert dest.read_bytes() == src.read_bytes()

    with pytest.raises(FileFetchError, match="exceeds"):
        env.fetch_file(str(src), dest, max_bytes=100)
    with pytest.raises(FileFetchError, match="could not read"):
        env.fetch_file(str(tmp_path), dest, max_bytes=100)  # a directory is not a regular file


def test_translated_docker_credential_alias_is_not_retried_remotely(remote_env, tmp_path, monkeypatch):
    import os
    from gateway.platforms import base

    home = tmp_path / "container-home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    secret = home / ".netrc"
    secret.write_bytes(b"placeholder")
    os.link(secret, workspace / "alias.txt")
    monkeypatch.setattr(base, "_parse_docker_volume_mounts", lambda: [(home, Path("/root")), (workspace, Path("/workspace"))])
    monkeypatch.setattr(base, "_cache_dir_container_mounts", list)
    remote_env.files = {"/workspace/alias.txt": b"placeholder"}
    assert base.BasePlatformAdapter.filter_media_delivery_paths([("/workspace/alias.txt", False)], session_key="s") == []
    assert remote_env.fetched == []


def test_distinct_remote_artifact_survives_rejected_host_path_collision(remote_env, tmp_path, monkeypatch):
    from gateway.platforms import base

    host = tmp_path / "host"
    host.mkdir()
    collision = host / "artifact.txt"
    collision.write_bytes(b"local")
    monkeypatch.setattr(base, "_path_under_denied_prefix", lambda path: path == collision)
    remote_env._remote_home = str(tmp_path)
    remote_env.files = {str(collision): b"remote artifact"}
    delivered = base.BasePlatformAdapter.filter_media_delivery_paths([(str(collision), False)])
    assert [(Path(path).read_bytes(), voice) for path, voice in delivered] == [(b"remote artifact", False)]
    assert remote_env.fetched == [str(collision)]
