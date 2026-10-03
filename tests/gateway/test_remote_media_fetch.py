"""MEDIA tags pointing into a remote terminal sandbox are fetched for delivery (#466).

The host cannot see files an agent writes inside an ssh/modal/daytona sandbox, so
``filter_media_delivery_paths`` used to drop them silently. With a remote backend active the path is
pulled through the environment's ``fetch_file`` into the document cache — after the SAME denylist
the host applies, so a sandbox path (or a symlink) at ``~/.ssh/...`` never crosses.

The last two tests cover WHICH sandbox the fallback finds. A gateway turn's delivery runs after the
session ContextVars are cleared, so the lookup has to work from the ``session_key`` the call site
carries as well as from the session id the turn registered under -- and a key the caller supplies
outranks whatever is cached under the context-less ``"default"`` key, or an artifact is fetched out
of an unrelated session's sandbox. Regression for #130875.
"""

from pathlib import Path

import pytest

import gateway.media_fetch as media_fetch
from gateway.platforms.base import BasePlatformAdapter
from tools.environments.base import BaseEnvironment, FileFetchError


class _FakeRemoteEnv:
    """Stands in for a live SSHEnvironment: a tiny remote filesystem + symlink table."""

    _remote_home = "/home/agent"

    def __init__(self, payload: bytes = b"hello from sandbox"):
        self.files = {"/home/agent/out/report.txt": payload,
                      "/home/agent/.ssh/id_rsa": b"SECRET"}
        self.links = {"/home/agent/out/innocent.txt": "/home/agent/.ssh/id_rsa"}
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
        proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
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


SESSION_KEY = "slack:D0BQR84JBLY"
SESSION_ID = "0199ab00-1111-7000-8000-5f3b0c4d2e71"


@pytest.fixture
def registered_sandbox(monkeypatch, tmp_path):
    """A live ssh sandbox registered the way a turn registers it, with the turn's context GONE.

    Nothing is patched over ``_active_remote_env``: which sandbox it finds is what these tests are
    about. The registry key comes from the real resolver, so this asserts a relation between the
    turn's registration and the delivery lookup rather than a hardcoded ``session:`` spelling.
    """
    import tools.terminal_tool as terminal_tool
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "cache" / "documents")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path / "cache" / "documents",))
    env = _FakeRemoteEnv()

    def _register(task_id_from_turn: str) -> str:
        tokens = set_session_vars(platform="slack", session_key=SESSION_KEY, session_id=SESSION_ID)
        try:
            key = terminal_tool._resolve_container_task_id(task_id_from_turn)
        finally:
            clear_session_vars(tokens)  # the handler returned; delivery runs outside its context
        monkeypatch.setitem(terminal_tool._active_environments, key, env)
        return key

    return env, _register


def test_supplied_session_key_outranks_the_context_less_default_sandbox(monkeypatch, registered_sandbox):
    """Delivery after the turn's context is cleared: the key the call site hands down decides.

    Without it the session's sandbox is invisible and the artifact is lost. With a stale environment
    cached under the context-less ``"default"`` key -- a CLI or cron task's -- consulting that one
    first is worse than losing the file: it delivers a stranger's.
    """
    import tools.terminal_tool as terminal_tool
    env, register = registered_sandbox
    key = register(SESSION_ID)
    assert SESSION_KEY in key and SESSION_ID not in key  # ssh keys by session KEY, not session id
    unrelated = _FakeRemoteEnv(b"from an unrelated sandbox")
    monkeypatch.setitem(terminal_tool._active_environments, "default", unrelated)

    delivered = BasePlatformAdapter.filter_media_delivery_paths(
        [("/home/agent/out/report.txt", False)], session_key=SESSION_KEY)

    assert [Path(p).read_bytes() for p, _ in delivered] == [b"hello from sandbox"]
    assert env.fetched == ["/home/agent/out/report.txt"]
    assert unrelated.fetched == []


def test_sandbox_keyed_by_session_id_is_found_without_a_session_key(monkeypatch, registered_sandbox):
    """Per-session-isolated docker keys by task id, so the session-id lookup must keep working."""
    import tools.terminal_tool as terminal_tool
    from gateway.session_context import scoped_current_session_id
    env, _ = registered_sandbox
    monkeypatch.setitem(terminal_tool._active_environments, SESSION_ID, env)

    with scoped_current_session_id(SESSION_ID):
        delivered = BasePlatformAdapter.filter_media_delivery_paths([("/home/agent/out/report.txt", False)])

    assert [Path(p).read_bytes() for p, _ in delivered] == [b"hello from sandbox"]
