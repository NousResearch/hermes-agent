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

import contextvars
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import media_fetch
from gateway.config import Platform
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


SESSION_KEY = "slack:D0BQR84JBLY"
SESSION_ID = "0199ab00-1111-7000-8000-5f3b0c4d2e71"


@pytest.fixture
def registered_sandbox(monkeypatch, tmp_path):
    """A live ssh sandbox registered the way a turn registers it, with the turn's context GONE.

    Nothing is patched over ``_active_remote_env``: which sandbox it finds is what these tests are
    about. The registry key comes from the real resolver, so this asserts a relation between the
    turn's registration and the delivery lookup rather than a hardcoded ``session:`` spelling.
    """
    from tools import terminal_tool
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)  # strict skips the remote fetch
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "cache" / "documents")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path / "cache" / "documents",))
    env = _FakeRemoteEnv()

    def _turn(task_id_from_turn: str) -> str:
        tokens = set_session_vars(platform="slack", session_key=SESSION_KEY, session_id=SESSION_ID)
        try:
            return terminal_tool._resolve_container_task_id(task_id_from_turn)
        finally:
            clear_session_vars(tokens)  # the handler returned; delivery runs outside its context

    def _register(task_id_from_turn: str) -> str:
        # The turn binds and clears in a context of its own, as the handler's task does, so the
        # cleared vars never leak into this thread's context and the tests that run after it.
        key = contextvars.copy_context().run(_turn, task_id_from_turn)
        monkeypatch.setitem(terminal_tool._active_environments, key, env)
        return key

    return env, _register


def test_supplied_session_key_outranks_the_context_less_default_sandbox(monkeypatch, registered_sandbox):
    """Delivery after the turn's context is cleared: the key the call site hands down decides.

    Without it the session's sandbox is invisible and the artifact is lost. With a stale environment
    cached under the context-less ``"default"`` key -- a CLI or cron task's -- consulting that one
    first is worse than losing the file: it delivers a stranger's.
    """
    from tools import terminal_tool
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


class _DeliveryAdapter(BasePlatformAdapter):
    """Concrete adapter whose session store maps the turn's key to its session id."""

    def __init__(self, session_ids: dict):
        from gateway.config import Platform, PlatformConfig
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self._session_store = SimpleNamespace(peek_session_id=session_ids.get)

    async def connect(self) -> bool:  # pragma: no cover - not exercised
        return True

    async def disconnect(self) -> None:  # pragma: no cover - not exercised
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - not exercised
        return {"name": chat_id, "type": "dm"}

    async def send(self, chat_id, content, reply_to=None, metadata=None):  # pragma: no cover
        raise AssertionError("extraction must not send")


@pytest.mark.asyncio
@pytest.mark.parametrize("session_ids, expected", [
    ({SESSION_KEY: SESSION_ID}, [b"hello from sandbox"]),
    ({}, []),  # the store cannot name the session: no id, and the key alone does not find it
])
async def test_per_session_docker_sandbox_is_found_through_the_session_id(
        monkeypatch, registered_sandbox, session_ids, expected):
    """Per-session-isolated docker keys its sandbox by the turn's session ID, which delivery only
    has through the adapter's session store -- the key it is handed resolves to a different slot."""
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    _, register = registered_sandbox
    key = register(SESSION_ID)
    assert SESSION_ID in key and SESSION_KEY not in key  # docker isolation keys by session ID
    source = SessionSource(platform=Platform.SLACK, chat_id="D0BQR84JBLY", chat_type="dm", user_id="U1")

    extracted = await _DeliveryAdapter(session_ids)._extract_response_content(
        "here it is MEDIA:/home/agent/out/report.txt", MessageEvent(text="hi", source=source),
        SESSION_KEY, is_ephemeral_response=False)

    assert [Path(p).read_bytes() for p, _ in extracted.media_files] == expected
