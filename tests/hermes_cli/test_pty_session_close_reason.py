"""4410 also ends a shell the server closed (expiry, a deleted profile, shutdown). Only a shell that
exited by itself says so in the close reason, so a viewer can end its tab the way native Desktop
does when its shell exits, and keep the tab for the other cases."""
import asyncio

import pytest

from tests.hermes_cli.test_pty_session import FakeBridge, FakeWS
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_eof_close_says_the_process_exited():
    from hermes_cli.pty_session import PtySession
    s = PtySession("k", FakeBridge([b"logout", None]), buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    await async_wait_until(lambda: ws.close_code is not None, timeout=2.0)
    assert (ws.close_code, ws.close_reason) == (4410, "Process exited")
    await s.close()


@pytest.mark.asyncio
async def test_attaching_to_a_shell_that_exited_while_detached_says_it_exited():
    from hermes_cli.pty_session import PtySession
    s = PtySession("k", FakeBridge([b"logout", None]), buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await async_wait_until(lambda: not s.alive, timeout=2.0)
    ws = FakeWS()
    assert await s.attach(ws, initial_text="meta") is False
    assert ws.sent == [("text", "meta"), ("bytes", b"logout")]
    assert (ws.close_code, ws.close_reason) == (4410, "Process exited")
    await s.close()


@pytest.mark.asyncio
async def test_a_server_close_keeps_4410_without_the_exit_reason():
    from hermes_cli.pty_session import PtySession
    s = PtySession("k", FakeBridge([]), buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    await s.close()
    assert ws.close_code == 4410
    assert ws.close_reason != "Process exited"


@pytest.mark.asyncio
async def test_a_shell_reaped_while_its_redial_is_validated_closes_the_redial_as_an_exit(tmp_path, monkeypatch):
    """A redial whose shell exited, and which the reaper takes while ``resolve`` checks the
    profile off-loop, is still told the shell exited, so the tab closes instead of offering a new
    shell."""
    import secrets
    import threading
    from types import SimpleNamespace

    from hermes_cli import pty_session, web_host_terminal_sessions
    from hermes_cli.web_host_terminal_sessions import HostOwner, HostTerminalRegistry, TerminalExited

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    identity = ("local", "owner")
    registry = HostTerminalRegistry()
    token = secrets.token_urlsafe(32)
    bridge = FakeBridge([])
    bridge.dead = True  # the shell exited while no tab was attached
    registry._sessions[token] = pty_session.PtySession(token, bridge, buffer_cap=1024, read_timeout=0.01)
    registry.owners[token] = HostOwner(identity, home.resolve(), "sh", str(tmp_path))

    validating, release = threading.Event(), threading.Event()
    request_home = web_host_terminal_sessions._request_home

    def held_request_home(profile):
        validating.set()
        release.wait(5)
        return request_home(profile)

    monkeypatch.setattr(web_host_terminal_sessions, "_request_home", held_request_home)
    redial = SimpleNamespace(query_params={"attach": token})
    try:
        resolving = asyncio.ensure_future(registry.resolve(token, redial, identity))
        assert await asyncio.to_thread(validating.wait, 5)
        await registry.reap_idle()
        assert token not in registry._sessions and token in registry.exits
        release.set()
        with pytest.raises(TerminalExited):
            await resolving
    finally:
        release.set()
        await registry.close_all()
