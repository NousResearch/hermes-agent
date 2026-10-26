"""4410 also ends a shell the server closed (expiry, a deleted profile, shutdown). Only a shell that
exited by itself says so in the close reason, so a viewer can end its tab the way native Desktop
does when its shell exits, and keep the tab for the other cases."""
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
