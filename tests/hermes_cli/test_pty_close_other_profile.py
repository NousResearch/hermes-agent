"""``close_other_sessions`` must only displace *other-profile* siblings when asked.

Upstream closes every PTY sharing an attach token on a canonical-key change so a
profile switch cannot leave a lease-holding orphan. The dashboard tab, however,
keeps the terminals of conversations it left (same profile) so a later explicit
resume reattaches to the live one instead of being refused the session lease.
"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pty_session_fixtures", Path(__file__).with_name("test_pty_session.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
FakeBridge = fixtures.FakeBridge
make_registry = fixtures.make_registry


async def _started(key: str, reg):
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([b""])
    session = PtySession(key, bridge, buffer_cap=1024, read_timeout=0.01)
    await session.start()
    reg._sessions[key] = session
    return session, bridge


@pytest.mark.asyncio
async def test_same_profile_siblings_survive_but_other_profiles_close():
    reg = make_registry()
    old_profile, old_bridge = await _started("token\0alpha\0session-a", reg)
    default_fresh, default_bridge = await _started("token", reg)
    same_profile, same_bridge = await _started("token\0beta\0session-x", reg)
    forked_same, forked_bridge = await _started("token\0beta\0session-x\0deadbeef", reg)
    current, current_bridge = await _started("token\0beta\0session-b", reg)

    await reg.close_other_sessions("token", keep_key=current.key, keep_profile="beta")

    assert old_bridge.closed and old_profile.key not in reg._sessions
    assert default_bridge.closed and default_fresh.key not in reg._sessions
    for session, bridge in ((same_profile, same_bridge), (forked_same, forked_bridge), (current, current_bridge)):
        assert not bridge.closed
        assert reg._sessions[session.key] is session
    await reg.close_all()


@pytest.mark.asyncio
async def test_default_profile_keeps_bare_and_owner_suffixed_fresh_terminals():
    reg = make_registry()
    bare, bare_bridge = await _started("token", reg)
    forked_bare, forked_bridge = await _started("token\0cafebabe", reg)
    named, named_bridge = await _started("token\0alpha\0session-a", reg)
    current, current_bridge = await _started("token\0\0session-b", reg)

    await reg.close_other_sessions("token", keep_key=current.key, keep_profile="")

    assert named_bridge.closed and named.key not in reg._sessions
    for session, bridge in ((bare, bare_bridge), (forked_bare, forked_bridge), (current, current_bridge)):
        assert not bridge.closed
        assert reg._sessions[session.key] is session
    await reg.close_all()


@pytest.mark.asyncio
async def test_without_keep_profile_every_sibling_closes():
    reg = make_registry()
    _, a_bridge = await _started("token\0beta\0session-a", reg)
    _, bare_bridge = await _started("token", reg)
    current, current_bridge = await _started("token\0beta\0session-b", reg)
    _, other_token_bridge = await _started("other\0beta\0session-c", reg)

    await reg.close_other_sessions("token", keep_key=current.key)

    assert a_bridge.closed and bare_bridge.closed
    assert not current_bridge.closed and not other_token_bridge.closed
    assert set(reg._sessions) == {current.key, "other\0beta\0session-c"}
    await reg.close_all()
