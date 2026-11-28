"""A slash-confirm "Always Approve" must land in the PROFILE's own config (#142157).

The gate reads its flag from the config of the profile serving the chat: the inbound message
handler binds that profile's scope, so ``load_config()`` resolves the profile's
``config.yaml``. The "always" callback fires later - from the button's interaction handler,
outside that scope - so an unscoped ``save_config_value()`` wrote the LAUNCH home's
``config.yaml``: the profile never saw the opt-out, the confirmation prompt came back on every
``/new``, and the root config was rewritten on every click.

The opt-out belongs to the profile that was asked, so the home is captured with the prompt and
re-entered for the write.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import hermes_yaml as yaml
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    """The bare runner the existing destructive-slash tests drive (no gateway process)."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    # No send_slash_confirm override -> button render returns None,
    # _request_slash_confirm falls back to the text path.
    adapter.send_slash_confirm = AsyncMock(return_value=None)
    runner.adapters = {Platform.TELEGRAM: adapter}

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()

    runner._running_agents = {}
    runner._pending_messages = {}
    import itertools as _it
    runner._slash_confirm_counter = _it.count(1)
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner._thread_metadata_for_source = lambda *a, **kw: None
    runner._reply_anchor_for_event = lambda _e: None
    return runner


def _profile_home(tmp_path, monkeypatch) -> tuple:
    """A launch home with a routed profile under it; both get a config.yaml."""
    launch_home = tmp_path / "hermes"
    profile_home = launch_home / "profiles" / "brain"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("display:\n  theme: dark\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    return launch_home, profile_home


@pytest.mark.asyncio
async def test_always_approve_writes_the_profiles_own_config(tmp_path, monkeypatch):
    """The opt-out lands in the file the gate was read from, not in the launch home's."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import slash_confirm as _slash_confirm_mod

    launch_home, profile_home = _profile_home(tmp_path, monkeypatch)
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._session_key_for_source = lambda src: session_key
    _slash_confirm_mod.clear(session_key)

    # The gate runs inside the routed profile's scope (the inbound handler binds it).
    token = set_hermes_home_override(str(profile_home))
    try:
        await runner._maybe_confirm_destructive_slash(
            event=_make_event("/new"),
            command="new",
            title="/new",
            detail="Discards history.",
            execute=AsyncMock(return_value="fresh"),
        )
    finally:
        reset_hermes_home_override(token)

    pending = _slash_confirm_mod.get_pending(session_key)
    assert pending is not None, "the gate did not ask for confirmation"

    # The callback fires later, OUTSIDE the profile scope: the button's interaction handler
    # runs in the ambient scope, not the one the prompt was raised in.
    resolved = await _slash_confirm_mod.resolve(session_key, pending["confirm_id"], "always")
    assert resolved is not None

    profile_cfg = yaml.safe_load((profile_home / "config.yaml").read_text(encoding="utf-8-sig"))
    assert profile_cfg["approvals"]["destructive_slash_confirm"] is False, (
        "the opt-out must be written to the profile that was asked")
    launch_cfg_path = launch_home / "config.yaml"
    launch_cfg = (
        yaml.safe_load(launch_cfg_path.read_text(encoding="utf-8-sig"))
        if launch_cfg_path.exists() else {}
    )
    assert "approvals" not in (launch_cfg or {}), (
        "the launch home's config must not be rewritten by a routed profile's opt-out")
    _slash_confirm_mod.clear(session_key)


@pytest.mark.asyncio
async def test_always_approve_keeps_writing_the_home_the_gate_read(tmp_path, monkeypatch):
    """A single-profile gateway keeps writing the launch home's config (unchanged behaviour)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import slash_confirm as _slash_confirm_mod

    launch_home = tmp_path / "hermes"
    launch_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text("display:\n  theme: dark\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._session_key_for_source = lambda src: session_key
    _slash_confirm_mod.clear(session_key)

    token = set_hermes_home_override(str(launch_home))
    try:
        await runner._maybe_confirm_destructive_slash(
            event=_make_event("/new"),
            command="new",
            title="/new",
            detail="Discards history.",
            execute=AsyncMock(return_value="fresh"),
        )
    finally:
        reset_hermes_home_override(token)

    pending = _slash_confirm_mod.get_pending(session_key)
    assert pending is not None
    await _slash_confirm_mod.resolve(session_key, pending["confirm_id"], "always")

    launch_cfg = yaml.safe_load((launch_home / "config.yaml").read_text(encoding="utf-8-sig"))
    assert launch_cfg["approvals"]["destructive_slash_confirm"] is False
    _slash_confirm_mod.clear(session_key)


@pytest.mark.asyncio
async def test_the_opt_out_is_per_profile_across_an_a_b_a_round_trip(tmp_path, monkeypatch):
    """A -> B -> A: the opt-out belongs to the profile that was asked, and no other.

    The contributing rubric asks a profile-scope change to exercise two homes with real imports
    and the A->B->A round trip. Here: profile A opts out, profile B is still asked (A's opt-out
    did not leak), and A is still opted out when the round trip comes back to it.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import slash_confirm as _slash_confirm_mod

    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    for home in (home_a, home_b):
        home.mkdir()
        (home / "config.yaml").write_text("display:\n  theme: dark\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home_a))

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._session_key_for_source = lambda src: session_key

    async def _ask(home):
        """Raise the gate in *home*'s scope; return the pending confirm (or None)."""
        _slash_confirm_mod.clear(session_key)
        token = set_hermes_home_override(str(home))
        try:
            await runner._maybe_confirm_destructive_slash(
                event=_make_event("/new"),
                command="new",
                title="/new",
                detail="Discards history.",
                execute=AsyncMock(return_value="fresh"),
            )
        finally:
            reset_hermes_home_override(token)
        return _slash_confirm_mod.get_pending(session_key)

    # A is asked, and opts out.
    pending = await _ask(home_a)
    assert pending is not None, "the gate did not ask for confirmation in A"
    await _slash_confirm_mod.resolve(session_key, pending["confirm_id"], "always")
    a_cfg = yaml.safe_load((home_a / "config.yaml").read_text(encoding="utf-8-sig"))
    assert a_cfg["approvals"]["destructive_slash_confirm"] is False

    # B is a different profile: A's opt-out must not have leaked into it.
    pending = await _ask(home_b)
    assert pending is not None, "A's opt-out leaked into B"
    b_cfg = yaml.safe_load((home_b / "config.yaml").read_text(encoding="utf-8-sig"))
    assert "approvals" not in (b_cfg or {}), "B's config was rewritten by A's opt-out"
    await _slash_confirm_mod.resolve(session_key, pending["confirm_id"], "cancel")

    # Back in A: the opt-out still stands, so the gate does not ask again.
    pending = await _ask(home_a)
    assert pending is None, "A's opt-out did not survive the round trip"
    _slash_confirm_mod.clear(session_key)
