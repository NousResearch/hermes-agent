"""E2E regression tests for the Buzz thread-topology salvage cluster.

Covers the composed behavior of PRs #77080 / #79578 / #80120 / #85613 /
#86232 / #89868 (+ issues #75082, #95841, #95842):

1. NIP-10 thread-root anchoring — replies join the EXISTING thread root
   instead of nesting a new sub-thread per turn, across send(),
   send_image(), and inbound session thread_id.
2. reply_in_thread / reply_to_mode config honoring — the opt-out posts
   flat on every send path, including progress routing and the
   out-of-process cron sender.
3. _PLATFORM_DEFAULTS coverage — buzz no longer inherits the verbose
   _GLOBAL_DEFAULTS (#95841).

Uses the real adapter module (no gateway process) with synthetic NIP-10
events shaped like live relay traffic.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_buzz_module():
    path = REPO_ROOT / "plugins" / "platforms" / "buzz" / "adapter.py"
    spec = importlib.util.spec_from_file_location("plugin_adapter_buzz_threads", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_buzz_mod = _load_buzz_module()
BuzzAdapter = _buzz_mod.BuzzAdapter

CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"
SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
OTHER_PUBKEY = "b" * 64
ROOT_EVT = "a" * 64
MID_EVT = "c" * 64


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch, tmp_path):
    for var in (
        "BUZZ_RELAY_URL", "BUZZ_CHANNELS", "BUZZ_HOME_CHANNEL",
        "BUZZ_POLL_INTERVAL", "BUZZ_CLI_PATH", "BUZZ_CREDENTIALS_FILE",
        "BUZZ_ALLOWED_USERS", "BUZZ_ALLOW_ALL_USERS", "BUZZ_PRIVATE_KEY",
        "BUZZ_REQUIRE_MENTION", "BUZZ_REPLY_IN_THREAD", "BUZZ_REPLY_TO_MODE",
        "BUZZ_TRANSPORT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path / "no-creds")
    yield


def _make_adapter(extra=None, **cfg_kwargs):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(
        enabled=True,
        extra={"relay_url": "https://test.relay", **(extra or {})},
        **cfg_kwargs,
    )
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._self_npub = _buzz_mod.hex_to_npub(SELF_PUBKEY)
    adapter._display_name = "Chip"
    adapter._private_key = "nsec1test"
    return adapter


class _CapturingCli:
    def __init__(self, payload=None):
        self.calls = []
        self.payload = payload or {"accepted": True, "event_id": "evt-out"}

    async def __call__(self, args, *, input_text=None):
        self.calls.append((list(args), input_text))
        return 0, json.dumps(self.payload), ""


def _nip10_reply_event(event_id, *, root, parent, content="in thread", pubkey=OTHER_PUBKEY):
    """A kind-9 event shaped like a live relay in-thread reply."""
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": 1000,
        "kind": 9,
        "tags": [
            ["h", CHANNEL],
            ["e", root, "", "root"],
            ["e", parent, "", "reply"],
        ],
    }


def _top_level_event(event_id, content="@Chip hello", pubkey=OTHER_PUBKEY):
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": 1000,
        "kind": 9,
        "tags": [["h", CHANNEL]],
    }



async def _stub_cli(args, *, input_text=None):
    return 0, "[]", ""

# ── 1. Thread-root anchoring ──────────────────────────────────────────────


class TestThreadRootAnchoring:

    @pytest.mark.asyncio
    async def test_reply_to_in_thread_trigger_anchors_to_root(self):
        """E2E: inbound NIP-10 reply -> send(reply_to=<trigger>) -> --reply-to <root>."""
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        adapter._message_handler = AsyncMock()
        adapter.handle_message = AsyncMock()
        adapter.send_reaction = AsyncMock(return_value=True)
        adapter._run_cli = _stub_cli

        event = _nip10_reply_event("trigger-evt", root=ROOT_EVT, parent=MID_EVT,
                                   content="@Chip what next?")
        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

        cli = _CapturingCli()
        adapter._run_cli = cli
        await adapter.send(CHANNEL, "the answer", reply_to="trigger-evt")
        args, _ = cli.calls[0]
        assert args[args.index("--reply-to") + 1] == ROOT_EVT

    @pytest.mark.asyncio
    async def test_reply_to_top_level_trigger_opens_one_thread(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        adapter._message_handler = AsyncMock()
        adapter.handle_message = AsyncMock()
        adapter.send_reaction = AsyncMock(return_value=True)
        adapter._run_cli = _stub_cli

        await adapter._handle_event(
            CHANNEL, adapter._channel_state[CHANNEL], _top_level_event("root-msg")
        )
        cli = _CapturingCli()
        adapter._run_cli = cli
        await adapter.send(CHANNEL, "answer", reply_to="root-msg")
        args, _ = cli.calls[0]
        assert args[args.index("--reply-to") + 1] == "root-msg"

    @pytest.mark.asyncio
    async def test_inbound_thread_id_is_nip10_root(self):
        """Session scoping: dispatched source.thread_id is the stable root."""
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        dispatched = []

        async def capture(event):
            dispatched.append(event)

        adapter._message_handler = AsyncMock()
        adapter.handle_message = capture
        adapter.send_reaction = AsyncMock(return_value=True)
        adapter._run_cli = _stub_cli

        event = _nip10_reply_event("child-evt", root=ROOT_EVT, parent=MID_EVT,
                                   content="@Chip follow-up")
        await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)
        assert dispatched and dispatched[0].source.thread_id == ROOT_EVT
        assert dispatched[0].source.message_id == "child-evt"

    @pytest.mark.asyncio
    async def test_send_image_anchors_to_root_too(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG fake")
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        adapter._record_thread_root(
            "trigger-evt", _nip10_reply_event("trigger-evt", root=ROOT_EVT, parent=MID_EVT)
        )
        cli = _CapturingCli()
        adapter._run_cli = cli
        await adapter.send_image(CHANNEL, str(img), caption="pic", reply_to="trigger-evt")
        args, _ = cli.calls[0]
        assert args[args.index("--reply-to") + 1] == ROOT_EVT


# ── 2. Config honoring: reply_in_thread / reply_to_mode ─────────────────


class TestReplyThreadingConfig:

    @pytest.mark.asyncio
    async def test_default_threads_replies(self):
        adapter = _make_adapter()
        cli = _CapturingCli()
        adapter._run_cli = cli
        await adapter.send(CHANNEL, "hi", reply_to="evt-1")
        assert "--reply-to" in cli.calls[0][0]

    @pytest.mark.asyncio
    async def test_reply_in_thread_false_posts_flat(self):
        adapter = _make_adapter(extra={"reply_in_thread": False})
        assert adapter._reply_to_mode == "off"
        cli = _CapturingCli()
        adapter._run_cli = cli
        await adapter.send(CHANNEL, "hi", reply_to="evt-1",
                           metadata={"thread_id": "evt-1", "reply_to_message_id": "evt-1"})
        assert "--reply-to" not in cli.calls[0][0]

    @pytest.mark.asyncio
    async def test_reply_to_mode_off_posts_flat(self):
        adapter = _make_adapter(reply_to_mode="off")
        cli = _CapturingCli()
        adapter._run_cli = cli
        await adapter.send(CHANNEL, "hi", reply_to="evt-1")
        assert "--reply-to" not in cli.calls[0][0]

    @pytest.mark.asyncio
    async def test_env_reply_in_thread_false_wins(self, monkeypatch):
        monkeypatch.setenv("BUZZ_REPLY_IN_THREAD", "false")
        adapter = _make_adapter()
        assert adapter._reply_to_mode == "off"

    @pytest.mark.asyncio
    async def test_reply_in_thread_true_keeps_threading(self):
        adapter = _make_adapter(extra={"reply_in_thread": True})
        assert adapter._reply_to_mode != "off"

    @pytest.mark.asyncio
    async def test_send_image_honors_opt_out(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG fake")
        adapter = _make_adapter(extra={"reply_in_thread": False})
        cli = _CapturingCli()
        adapter._run_cli = cli
        await adapter.send_image(CHANNEL, str(img), caption="pic", reply_to="evt-1")
        assert "--reply-to" not in cli.calls[0][0]

    def test_apply_yaml_config_bridges_keys(self, monkeypatch):
        monkeypatch.delenv("BUZZ_REPLY_IN_THREAD", raising=False)
        monkeypatch.delenv("BUZZ_REPLY_TO_MODE", raising=False)
        _buzz_mod._apply_yaml_config(
            {}, {"extra": {"reply_in_thread": False, "reply_to_mode": "off"}}
        )
        import os
        assert os.environ["BUZZ_REPLY_IN_THREAD"] == "false"
        assert os.environ["BUZZ_REPLY_TO_MODE"] == "off"
        monkeypatch.delenv("BUZZ_REPLY_IN_THREAD", raising=False)
        monkeypatch.delenv("BUZZ_REPLY_TO_MODE", raising=False)

    @pytest.mark.asyncio
    async def test_standalone_send_honors_opt_out(self, monkeypatch, tmp_path):
        """Out-of-process cron delivery must not thread when opted out."""
        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_cli.chmod(0o755)
        monkeypatch.setenv("BUZZ_REPLY_IN_THREAD", "false")

        captured = {}

        async def fake_exec(cli_path, args, *, relay_url, private_key, auth_tag="", input_text=None, timeout=None):
            captured["args"] = args
            return 0, json.dumps({"accepted": True, "event_id": "evt-cron"}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        class _PC:
            extra = {"relay_url": "https://test.relay", "cli_path": str(fake_cli)}

        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1test")
        result = await _buzz_mod._standalone_send(_PC(), CHANNEL, "cron msg", thread_id="evt-1")
        assert result.get("success") is True
        assert "--reply-to" not in captured["args"]

    @pytest.mark.asyncio
    async def test_standalone_send_threads_by_default(self, monkeypatch, tmp_path):
        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_cli.chmod(0o755)
        captured = {}

        async def fake_exec(cli_path, args, *, relay_url, private_key, auth_tag="", input_text=None, timeout=None):
            captured["args"] = args
            return 0, json.dumps({"accepted": True, "event_id": "evt-cron"}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        class _PC:
            extra = {"relay_url": "https://test.relay", "cli_path": str(fake_cli)}

        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1test")
        result = await _buzz_mod._standalone_send(_PC(), CHANNEL, "cron msg", thread_id="evt-1")
        assert result.get("success") is True
        assert "--reply-to" in captured["args"]
        assert captured["args"][captured["args"].index("--reply-to") + 1] == "evt-1"


# ── 3. Progress routing honors the opt-out ───────────────────────────────


class TestProgressRouting:

    def test_adapter_owned_progress_does_not_get_a_core_synthetic_thread(self):
        from gateway.run import _resolve_progress_thread_id

        assert _resolve_progress_thread_id(
            "buzz", source_thread_id=None, event_message_id="evt-1",
            reply_in_thread=True,
        ) is None

    def test_buzz_progress_flat_when_opted_out(self):
        from gateway.run import _resolve_progress_thread_id

        assert _resolve_progress_thread_id(
            "buzz", source_thread_id=None, event_message_id="evt-1",
            reply_in_thread=False,
        ) is None


# ── 4. Per-channel reply placement ──────────────────────────────────


def _reply_arg(args):
    if "--reply-to" not in args:
        return None
    return args[args.index("--reply-to") + 1]


@pytest.mark.parametrize(
    ("mode", "placement", "expected"),
    [
        ("flat", "top_level", None),
        ("flat", "in_thread", None),
        ("threaded", "top_level", "trigger-evt"),
        ("threaded", "in_thread", ROOT_EVT),
        ("hybrid", "top_level", None),
        ("hybrid", "in_thread", ROOT_EVT),
    ],
)
@pytest.mark.asyncio
async def test_channel_reply_mode_matrix_covers_text_image_and_file(
    tmp_path, mode, placement, expected
):
    """Every reply-capable send path obeys one channel policy resolver."""
    adapter = _make_adapter(
        extra={
            "channel_modes": {CHANNEL: {"replies": mode}},
            # Explicit per-channel modes must win over inherited global flat.
            "reply_in_thread": False,
        }
    )
    cli = _CapturingCli()
    adapter._run_cli = cli
    attachment = tmp_path / "artifact.png"
    attachment.write_bytes(b"\x89PNG fake")
    metadata = {
        "reply_to_message_id": "trigger-evt",
        "buzz_trigger_placement": placement,
    }
    if placement == "in_thread":
        metadata["thread_id"] = ROOT_EVT

    await adapter.send(
        CHANNEL, "text", reply_to="trigger-evt", metadata=metadata
    )
    await adapter.send_image(
        CHANNEL,
        str(attachment),
        caption="image",
        reply_to="trigger-evt",
        metadata=metadata,
    )
    await adapter._send_local_file(
        CHANNEL,
        str(attachment),
        caption="file",
        reply_to="trigger-evt",
        metadata=metadata,
    )

    assert [_reply_arg(args) for args, _ in cli.calls] == [expected] * 3


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("threaded", ["top-trigger", ROOT_EVT]),
        ("hybrid", [None, ROOT_EVT]),
    ],
)
@pytest.mark.asyncio
async def test_gateway_buzz_metadata_keeps_trigger_and_placement_for_progress(
    mode, expected
):
    """Gateway metadata stays rich even when inherited global replies are flat."""
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    adapter = _make_adapter(
        extra={
            "reply_in_thread": False,
            "channel_modes": {CHANNEL: {"replies": mode}},
        }
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.BUZZ: adapter}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    top_source = SimpleNamespace(
        platform=Platform.BUZZ,
        chat_id=CHANNEL,
        chat_type="group",
        thread_id=None,
        message_id="top-trigger",
        profile=None,
    )
    thread_source = SimpleNamespace(
        platform=Platform.BUZZ,
        chat_id=CHANNEL,
        chat_type="group",
        thread_id=ROOT_EVT,
        message_id="thread-trigger",
        profile=None,
    )

    top_meta = runner._thread_metadata_for_source(top_source)
    thread_meta = runner._thread_metadata_for_source(thread_source)
    assert top_meta == {
        "reply_to_message_id": "top-trigger",
        "buzz_trigger_placement": "top_level",
    }
    assert thread_meta == {
        "thread_id": ROOT_EVT,
        "reply_to_message_id": "thread-trigger",
        "buzz_trigger_placement": "in_thread",
    }

    cli = _CapturingCli()
    adapter._run_cli = cli
    await adapter.send(CHANNEL, "top progress", metadata=top_meta)
    await adapter.send(CHANNEL, "thread progress", metadata=thread_meta)
    assert [_reply_arg(args) for args, _ in cli.calls] == expected


@pytest.mark.asyncio
async def test_base_final_response_threads_top_level_buzz_attachment(tmp_path):
    """The real Base final-response path gives Buzz files trigger metadata."""
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import build_session_key

    attachment = tmp_path / "report.pdf"
    attachment.write_bytes(b"report")
    adapter = _make_adapter(
        extra={"channel_modes": {CHANNEL: {"replies": "threaded"}}},
        typing_indicator=False,
    )
    cli = _CapturingCli()
    adapter._run_cli = cli

    async def handler(_event):
        return f"MEDIA:{attachment}"

    adapter.set_message_handler(handler)
    source = adapter.build_source(
        chat_id=CHANNEL,
        chat_type="group",
        user_id=OTHER_PUBKEY,
        message_id="top-trigger",
    )
    event = MessageEvent(
        text="make a report",
        message_type=MessageType.TEXT,
        source=source,
        message_id="top-trigger",
    )

    await adapter._process_message_background(event, build_session_key(source))

    attachment_args = next(args for args, _ in cli.calls if "--file" in args)
    assert _reply_arg(attachment_args) == "top-trigger"


@pytest.mark.asyncio
async def test_hybrid_unknown_origin_falls_back_flat_but_explicit_thread_is_supported():
    adapter = _make_adapter(
        extra={"channel_modes": {CHANNEL: {"replies": "hybrid"}}}
    )
    adapter._thread_roots["known-before-eviction"] = ROOT_EVT
    adapter._thread_roots.clear()
    cli = _CapturingCli()
    adapter._run_cli = cli

    await adapter.send(CHANNEL, "unknown", reply_to="known-before-eviction")
    await adapter.send(
        CHANNEL,
        "synthetic thread target",
        metadata={"thread_id": ROOT_EVT},
    )

    assert [_reply_arg(args) for args, _ in cli.calls] == [None, ROOT_EVT]


@pytest.mark.asyncio
async def test_channel_override_does_not_change_direct_message_reply_behavior():
    adapter = _make_adapter(
        extra={"channel_modes": {CHANNEL: {"replies": "flat"}}}
    )
    adapter._channel_state[CHANNEL] = {
        "chat_type": "dm",
        "last_ts": 0,
        "seen": {},
    }
    cli = _CapturingCli()
    adapter._run_cli = cli

    await adapter.send(CHANNEL, "dm reply", reply_to="dm-trigger")

    assert _reply_arg(cli.calls[0][0]) == "dm-trigger"


@pytest.mark.asyncio
async def test_live_reconstructed_channel_isolation_and_reset_inheritance():
    other_channel = "38a45d99-7904-5bab-9a92-9d4e6e671812"
    configured = {
        CHANNEL: {"replies": "hybrid"},
        other_channel: {"replies": "flat"},
    }
    adapter = _make_adapter(
        extra={"channel_modes": {other_channel: {"replies": "flat"}}}
    )
    adapter.apply_channel_policy(CHANNEL, "replies", "hybrid")
    rebuilt = _make_adapter(extra={"channel_modes": configured})
    metadata = {
        "thread_id": ROOT_EVT,
        "reply_to_message_id": "trigger-evt",
        "buzz_trigger_placement": "in_thread",
    }

    for current in (adapter, rebuilt):
        cli = _CapturingCli()
        current._run_cli = cli
        await current.send(CHANNEL, "threaded", metadata=metadata)
        await current.send(other_channel, "flat", metadata=metadata)
        assert [_reply_arg(args) for args, _ in cli.calls] == [ROOT_EVT, None]

    adapter.apply_channel_policy(CHANNEL, "replies", None)
    cli = _CapturingCli()
    adapter._run_cli = cli
    await adapter.send(CHANNEL, "inherited threaded", metadata=metadata)
    assert _reply_arg(cli.calls[0][0]) == ROOT_EVT


@pytest.mark.parametrize(
    ("channel_mode", "global_alias", "expected"),
    [
        ("flat", True, None),
        ("hybrid", False, ROOT_EVT),
        ("threaded", False, ROOT_EVT),
        (None, False, None),
    ],
)
@pytest.mark.asyncio
async def test_standalone_send_reads_persisted_channel_mode_and_global_alias(
    monkeypatch, tmp_path, channel_mode, global_alias, expected
):
    fake_cli = tmp_path / "buzz"
    fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_cli.chmod(0o755)
    captured = {}

    async def fake_exec(
        cli_path, args, *, relay_url, private_key, auth_tag="", input_text=None,
        timeout=None,
    ):
        captured["args"] = args
        return 0, json.dumps({"accepted": True, "event_id": "evt-cron"}), ""

    monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1test")
    modes = {CHANNEL: {"replies": channel_mode}} if channel_mode else {}

    class _PC:
        reply_to_mode = "first"
        extra = {
            "relay_url": "https://test.relay",
            "cli_path": str(fake_cli),
            "reply_in_thread": global_alias,
            "channel_modes": modes,
        }

    result = await _buzz_mod._standalone_send(
        _PC(), CHANNEL, "cron msg", thread_id=ROOT_EVT
    )
    assert result.get("success") is True
    assert _reply_arg(captured["args"]) == expected


# ── 5. Display defaults (#95841) ──────────────────────────────────────


class TestDisplayDefaults:

    def test_buzz_has_platform_defaults_entry(self):
        from gateway.display_config import _PLATFORM_DEFAULTS

        assert "buzz" in _PLATFORM_DEFAULTS

    def test_buzz_does_not_inherit_verbose_global_tool_progress(self):
        from gateway.display_config import resolve_display_setting

        # No user config: must come from the buzz platform tier, not the
        # verbose _GLOBAL_DEFAULTS ("all").
        assert resolve_display_setting({}, "buzz", "tool_progress") != "all"
