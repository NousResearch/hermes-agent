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
        fake_cli.write_text("#!/bin/sh\n")
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
        fake_cli.write_text("#!/bin/sh\n")
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

    def test_buzz_progress_threads_by_default(self):
        from gateway.run import _resolve_progress_thread_id

        assert _resolve_progress_thread_id(
            "buzz", source_thread_id=None, event_message_id="evt-1",
            reply_in_thread=True,
        ) == "evt-1"

    def test_buzz_progress_flat_when_opted_out(self):
        from gateway.run import _resolve_progress_thread_id

        assert _resolve_progress_thread_id(
            "buzz", source_thread_id=None, event_message_id="evt-1",
            reply_in_thread=False,
        ) is None


# ── 4. Display defaults (#95841) ─────────────────────────────────────────


class TestDisplayDefaults:

    def test_buzz_has_platform_defaults_entry(self):
        from gateway.display_config import _PLATFORM_DEFAULTS

        assert "buzz" in _PLATFORM_DEFAULTS

    def test_buzz_does_not_inherit_verbose_global_tool_progress(self):
        from gateway.display_config import resolve_display_setting

        # No user config: must come from the buzz platform tier, not the
        # verbose _GLOBAL_DEFAULTS ("all").
        assert resolve_display_setting({}, "buzz", "tool_progress") != "all"
