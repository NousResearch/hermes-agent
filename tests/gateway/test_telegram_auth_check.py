"""Tests for Telegram adapter early authorization check.

Verifies that unauthorized users are blocked before any text batching,
event building, or response generation occurs.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType


def _make_adapter(allow_from=None, allowed_chats=None, group_allowed_chats=None, callback_auth=None, **extra_overrides):
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
    except ModuleNotFoundError:  # PR branch before Telegram plugin extraction
        from gateway.platforms.telegram import TelegramAdapter

    extra = {}
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if allowed_chats is not None:
        extra["allowed_chats"] = allowed_chats
    if group_allowed_chats is not None:
        extra["group_allowed_chats"] = group_allowed_chats
    extra.update(extra_overrides)

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    if callback_auth is not None:
        adapter._is_callback_user_authorized = callback_auth
    return adapter


def _make_message(text="hello", *, from_user_id=111, chat_id=-100, chat_type="group"):
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="Test", is_forum=False),
        from_user=SimpleNamespace(id=from_user_id, full_name="Test User", first_name="Test"),
        reply_to_message=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )


@pytest.mark.asyncio
async def test_unauthorized_user_blocked_before_event_building():
    """Unauthorized user's message should be blocked before _build_message_event."""
    adapter = _make_adapter(group_allow_from=["222"])  # Only user 222 allowed in groups

    build_called = False
    original_build = adapter._build_message_event

    def track_build(*a, **kw):
        nonlocal build_called
        build_called = True
        return original_build(*a, **kw)

    adapter._build_message_event = track_build

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(from_user_id=111, chat_type="group"),  # User 111 NOT in group_allow_from
        effective_message=None,
    )

    await adapter._handle_text_message(update, SimpleNamespace())

    assert build_called is False, "build_message_event should not be called for unauthorized user"


@pytest.mark.asyncio
async def test_command_from_unauthorized_user_blocked():
    """Commands from unauthorized users should be blocked."""
    adapter = _make_adapter(group_allow_from=["222"])
    adapter.handle_message = AsyncMock()

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(text="/start", from_user_id=111, chat_type="group"),
        effective_message=None,
    )

    await adapter._handle_command(update, SimpleNamespace())

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_location_from_unauthorized_user_blocked():
    """Location messages from unauthorized users should be blocked."""
    adapter = _make_adapter(group_allow_from=["222"])

    msg = _make_message(from_user_id=111, chat_type="group")
    msg.text = None
    msg.location = SimpleNamespace(latitude=53.3498, longitude=-6.2603)

    update = SimpleNamespace(
        update_id=1,
        message=msg,
        effective_message=None,
    )

    # Should not raise — just silently return
    await adapter._handle_location_message(update, SimpleNamespace())


def test_is_user_authorized_from_message_allow_from():
    """_is_user_authorized_from_message should respect adapter-level allow_from for DMs."""
    adapter = _make_adapter(allow_from=["111", "222"])

    msg = _make_message(from_user_id=111, chat_type="dm")
    assert adapter._is_user_authorized_from_message(msg) is True

    msg = _make_message(from_user_id=333, chat_type="dm")
    assert adapter._is_user_authorized_from_message(msg) is False


def test_allowlist_dm_with_explicit_pair_behavior_reaches_gateway(monkeypatch):
    """Allowlist + unauthorized_dm_behavior:pair must not early-drop unknown DMs.

    Regression for the gap left by #40863: early intake rejection discarded
    unauthorized DMs before gateway pairing could run, even when the operator
    explicitly set telegram.unauthorized_dm_behavior: pair.
    """
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    class Runner:
        def _is_user_authorized(self, source):
            return source.user_id == "111"

        def _get_unauthorized_dm_behavior(self, platform, *, profile=None):
            assert platform == Platform.TELEGRAM
            return "pair"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=999, chat_id=999, chat_type="private")

    assert adapter._is_user_authorized_from_message(msg) is True


def test_allowlist_dm_without_pair_behavior_still_early_rejects(monkeypatch):
    """Allowlist without pairing opt-in keeps the #9337/#40863 silent drop."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    class Runner:
        def _is_user_authorized(self, source):
            return source.user_id == "111"

        def _get_unauthorized_dm_behavior(self, platform, *, profile=None):
            return "ignore"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=999, chat_id=999, chat_type="private")

    assert adapter._is_user_authorized_from_message(msg) is False


def test_allow_from_dm_with_pair_override_reaches_gateway():
    """Adapter allow_from + unauthorized_dm_behavior:pair still forwards DMs."""
    adapter = _make_adapter(
        allow_from=["111"],
        unauthorized_dm_behavior="pair",
    )
    msg = _make_message(from_user_id=999, chat_id=999, chat_type="dm")
    assert adapter._is_user_authorized_from_message(msg) is True


def test_allowlist_group_with_pair_behavior_still_early_rejects(monkeypatch):
    """Pairing is DM-only — unauthorized group senders stay blocked early."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    class Runner:
        def _is_user_authorized(self, source):
            return source.user_id == "111"

        def _get_unauthorized_dm_behavior(self, platform, *, profile=None):
            return "pair"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter(group_allow_from=["111"])
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=999, chat_id=-100, chat_type="group")

    assert adapter._is_user_authorized_from_message(msg) is False


@pytest.mark.asyncio
async def test_unauthorized_dm_with_pair_behavior_builds_event(monkeypatch):
    """Unknown DM under pair behavior must reach event construction."""
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    class Runner:
        def _is_user_authorized(self, source):
            return source.user_id == "111"

        def _get_unauthorized_dm_behavior(self, platform, *, profile=None):
            return "pair"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    build_called = False
    original_build = adapter._build_message_event

    def track_build(*a, **kw):
        nonlocal build_called
        build_called = True
        return original_build(*a, **kw)

    adapter._build_message_event = track_build
    adapter._enqueue_text_event = lambda event: None
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    adapter._clean_bot_trigger_text = lambda text: text
    adapter._should_process_message = lambda *a, **kw: True

    update = SimpleNamespace(
        update_id=1,
        message=_make_message(from_user_id=999, chat_id=999, chat_type="private"),
        effective_message=None,
    )
    await adapter._handle_text_message(update, SimpleNamespace())
    assert build_called is True


def test_is_user_authorized_from_message_wildcard():
    """_is_user_authorized_from_message should accept wildcard '*'."""
    adapter = _make_adapter(allow_from=["*"])

    msg = _make_message(from_user_id=999)
    assert adapter._is_user_authorized_from_message(msg) is True


def test_is_user_authorized_from_message_no_from_user():
    """_is_user_authorized_from_message should return True for messages without from_user."""
    adapter = _make_adapter(allow_from=["111"])

    msg = _make_message()
    msg.from_user = None
    assert adapter._is_user_authorized_from_message(msg) is True


def test_is_user_authorized_from_message_callback():
    """_is_user_authorized_from_message should use _is_callback_user_authorized."""
    adapter = _make_adapter(callback_auth=lambda uid, **_kw: uid == "555")

    msg = _make_message(from_user_id=555)
    assert adapter._is_user_authorized_from_message(msg) is True

    msg = _make_message(from_user_id=666)
    assert adapter._is_user_authorized_from_message(msg) is False


def test_unknown_dm_with_no_allowlist_passes_to_pairing(monkeypatch):
    """Unknown DMs must still reach the gateway pairing flow when no allowlist exists."""
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = _make_adapter()
    msg = _make_message(from_user_id=111, chat_id=111, chat_type="private")

    assert adapter._is_user_authorized_from_message(msg) is True


_ALL_AUTH_ENV_KEYS = (
    "TELEGRAM_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_CHATS",
    "TELEGRAM_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
)


def test_allow_all_users_false_is_not_configured(monkeypatch):
    """A literal ``false`` ALLOW_ALL toggle must not count as configured auth (#68794).

    ``.env`` files and config templates routinely persist an explicit
    ``GATEWAY_ALLOW_ALL_USERS=false``. An unset key and a ``false`` key are
    semantically identical everywhere else, so the intake prefilter must treat
    both as "no auth configured" and let unknown DMs reach pairing.
    """
    for key in _ALL_AUTH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    adapter = _make_adapter()

    # Only an explicit falsey ALLOW_ALL toggle is set → not configured.
    for key in ("GATEWAY_ALLOW_ALL_USERS", "TELEGRAM_ALLOW_ALL_USERS"):
        for val in ("false", "False", "0", "no", ""):
            monkeypatch.setenv(key, val)
            assert adapter._telegram_auth_env_configured() is False, (key, val)
        monkeypatch.delenv(key, raising=False)

    # A truthy ALLOW_ALL toggle → configured.
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    assert adapter._telegram_auth_env_configured() is True
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    # A real allowlist value (identities/chats) still counts as configured.
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "222")
    assert adapter._telegram_auth_env_configured() is True


def test_unknown_dm_reaches_pairing_when_allow_all_is_false(monkeypatch):
    """End-to-end: with only ``GATEWAY_ALLOW_ALL_USERS=false`` set, an unknown DM
    must fall through to pairing instead of being rejected by the runner's
    allowlist check (#68794)."""
    for key in _ALL_AUTH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")

    rejected = []

    class Runner:
        def _is_user_authorized(self, source):
            # If the prefilter mis-detects `false` as configured, it consults
            # this and drops the unknown DM before pairing.
            rejected.append(source.user_id)
            return False

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=111, chat_id=111, chat_type="private")

    assert adapter._is_user_authorized_from_message(msg) is True
    assert rejected == []  # runner allowlist was never consulted


def test_runner_auth_gets_group_user_allowlist_context(monkeypatch):
    """Group user allowlists need a group-shaped source, not a DM-shaped one."""
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "111")
    seen_sources = []

    class Runner:
        def _is_user_authorized(self, source):
            seen_sources.append(source)
            return source.chat_type == "group" and source.chat_id == "-100" and source.user_id == "111"

        async def handle(self, event):
            return None

    runner = Runner()
    adapter = _make_adapter()
    adapter._message_handler = runner.handle
    msg = _make_message(from_user_id=111, chat_id=-100, chat_type="group")

    assert adapter._is_user_authorized_from_message(msg) is True
    assert seen_sources
    assert seen_sources[0].chat_type == "group"
    assert seen_sources[0].chat_id == "-100"


@pytest.mark.asyncio
async def test_unmentioned_group_location_from_removed_user_not_observed():
    """Removed users must not persist unmentioned group locations into observed context."""
    adapter = _make_adapter(
        group_allow_from=["222"],
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
        require_mention=True,
        observe_unmentioned_group_messages=True,
    )
    observed = []
    adapter._observe_unmentioned_group_message = lambda *args, **kwargs: observed.append((args, kwargs))

    msg = _make_message(text=None, from_user_id=111, chat_id=-100, chat_type="group")
    msg.location = SimpleNamespace(latitude=53.3498, longitude=-6.2603)
    update = SimpleNamespace(update_id=1, message=msg, effective_message=None)

    await adapter._handle_location_message(update, SimpleNamespace())

    assert observed == []


@pytest.mark.asyncio
async def test_allow_all_false_unknown_dm_reaches_real_runner_pairing(monkeypatch, tmp_path):
    """Real-runner regression: with only ``GATEWAY_ALLOW_ALL_USERS=false`` set, an
    unknown Telegram DM must reach ``GatewayRunner``'s unauthorized-DM branch and
    actually generate + deliver a pairing code (#68794).

    The focused adapter-prefilter tests above prove the intake filter treats a
    literal ``false`` toggle as "not configured" and lets the DM through. This
    exercises the downstream path the prefilter hands off to: the runner's own
    ``_is_user_authorized`` must likewise reject the unknown user under a falsey
    global toggle, so ``_handle_message`` falls into the pairing branch that
    calls ``pairing_store.generate_code`` and sends the code via the adapter.
    """
    from gateway.config import GatewayConfig
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    import gateway.run as gateway_run

    for key in _ALL_AUTH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Only an explicit falsey global toggle — no allowlist of any kind.
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter

    generated = []
    original_generate = runner.pairing_store.generate_code

    def tracking_generate(*args, **kwargs):
        code = original_generate(*args, **kwargs)
        generated.append((args, code))
        return code

    runner.pairing_store.generate_code = tracking_generate

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="111",
        chat_type="dm",
        user_id="111",
        user_name="stranger",
    )
    event = MessageEvent(text="hi", source=source, internal=False)

    await runner._handle_message(event)

    # The unknown DM must have driven the real pairing branch, not been dropped.
    assert generated, "unknown DM under GATEWAY_ALLOW_ALL_USERS=false should generate a pairing code"
    assert adapter.send.await_count == 1
    sent_text = adapter.send.await_args.args[1]
    assert "pairing code" in sent_text.lower()
