"""Tests for the Buzz reaction lifecycle (👀 → 🧠 → ✅/❌).

Behavioral contract (design spec, PM-owned):
- Reaction lifecycle is Buzz-owned and serialized per inbound message.
- 👀 is queued at dispatch of an authorized conversational message, before
  handle_message runs; 🧠 when background processing starts; ✅/❌ on the
  real outcome; CANCELLED removes the working reaction and adds nothing.
- Replacement is remove-gated: a failed remove suppresses the replacement
  add so lifecycle reactions never stack.
- Gate: gateway.platforms.buzz.extra.reactions (default true). No env var.
- Every reaction operation is best-effort: failures never affect the reply.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")

BuzzAdapter = _buzz_mod.BuzzAdapter

SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
SELF_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"
OTHER_PUBKEY = "a" * 64
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"

EYES = "\U0001f440"
BRAIN = "\U0001f9e0"
OK = "\U00002705"
FAIL = "\U0000274c"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Hermetic: no ambient Buzz env vars (the gate must be config-only)."""
    for var in (
        "BUZZ_RELAY_URL",
        "BUZZ_PRIVATE_KEY",
        "BUZZ_REACTIONS",  # must never be consulted
        "BUZZ_CHANNELS",
        "BUZZ_HOME_CHANNEL",
        "BUZZ_ALLOWED_USERS",
        "BUZZ_ALLOW_ALL_USERS",
        "BUZZ_POLL_INTERVAL",
        "BUZZ_AUTH_TAG",
        "BUZZ_CLI_PATH",
        "BUZZ_CREDENTIALS_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path / "no-creds")
    yield


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(
        enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})}
    )
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._self_npub = SELF_NPUB
    adapter._display_name = "Chip"
    adapter._private_key = "nsec1test"
    return adapter


def _message_event(
    adapter,
    text="hello",
    message_id="e1",
    chat_id=CHANNEL,
    chat_type="group",
    user_id=OTHER_PUBKEY,
):
    from gateway.platforms.base import MessageEvent, MessageType

    source = adapter.build_source(
        chat_id=chat_id,
        chat_name=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name="other",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


class _RecordingCli:
    """Fake ``_run_cli`` recording every call as (args, input_text).

    Optional gate/event hooks let tests hold calls open to prove ordering.
    Reaction failures are count-limited so a test can fail the FIRST
    remove/add and let later retries succeed.
    """

    def __init__(self):
        self.calls = []
        self.release_first: asyncio.Event | None = None  # optional gate
        self.fail_removes: dict = {}  # emoji -> remaining failure count
        self.fail_adds: dict = {}  # emoji -> remaining failure count

    def fail_remove(self, emoji, times=1):
        self.fail_removes[emoji] = times

    def fail_add(self, emoji, times=1):
        self.fail_adds[emoji] = times

    async def __call__(self, args, *, input_text=None):
        args = list(args)
        if args[0] == "reactions":
            cmd = args[1]
            emoji = args[args.index("--emoji") + 1]
            fails = self.fail_removes if cmd == "remove" else self.fail_adds
            if fails.get(emoji, 0) > 0:
                fails[emoji] -= 1
                self.calls.append((args, input_text))
                return 2, "", "relay error"
        if self.release_first is not None and not self.calls and args[0] == "reactions":
            await self.release_first.wait()
        self.calls.append((args, input_text))
        return 0, '{"accepted": true, "event_id": "self-sent"}', ""

    def reaction_ops(self):
        """Reaction operations in issue order: ('add'|'remove', emoji)."""
        ops = []
        for args, _input in self.calls:
            if args[0] == "reactions":
                ops.append((args[1], args[args.index("--emoji") + 1]))
        return ops


async def _drain_reactions(adapter, rounds=6):
    """Wait until no reaction tasks remain (terminal hooks enqueue async)."""
    for _ in range(rounds):
        tasks = [t for t in list(adapter._reaction_tasks) if not t.done()]
        if not tasks:
            break
        await asyncio.gather(*tasks, return_exceptions=True)


async def _drain_sessions(adapter, rounds=8):
    """Wait until no background session/reaction tasks remain.

    ``handle_message`` spawns the handler in a background task, so callers
    asserting handler-side effects must drain before asserting.
    """
    for _ in range(rounds):
        session = [
            t
            for t in list(getattr(adapter, "_session_tasks", {}).values())
            if not t.done()
        ]
        pending = [t for t in list(adapter._reaction_tasks) if not t.done()]
        if not session and not pending:
            break
        await asyncio.gather(*(session + pending), return_exceptions=True)


# ── Configuration gate ────────────────────────────────────────────────────


class TestReactionGate:
    def test_default_enabled_when_absent(self):
        assert _make_adapter()._reactions_enabled() is True

    @pytest.mark.parametrize("value", [False, "false", "False", "0", "no", "off"])
    def test_disabled_values(self, value):
        assert _make_adapter(extra={"reactions": value})._reactions_enabled() is False

    @pytest.mark.parametrize("value", [True, "true", "yes", "on", "1"])
    def test_enabled_values(self, value):
        assert _make_adapter(extra={"reactions": value})._reactions_enabled() is True

    def test_invalid_value_falls_back_to_default_true(self):
        assert _make_adapter(extra={"reactions": "banana"})._reactions_enabled() is True

    def test_gate_is_per_adapter_multiplex_safe(self):
        off = _make_adapter(extra={"reactions": False})
        on = _make_adapter(extra={})
        assert off._reactions_enabled() is False and on._reactions_enabled() is True


# ── Low-level CLI primitives ──────────────────────────────────────────────


class TestReactionPrimitives:
    @pytest.mark.asyncio
    async def test_add_uses_event_and_emoji_no_channel(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        adapter._run_cli = cli
        assert await adapter.send_reaction(CHANNEL, "e1", EYES) is True
        assert cli.calls == [
            (["reactions", "add", "--event", "e1", "--emoji", EYES], None)
        ]

    @pytest.mark.asyncio
    async def test_remove_uses_event_and_emoji_no_channel(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        adapter._run_cli = cli
        assert await adapter.remove_reaction(CHANNEL, "e1", EYES) is True
        assert cli.calls == [
            (["reactions", "remove", "--event", "e1", "--emoji", EYES], None)
        ]

    @pytest.mark.asyncio
    async def test_remove_failure_returns_false(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        cli.fail_remove(EYES)
        adapter._run_cli = cli
        assert await adapter.remove_reaction(CHANNEL, "e1", EYES) is False


# ── Transition coordinator (serialization + replacement algorithm) ───────


class TestTransitionCoordinator:
    def _seed(self, adapter, cli, message_id="e1", chat_id=CHANNEL):
        adapter._run_cli = cli
        adapter._reaction_begin(chat_id, message_id)
        return (chat_id, message_id)

    @pytest.mark.asyncio
    async def test_success_sequence_ordered(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        key = self._seed(adapter, cli)
        await adapter.on_processing_start(_evt_with(key))
        await adapter.on_processing_complete(_evt_with(key), _outcome_success())
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == [
            ("add", EYES),
            ("remove", EYES),
            ("add", BRAIN),
            ("remove", BRAIN),
            ("add", OK),
        ]
        # Terminal state is cleaned up.
        assert key not in adapter._reaction_lifecycle

    @pytest.mark.asyncio
    async def test_failure_sequence_ordered(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        key = self._seed(adapter, cli)
        await adapter.on_processing_start(_evt_with(key))
        await adapter.on_processing_complete(_evt_with(key), _outcome_failure())
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == [
            ("add", EYES),
            ("remove", EYES),
            ("add", BRAIN),
            ("remove", BRAIN),
            ("add", FAIL),
        ]

    @pytest.mark.asyncio
    async def test_cancelled_removes_without_terminal(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        key = self._seed(adapter, cli)
        await adapter.on_processing_start(_evt_with(key))
        await adapter.on_processing_complete(_evt_with(key), _outcome_cancelled())
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == [
            ("add", EYES),
            ("remove", EYES),
            ("add", BRAIN),
            ("remove", BRAIN),
        ]
        assert key not in adapter._reaction_lifecycle

    @pytest.mark.asyncio
    async def test_failed_remove_suppresses_replacement_add(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        cli.fail_remove(EYES)  # first remove fails; later removes succeed
        key = self._seed(adapter, cli)
        await adapter.on_processing_start(_evt_with(key))
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == [("add", EYES), ("remove", EYES)]
        # Old reaction stays authoritative; a later terminal still tries.
        await adapter.on_processing_complete(_evt_with(key), _outcome_failure())
        await _drain_reactions(adapter)
        assert cli.reaction_ops()[-2:] == [("remove", EYES), ("add", FAIL)]

    @pytest.mark.asyncio
    async def test_initial_add_failure_then_direct_brain(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        cli.fail_add(EYES)
        key = self._seed(adapter, cli)
        await adapter.on_processing_start(_evt_with(key))
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == [("add", EYES), ("add", BRAIN)]

    @pytest.mark.asyncio
    async def test_transitions_serialize_never_stack(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        cli.release_first = asyncio.Event()
        key = self._seed(adapter, cli)

        start_task = asyncio.create_task(adapter.on_processing_start(_evt_with(key)))
        # Working + terminal are queued while the 👀 add is still held open.
        await asyncio.sleep(0)
        await adapter.on_processing_complete(_evt_with(key), _outcome_failure())
        cli.release_first.set()
        await asyncio.gather(start_task, return_exceptions=True)
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == [
            ("add", EYES),
            ("remove", EYES),
            ("add", BRAIN),
            ("remove", BRAIN),
            ("add", FAIL),
        ]

    @pytest.mark.asyncio
    async def test_hooks_noop_for_unknown_event(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        adapter._run_cli = cli
        await adapter.on_processing_start(_evt_with((CHANNEL, "unknown")))
        await adapter.on_processing_complete(
            _evt_with((CHANNEL, "unknown")), _outcome_success()
        )
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == []

    @pytest.mark.asyncio
    async def test_hooks_noop_when_disabled(self):
        adapter = _make_adapter(extra={"reactions": False})
        cli = _RecordingCli()
        adapter._run_cli = cli
        key = (CHANNEL, "e1")
        await adapter.on_processing_start(_evt_with(key))
        await adapter.on_processing_complete(_evt_with(key), _outcome_success())
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == []

    @pytest.mark.asyncio
    async def test_hooks_noop_without_ids(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        adapter._run_cli = cli
        event = _message_event(adapter)
        event.message_id = None
        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, _outcome_success())
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == []

    @pytest.mark.asyncio
    async def test_unauthorized_sender_gets_no_lifecycle(self):
        adapter = _make_adapter()
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: False)
        cli = _RecordingCli()
        adapter._run_cli = cli
        await adapter._dispatch_message(
            text="hello",
            chat_id=CHANNEL,
            chat_type="group",
            user_id=OTHER_PUBKEY,
            user_name="other",
            message_id="e1",
            created_at=1000,
        )
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == []

    @pytest.mark.asyncio
    async def test_no_auth_callback_still_gets_lifecycle(self):
        adapter = _make_adapter()  # no authorization callback installed
        cli = _RecordingCli()
        adapter._run_cli = cli
        seen_at_handler = {}

        async def handler(event):
            seen_at_handler["called"] = True
            seen_at_handler["tracked"] = (CHANNEL, "e1") in adapter._reaction_lifecycle
            return "ok"

        adapter._message_handler = handler
        await adapter._dispatch_message(
            text="hello",
            chat_id=CHANNEL,
            chat_type="group",
            user_id=OTHER_PUBKEY,
            user_name="other",
            message_id="e1",
            created_at=1000,
        )
        # The 👀 must already be lifecycle-tracked before the handler ran
        # (asserted inside the handler: terminal cleanup pops the state by
        # the time the turn finishes).
        await _drain_sessions(adapter)
        assert seen_at_handler.get("called") is True
        assert seen_at_handler["tracked"] is True
        assert ("add", EYES) in cli.reaction_ops()


# ── Dispatch wiring ───────────────────────────────────────────────────────


class TestDispatchWiring:
    @pytest.mark.asyncio
    async def test_command_message_gets_no_reaction(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        adapter._run_cli = cli
        adapter._message_handler = AsyncMock(return_value="ok")
        await adapter._dispatch_message(
            text="/status",
            chat_id=CHANNEL,
            chat_type="group",
            user_id=OTHER_PUBKEY,
            user_name="other",
            message_id="e1",
            created_at=1000,
        )
        await _drain_reactions(adapter)
        assert cli.reaction_ops() == []

    @pytest.mark.asyncio
    async def test_dispatch_queues_eyes_before_handler_and_no_duplicate(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        adapter._run_cli = cli

        seen_at_handler = {}

        async def handler(event):
            seen_at_handler["tracked"] = (CHANNEL, "e1") in adapter._reaction_lifecycle
            return "ok"

        adapter._message_handler = handler
        await adapter._dispatch_message(
            text="hello",
            chat_id=CHANNEL,
            chat_type="group",
            user_id=OTHER_PUBKEY,
            user_name="other",
            message_id="e1",
            created_at=1000,
        )
        await _drain_sessions(adapter)
        assert seen_at_handler["tracked"] is True
        # Exactly one 👀 add: the lifecycle one — the old post-dispatch
        # duplicate must be gone.
        assert cli.reaction_ops().count(("add", EYES)) == 1


# ── Full base-class wiring (integration through handle_message) ──────────


class TestBaseHookIntegration:
    async def _run_turn(self, adapter, handler, text="hello"):
        adapter._message_handler = handler
        await adapter._dispatch_message(
            text=text,
            chat_id=CHANNEL,
            chat_type="group",
            user_id=OTHER_PUBKEY,
            user_name="other",
            message_id="e1",
            created_at=1000,
        )
        for _ in range(8):
            session = [
                t
                for t in list(getattr(adapter, "_session_tasks", {}).values())
                if not t.done()
            ]
            pending = [t for t in list(adapter._reaction_tasks) if not t.done()]
            if not session and not pending:
                break
            await asyncio.gather(*(session + pending), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_real_flow_success_full_lifecycle(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        adapter._run_cli = cli
        await self._run_turn(adapter, AsyncMock(return_value="ok"))
        assert cli.reaction_ops() == [
            ("add", EYES),
            ("remove", EYES),
            ("add", BRAIN),
            ("remove", BRAIN),
            ("add", OK),
        ]

    @pytest.mark.asyncio
    async def test_real_flow_handler_exception_marks_failure(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        adapter._run_cli = cli

        async def boom(event):
            raise RuntimeError("handler exploded")

        await self._run_turn(adapter, boom)
        assert cli.reaction_ops()[-1] == ("add", FAIL)
        assert ("remove", BRAIN) in cli.reaction_ops()


# ── Disconnect hygiene ────────────────────────────────────────────────────


class TestAbandonedLifecycleCleanup:
    @pytest.mark.asyncio
    async def test_stale_completed_chain_is_removed_without_terminal_hook(self):
        adapter = _make_adapter()
        adapter._reaction_cleanup_ttl = 0
        key = (CHANNEL, "leaked")
        adapter._reaction_begin(*key)
        await _drain_reactions(adapter)

        assert key in adapter._reaction_lifecycle
        await adapter._reaction_cleanup_once()

        assert key not in adapter._reaction_lifecycle

    @pytest.mark.asyncio
    async def test_cleanup_keeps_live_transition_until_timeout_cancels_it(self):
        adapter = _make_adapter()
        adapter._reaction_cleanup_ttl = 0
        key = (CHANNEL, "in-flight")
        blocker = asyncio.Event()

        async def stuck(*_args, **_kwargs):
            await blocker.wait()
            return 0, "", ""

        adapter._run_cli = stuck
        adapter._reaction_begin(*key)
        await asyncio.sleep(0)
        task = adapter._reaction_lifecycle[key]["tail_task"]
        assert not task.done()

        await adapter._reaction_cleanup_once()
        # Cancellation is delivered on the next loop pass; wait for the
        # task to finish processing it.
        await asyncio.gather(task, return_exceptions=True)

        assert task.cancelled()
        assert key not in adapter._reaction_lifecycle

    @pytest.mark.asyncio
    async def test_cleanup_does_not_touch_young_in_flight_state(self):
        """A valid transition in flight must survive a sweep."""
        adapter = _make_adapter()
        adapter._reaction_cleanup_ttl = 300.0
        key = (CHANNEL, "young")
        blocker = asyncio.Event()

        async def slow_but_fine(*_args, **_kwargs):
            await blocker.wait()
            return 0, "", ""

        adapter._run_cli = slow_but_fine
        adapter._reaction_begin(*key)
        await asyncio.sleep(0)
        task = adapter._reaction_lifecycle[key]["tail_task"]

        await adapter._reaction_cleanup_once()

        assert key in adapter._reaction_lifecycle
        assert not task.cancelled()
        blocker.set()
        await _drain_reactions(adapter)

    @pytest.mark.asyncio
    async def test_single_shared_cleanup_task_starts_and_ends_with_state(self):
        adapter = _make_adapter()
        adapter._reaction_cleanup_interval = 0.001

        key = (CHANNEL, "sweeper")
        adapter._reaction_begin(*key)
        first = adapter._reaction_cleanup_task
        assert first is not None and not first.done()

        # A second entry reuses the same sweeper task.
        key2 = (CHANNEL, "sweeper2")
        adapter._reaction_begin(*key2)
        assert adapter._reaction_cleanup_task is first

        # Terminal completion empties the map; the sweeper exits on its own.
        await adapter.on_processing_complete(_evt_with(key2), _outcome_success())
        await adapter.on_processing_complete(_evt_with(key), _outcome_success())
        await _drain_reactions(adapter)
        for _ in range(20):
            if adapter._reaction_cleanup_task.done():
                break
            await asyncio.sleep(0.001)
        assert adapter._reaction_cleanup_task.done()
        # A fresh entry restarts it (done task is replaced).
        adapter._reaction_begin(CHANNEL, "sweeper3")
        assert adapter._reaction_cleanup_task is not first
        await adapter.on_processing_complete(_evt_with((CHANNEL, "sweeper3")), _outcome_success())
        await _drain_reactions(adapter)


class TestDisconnectCleanup:
    @pytest.mark.asyncio
    async def test_disconnect_cancels_in_flight_reaction_tasks(self):
        adapter = _make_adapter()
        cli = _RecordingCli()
        cli.release_first = asyncio.Event()
        adapter._run_cli = cli
        key = (CHANNEL, "e1")
        adapter._reaction_begin(*key)
        await adapter.on_processing_start(_evt_with(key))
        await asyncio.sleep(0)  # let the task start and block on the CLI

        await asyncio.wait_for(adapter.disconnect(), timeout=5)

        assert not adapter._reaction_tasks or all(
            t.done() for t in adapter._reaction_tasks
        )
        assert adapter._reaction_lifecycle == {}


# ── helpers ───────────────────────────────────────────────────────────────


def _evt_with(key):
    """Minimal stand-in carrying (chat_id, message_id) for hook calls."""

    class _Src:
        chat_id = key[0]

    class _Evt:
        source = _Src()
        message_id = key[1]

    return _Evt()


def _outcome_success():
    from gateway.platforms.base import ProcessingOutcome

    return ProcessingOutcome.SUCCESS


def _outcome_failure():
    from gateway.platforms.base import ProcessingOutcome

    return ProcessingOutcome.FAILURE


def _outcome_cancelled():
    from gateway.platforms.base import ProcessingOutcome

    return ProcessingOutcome.CANCELLED
