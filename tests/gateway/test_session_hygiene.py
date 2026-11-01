"""Tests for gateway session hygiene — auto-compression of large sessions.

Verifies that the gateway detects pathologically large transcripts and
triggers auto-compression before running the agent.  (#628)

The hygiene system uses the SAME compression config as the agent:
  compression.threshold × model context length
so CLI and messaging platforms behave identically.
"""

import asyncio
import importlib
import sys
import threading
import time
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(n_messages: int, content_size: int = 100) -> list:
    """Build a fake transcript with n_messages user/assistant pairs."""
    history = []
    content = "x" * content_size
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": content, "timestamp": f"t{i}"})
    return history


class HygieneCaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="hygiene-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


# ---------------------------------------------------------------------------
# End-to-end hygiene through GatewayRunner._handle_message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_hygiene_preserves_transcript_when_no_rotation(monkeypatch, tmp_path):
    """Regression for #21301: the hygiene agent is built without a session_db,
    so _compress_context cannot rotate. When it neither rotates NOR compacts
    in place, the transcript MUST be preserved — an unconditional
    rewrite_transcript() would replace the original messages with only the
    summary (permanent data loss). Mirrors the /compress guard (#44794)."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class NonRotatingCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id", "fake-session")
            self.compression_in_place = False  # not in-place either
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            self.compress_task_id = None
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            # Capture the task scope the gateway forwards (#98206): the dedup
            # reset inside compress_context is only effective under the live
            # session row id, the same task_id the main turn hands to
            # run_conversation.
            self.compress_task_id = _kwargs.get("task_id")
            # No session_db → cannot rotate: session_id is UNCHANGED, and this
            # is a failure-to-rotate, not an in-place success.
            return ([{"role": "assistant", "content": "summary only"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = NonRotatingCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:17585",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = _make_history(6, content_size=400)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100,
    )
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "795544298")

    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            thread_id="17585",
            user_id="12345",
        ),
        message_id="1",
    )

    # Pre-load a failure streak so we can prove the recovery gate is WIRED UP,
    # not merely that the predicate is correct in isolation (#79624). Deleting
    # the whole `if not _hyg_aborted: if hygiene_compaction_recovered(...)`
    # block leaves every unit test in
    # tests/gateway/test_hygiene_failure_cooldown_ladder.py green, so this E2E
    # is the only thing binding the call site.
    reset_calls = []
    _real_reset = gateway_run._reset_hygiene_failure_streak
    monkeypatch.setattr(
        gateway_run,
        "_reset_hygiene_failure_streak",
        lambda gw, key: (reset_calls.append(key), _real_reset(gw, key))[1],
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    # The transcript must NOT be rewritten — the original is preserved.
    runner.session_store.rewrite_transcript.assert_not_called()
    # #98206: hygiene compaction must scope the dedup reset to the LIVE
    # session row id ("sess-1" from this test's SessionEntry) — the same
    # task_id the main turn passes to run_conversation. Under the "default"
    # fallback the skill_view/read_file dedup records survive compression
    # and a re-read returns a stub pointing at pruned context.
    assert NonRotatingCompressAgent.last_instance.compress_task_id == "sess-1"

    # This run neither rotated nor compacted in place, so it did NOT recover
    # the session: the reset must NOT have been reached. Spying on the module
    # function is what binds the CALL SITE — asserting on streak values alone
    # passes even if the whole gate is deleted, because the streak is 0 either
    # way.
    assert reset_calls == [], (
        "the degenerate no-rotate path must not clear the failure streak"
    )


@pytest.mark.asyncio
async def test_session_hygiene_preserves_transcript_when_in_place_configured_but_no_db(monkeypatch, tmp_path):
    """Regression: when compression.in_place is True but the hygiene agent has
    no session_db, archive_and_compact cannot run — _last_compaction_in_place
    stays False.  The guard must read the *result* flag, not the *config* flag,
    otherwise the transcript is unconditionally rewritten with only the summary
    (permanent data loss identical to #21301)."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class InPlaceConfiguredAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id", "fake-session")
            self.compression_in_place = True
            self._last_compaction_in_place = False
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            return ([{"role": "assistant", "content": "summary only"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = InPlaceConfiguredAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:17585",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = _make_history(6, content_size=400)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100,
    )
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "795544298")

    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            thread_id="17585",
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    # The config says in_place=True, but the DB write failed (no session_db)
    # so _last_compaction_in_place is False. Transcript must NOT be rewritten.
    runner.session_store.rewrite_transcript.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("warning_notifications", [True, False])
async def test_session_hygiene_timeout_continues_to_agent_and_sets_cooldown(monkeypatch, tmp_path, warning_notifications):
    """A timed-out SessionDB-bound worker cannot compact after the live turn starts.

    The worker remains alive long enough to cross the old race window. The
    timeout must fence its eventual commit, continue to the live agent, and
    clean up the temporary agent only after the worker actually returns.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    worker_started = threading.Event()
    release_worker = threading.Event()
    lease_released = threading.Event()
    cleanup_done = threading.Event()
    fake_db = MagicMock()
    # The DB-backed cooldown check calls this before compressing; a bare
    # MagicMock return would be truthy and skip compression entirely.
    fake_db.get_compression_failure_cooldown.return_value = None

    class SlowCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id", "fake-session")
            self._session_db = kwargs.get("session_db")
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
            )
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock(side_effect=cleanup_done.set)
            type(self).last_instance = self

        def _compress_context(
            self, messages, *_args, commit_fence=None, **_kwargs
        ):
            if commit_fence is not None:
                commit_fence.register_cancelled_lock_release(lease_released.set)
            worker_started.set()
            assert release_worker.wait(timeout=10)
            if commit_fence is not None and not commit_fence.begin_commit():
                return (messages, None)
            try:
                self._session_db.archive_and_compact(
                    self.session_id,
                    [{"role": "assistant", "content": "too late"}],
                )
                self._last_compaction_in_place = True
                return ([{"role": "assistant", "content": "too late"}], None)
            finally:
                if commit_fence is not None:
                    commit_fence.finish_commit()

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = SlowCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_timeout_seconds: 0.01\n"
        "  hygiene_failure_cooldown_seconds: 120\n"
        f"display: {{suppress_warning_notifications: {str(not warning_notifications).lower()}}}\n"
    )

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:12345",
        session_id="sess-timeout",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = _make_history(6, content_size=400)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = SimpleNamespace(_db=fake_db)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100,
    )

    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    assert worker_started.is_set()
    assert runner._run_agent.await_count == 1
    # Cooldown must be persisted to the state DB (survives restart, #74136),
    # not stashed in an in-memory dict.
    assert fake_db.record_compression_failure_cooldown.called
    _cd_args = fake_db.record_compression_failure_cooldown.call_args[0]
    assert _cd_args[0] == "sess-timeout"
    assert _cd_args[1] > time.time()
    timeout_warnings = [s for s in adapter.sent if "took too long" in s["content"]]
    assert len(timeout_warnings) == int(warning_notifications)
    fake_db.archive_and_compact.assert_not_called()
    assert lease_released.is_set()
    # Event/state assertions prove the host returned before the detached
    # worker's event-gated wait completed without a scheduler-sensitive clock
    # bound: cleanup runs only when that worker actually exits.
    SlowCompressAgent.last_instance.close.assert_not_called()

    release_worker.set()
    await asyncio.wait_for(asyncio.to_thread(cleanup_done.wait), timeout=2)

    # The late worker observed cancellation at the commit fence, so it never
    # mutated the live session after the new turn began. Cleanup still ran once
    # it was safe to tear down the helper agent's clients/providers.
    fake_db.archive_and_compact.assert_not_called()
    SlowCompressAgent.last_instance.close.assert_called_once()


    # Behavior witness 3: the #87011 contract remains truthful —
    # "session hygiene compression timed out" still means a real idle
    # timeout, not a turn-hold deferral. The turn-hold path must use a
    # distinct provenance stamp.
    # (Verified indirectly: the idle-timeout path would have sent the
    # idle-timeout message, which we already asserted absent above.)


@pytest.mark.asyncio
async def test_session_hygiene_forces_in_place_compaction_with_bound_session_db(
    monkeypatch, tmp_path
):
    """Regression for #60947: gateway hygiene should not rely on
    helper-agent session rotation to shrink a live gateway transcript.

    The hygiene pass runs before the user turn and already owns the gateway
    session binding, so it should force in-place compaction and bind the
    compressor to the gateway SessionDB. Otherwise a helper can return a
    summary without rotating/compacting, the guard preserves the original
    transcript, and the same oversized session is reloaded on every turn.
    """
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    stored_system_prompt = (
        "You are Hermes.\n\n"
        "<memory_provider_context>\n"
        "Pinboard provider instructions\n"
        "</memory_provider_context>"
    )
    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown.return_value = None
    async_session_db = SimpleNamespace(
        _db=fake_db,
        get_session=AsyncMock(
            return_value={
                "system_prompt": stored_system_prompt,
            }
        ),
    )

    class FakeInPlaceCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.platform = kwargs.get("platform")
            self.session_id = kwargs.get("session_id", "fake-session")
            self._session_db = kwargs.get("session_db")
            self._cached_system_prompt = None
            self.compression_in_place = False
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
            )
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            assert self.compression_in_place is True
            assert self._session_db is fake_db
            assert self.platform == "gateway_hygiene"
            assert self._cached_system_prompt == stored_system_prompt
            self._last_compaction_in_place = True
            return ([{"role": "assistant", "content": "compressed in place"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeInPlaceCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:private:12345",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="private",
    )
    runner.session_store.load_transcript.return_value = _make_history(12, content_size=400)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = async_session_db
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100,
    )

    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="12345",
        ),
        message_id="1",
    )

    # Spy on the recovery reset so this test binds the CALL SITE (#79624).
    # Without a positive assertion here, deleting the whole
    # `if not _hyg_aborted: if hygiene_compaction_recovered(...)` block leaves
    # every other hygiene and ladder test green.
    reset_calls = []
    _real_reset = gateway_run._reset_hygiene_failure_streak
    monkeypatch.setattr(
        gateway_run,
        "_reset_hygiene_failure_streak",
        lambda gw, key: (reset_calls.append(key), _real_reset(gw, key))[1],
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    agent = FakeInPlaceCompressAgent.last_instance
    assert agent is not None
    async_session_db.get_session.assert_awaited_once_with("sess-1")
    agent.context_compressor.bind_session_state.assert_called_once_with(fake_db, "sess-1")
    # In-place compaction already persisted via archive_and_compact() —
    # rewrite_transcript would replace_messages(active_only=False) and DELETE
    # the just-archived rows (#61145). The hygiene handler must skip it.
    runner.session_store.rewrite_transcript.assert_not_called()
    runner._run_agent.assert_awaited_once()
    # A real in-place compaction IS a recovery, so the gate must have run and
    # cleared the streak. This is the positive half of the wiring contract.
    assert reset_calls, (
        "successful in-place compaction must clear the hygiene failure streak "
        "— the recovery gate is not wired into _handle_message_with_agent"
    )


@pytest.mark.asyncio
async def test_session_hygiene_honors_configurable_hard_message_limit(
    monkeypatch, tmp_path
):
    """compression.hygiene_hard_message_limit overrides the default.

    Regression for user-reported fix: a gateway session with a small
    transcript (12 messages) should not hit hygiene compression by default,
    but WILL when the user lowers the hard-limit to 10.  Verifies the new
    config key is actually read and applied at the force-compress gate.
    """
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class FakeCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id", "fake-session")
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            self.session_id = f"{self.session_id}_compressed"
            return ([{"role": "assistant", "content": "compressed"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    # Write config.yaml with lowered hard-limit
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_hard_message_limit: 10\n"
    )

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:private:12345",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="private",
    )
    # 12 messages: below default → no compression without override,
    # but above the configured limit of 10 → should compress.
    runner.session_store.load_transcript.return_value = _make_history(12, content_size=40)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    # Pick a context length large enough that the token-based threshold
    # won't trigger for 12 short messages — hard-limit must be the ONLY
    # thing firing compression.
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 1_000_000,
    )

    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    # The compression agent was instantiated → hard-limit fired on the
    # configured value (10), not the hardcoded 400 default.
    assert FakeCompressAgent.last_instance is not None, (
        "Expected hygiene compression to fire when message count (12) "
        "exceeds configured hygiene_hard_message_limit (10)"
    )


# ---------------------------------------------------------------------------
# Cooldown persistence across gateway restarts (#74136)
# ---------------------------------------------------------------------------

def _make_cooldown_runner(monkeypatch, tmp_path, agent_cls, session_db, session_id):
    """Scaffolding for the restart-persistence tests: a fresh GatewayRunner
    wired to a REAL AsyncSessionDB facade (not a MagicMock) so the hygiene
    cooldown check/write paths exercise the actual SQLite-backed methods."""
    from hermes_state import AsyncSessionDB

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_failure_cooldown_seconds: 300\n",
        encoding="utf-8",
    )

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:12345",
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = _make_history(6, content_size=400)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    # The real async facade over the real SQLite-backed SessionDB — the
    # production shape.  A SimpleNamespace(_db=MagicMock()) here would let
    # the assertion pass against methods that don't actually persist.
    runner._session_db = AsyncSessionDB(session_db)
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100,
    )

    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="12345",
        ),
        message_id="1",
    )
    return runner, adapter, event


@pytest.mark.asyncio
async def test_hygiene_compression_cooldown_survives_gateway_restart(
    monkeypatch, tmp_path
):
    """Regression for #74136: the compression-failure cooldown must be
    persisted to the state DB, not an in-memory dict on the runner.

    Fail a hygiene compression on runner #1, tear the runner down, build a
    FRESH runner on the SAME database (simulating a gateway restart), and
    assert the second runner still honors the cooldown — i.e. it does not
    re-instantiate a compression agent for the same failing session.
    """
    from hermes_state import SessionDB

    gateway_run = importlib.import_module("gateway.run")
    session_id = "sess-restart"
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id, "telegram")

        main_thread = threading.get_ident()
        streak_threads = []
        original_cooldown_for_failure = gateway_run._hygiene_cooldown_for_failure

        def tracked_cooldown_for_failure(*args, **kwargs):
            streak_threads.append(threading.get_ident())
            return original_cooldown_for_failure(*args, **kwargs)

        monkeypatch.setattr(
            gateway_run,
            "_hygiene_cooldown_for_failure",
            tracked_cooldown_for_failure,
        )

        class AbortingCompressAgent:
            instances = 0

            def __init__(self, **kwargs):
                type(self).instances += 1
                self.session_id = kwargs.get("session_id", session_id)
                self._session_db = kwargs.get("session_db")
                self._last_compaction_in_place = False
                self.context_compressor = SimpleNamespace(
                    bind_session_state=MagicMock(),
                    _last_compress_aborted=True,
                    _last_summary_error="aux model exploded",
                    _last_aux_model_failure_model=None,
                )
                self.shutdown_memory_provider = MagicMock()
                self.close = MagicMock()

            def _compress_context(self, messages, *_args, **_kwargs):
                # Summary generation failed: compressor aborts and returns
                # the transcript unchanged.
                return (messages, None)

        runner1, _adapter1, event1 = _make_cooldown_runner(
            monkeypatch, tmp_path, AbortingCompressAgent, db, session_id
        )
        assert await runner1._handle_message(event1) == "ok"
        assert AbortingCompressAgent.instances == 1
        assert len(streak_threads) == 1
        assert streak_threads[0] != main_thread

        # The abort must have persisted a cooldown to the DB.
        state = db.get_compression_failure_cooldown(session_id)
        assert state is not None and state["remaining_seconds"] > 0, (
            "hygiene compression abort did not persist a cooldown to the "
            f"state DB; got {state!r}"
        )

        # --- simulate a gateway restart: brand-new runner, same DB ---
        del runner1

        class ShouldNotRunAgent:
            instances = 0

            def __init__(self, **kwargs):
                type(self).instances += 1
                self.context_compressor = SimpleNamespace(
                    bind_session_state=MagicMock(),
                    _last_compress_aborted=False,
                    _last_aux_model_failure_model=None,
                )
                self.shutdown_memory_provider = MagicMock()
                self.close = MagicMock()

            def _compress_context(self, messages, *_args, **_kwargs):
                return (messages, None)

        runner2, _adapter2, event2 = _make_cooldown_runner(
            monkeypatch, tmp_path, ShouldNotRunAgent, db, session_id
        )
        assert await runner2._handle_message(event2) == "ok"
        assert ShouldNotRunAgent.instances == 0, (
            "REGRESSION (#74136): a fresh GatewayRunner on the same state DB "
            "re-ran the failing hygiene compression — the failure cooldown "
            "was lost across the restart (in-memory dict instead of the "
            "DB-backed record/get methods)."
        )
        # The user turn itself still runs; only compression is skipped.
        assert runner2._run_agent.await_count == 1

        # Once the first deadline expires, the next failed attempt after a
        # restart must use rung 2 (900s), not start over at 300s (#86650).
        db.clear_compression_failure_cooldown(session_id)
        runner3, _adapter3, event3 = _make_cooldown_runner(
            monkeypatch, tmp_path, AbortingCompressAgent, db, session_id
        )
        assert await runner3._handle_message(event3) == "ok"
        assert AbortingCompressAgent.instances == 2
        assert len(streak_threads) == 2
        assert all(thread_id != main_thread for thread_id in streak_threads)
        escalated = db.get_compression_failure_cooldown(session_id)
        assert escalated is not None
        assert escalated["remaining_seconds"] == pytest.approx(900, abs=5)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Commit-fence cancel must not livelock hygiene (#96953)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hygiene_fence_cancel_records_cooldown_without_abort_flag(
    monkeypatch, tmp_path
):
    """A fence-cancelled hygiene worker returns the original transcript with
    ``_last_compress_aborted`` still False (failure_class=commit_fence_cancelled).

    That used to skip the abort-cooldown block, so the next turn immediately
    re-armed hygiene and waited up to the 600s ceiling behind a doomed attempt.
    """
    from hermes_state import SessionDB

    session_id = "sess-fence-cancel"

    class FenceCancelCompressAgent:
        instances = 0

        def __init__(self, **kwargs):
            type(self).instances += 1
            self.session_id = kwargs.get("session_id", session_id)
            self._session_db = kwargs.get("session_db")
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_summary_error=None,
                _last_aux_model_failure_model=None,
            )
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()

        def _compress_context(self, messages, *_args, commit_fence=None, **_kwargs):
            if commit_fence is not None:
                assert commit_fence.try_cancel_before_commit() is True
            return (messages, None)

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id, "telegram")
        runner1, adapter1, event1 = _make_cooldown_runner(
            monkeypatch, tmp_path, FenceCancelCompressAgent, db, session_id
        )
        assert await runner1._handle_message(event1) == "ok"
        assert FenceCancelCompressAgent.instances == 1
        state = db.get_compression_failure_cooldown(session_id)
        assert state is not None and state["remaining_seconds"] > 0, (
            "fence-cancelled hygiene compression did not persist a cooldown; "
            f"got {state!r}"
        )
        assert not any(
            "Shortening the conversation history failed" in s["content"] for s in adapter1.sent
        ), "fence-cancel during /stop or /restart must not toast an abort"

        class ShouldNotRunAgent:
            instances = 0

            def __init__(self, **kwargs):
                type(self).instances += 1
                self.context_compressor = SimpleNamespace(
                    bind_session_state=MagicMock(),
                    _last_compress_aborted=False,
                    _last_aux_model_failure_model=None,
                )
                self.shutdown_memory_provider = MagicMock()
                self.close = MagicMock()

            def _compress_context(self, messages, *_args, **_kwargs):
                return (messages, None)

        runner2, _adapter2, event2 = _make_cooldown_runner(
            monkeypatch, tmp_path, ShouldNotRunAgent, db, session_id
        )
        assert await runner2._handle_message(event2) == "ok"
        assert ShouldNotRunAgent.instances == 0, (
            "REGRESSION (#96953): hygiene re-armed after a commit-fence "
            "cancel instead of honoring the failure cooldown"
        )
        assert runner2._run_agent.await_count == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_hygiene_skips_when_compression_already_in_flight(
    monkeypatch, tmp_path
):
    """Do not spawn a sibling hygiene compressor while a lock is already held."""
    from hermes_state import SessionDB

    session_id = "sess-in-flight"

    class ShouldNotRunAgent:
        instances = 0

        def __init__(self, **kwargs):
            type(self).instances += 1
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
            )
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()

        def _compress_context(self, messages, *_args, **_kwargs):
            return (messages, None)

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id, "telegram")
        runner, _adapter, event = _make_cooldown_runner(
            monkeypatch, tmp_path, ShouldNotRunAgent, db, session_id
        )
        runner._session_has_compression_in_flight = AsyncMock(return_value=True)
        assert await runner._handle_message(event) == "ok"
        assert ShouldNotRunAgent.instances == 0
        assert runner._run_agent.await_count == 1
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Fail-closed in-context bound when hygiene has not landed (#111988)
# ---------------------------------------------------------------------------

_HARD_LIMIT = 20


def _make_bound_probe_transcript(total: int = 60) -> list:
    """Transcript with a setup prefix, a tool group straddling the tail cut, and a newest tail.

    With ``_HARD_LIMIT`` = 20 and one setup row, the newest-19 cut starts at index 41 — the
    first tool result of the group at 40/41/42.
    """
    rows = [{"role": "session_meta", "tools": [], "model": "m", "platform": "telegram", "timestamp": "t0"}]
    while len(rows) < total - 20:
        role = "user" if len(rows) % 2 else "assistant"
        rows.append({"role": role, "content": f"old-{len(rows)}", "timestamp": f"t{len(rows)}"})
    rows.append({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "t"}}]})
    rows.append({"role": "tool", "tool_call_id": "c1", "content": "old tool output 1"})
    rows.append({"role": "tool", "tool_call_id": "c2", "content": "old tool output 2"})
    rows.append({"role": "user", "content": "newest ask", "timestamp": "t-ask"})
    rows.append({"role": "assistant", "content": "newest reply", "timestamp": "t-reply"})
    while len(rows) < total:
        role = "user" if len(rows) % 2 else "assistant"
        rows.append({"role": role, "content": f"tail-{len(rows)}", "timestamp": f"t{len(rows)}"})
    if total == 60:
        assert rows[41].get("role") == "tool", "fixture must put the tail cut on a tool result"
    return rows


def _turn_payload(runner):
    """The history the gateway handed the turn runner for the model (one turn)."""
    assert runner._run_agent.await_count == 1
    return runner._run_agent.call_args.kwargs["history"]


def _make_bound_runner(monkeypatch, tmp_path, agent_cls, cfg_text, transcript):
    """``_make_cooldown_runner`` with a >hard-limit transcript and a lowered hard limit."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-bound", "telegram")
    runner, adapter, event = _make_cooldown_runner(
        monkeypatch, tmp_path, agent_cls, db, "sess-bound"
    )
    (tmp_path / "config.yaml").write_text(cfg_text, encoding="utf-8")
    runner.session_store.load_transcript.return_value = transcript
    return db, runner, adapter, event


class _LandedCompressAgent:
    """Rotating hygiene compressor that SUCCEEDS with a transcript longer than the hard limit."""

    last_instance = None

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id", "fake-session")
        self._session_db = kwargs.get("session_db")
        self._last_compaction_in_place = False
        self.context_compressor = SimpleNamespace(
            bind_session_state=MagicMock(),
            _last_compress_aborted=False,
            _last_aux_model_failure_model=None,
        )
        self.shutdown_memory_provider = MagicMock()
        self.close = MagicMock()
        type(self).last_instance = self

    def _compress_context(self, messages, *_args, **_kwargs):
        # 25 rows: over the 20-message limit but far under the input's token count, so the
        # adopted (landed) transcript must reach the model untouched.
        compressed = [{"role": "assistant", "content": f"summary {i}"} for i in range(25)]
        type(self).last_compressed = compressed
        self.session_id = f"{self.session_id}_compressed"
        return (compressed, None)


def test_bound_model_input_without_hygiene_is_deterministic_and_fail_closed():
    """The bound keeps the setup head + the newest tail, drops the middle, never mutates input."""
    from gateway.run_turn import bound_model_input_without_hygiene

    rows = _make_bound_probe_transcript()
    snapshot = [dict(r) for r in rows]

    bounded = bound_model_input_without_hygiene(rows, _HARD_LIMIT)

    assert len(bounded) <= _HARD_LIMIT
    assert len(bounded) < len(rows), "a >limit transcript must actually be bounded"
    assert bounded[0] is rows[0], "the leading system/setup row must survive"
    assert bounded[-1] is rows[-1], "the newest row must survive"
    assert bounded[1].get("role") != "tool", "the kept tail must not start on an orphan tool result"
    assert rows[:] == snapshot, "the source transcript must not be mutated in place"
    # Same input, same cut — no randomness, no clock.
    assert bound_model_input_without_hygiene(rows, _HARD_LIMIT) == bounded
    # At or below the limit nothing is dropped, and the SAME list comes back.
    assert bound_model_input_without_hygiene(rows, len(rows)) is rows
    assert bound_model_input_without_hygiene(rows, len(rows) + 5) is rows


@pytest.mark.parametrize("limit", [3, 4])
def test_bound_model_input_without_hygiene_starts_at_user_after_tool_round(limit):
    """A cut inside a tool round cannot leave a function call as the first model turn."""
    from gateway.run_turn import bound_model_input_without_hygiene

    rows = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "earlier request"},
        {"role": "assistant", "tool_calls": [{"id": "call", "function": {"name": "t"}}]},
        {"role": "tool", "tool_call_id": "call", "content": "result"},
        {"role": "user", "content": "latest request"},
    ]
    bounded = bound_model_input_without_hygiene(rows, limit)

    assert bounded == [rows[0], rows[-1]]
    assert len(bounded) <= limit
    assert rows[2]["role"] == "assistant" and rows[3]["role"] == "tool"


@pytest.mark.parametrize("limit", [3, 4])
def test_bound_model_input_without_hygiene_drops_assistant_only_tail(limit):
    """A completed stored transcript has no current user row; never keep its orphan reply."""
    from gateway.run_turn import bound_model_input_without_hygiene

    rows = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "earlier request"},
        {"role": "assistant", "tool_calls": [{"id": "call", "function": {"name": "t"}}]},
        {"role": "tool", "tool_call_id": "call", "content": "result"},
        {"role": "assistant", "content": "final reply"},
    ]

    bounded = bound_model_input_without_hygiene(rows, limit)

    assert bounded == [rows[0]]
    assert len(bounded) <= limit


@pytest.mark.asyncio
async def test_hygiene_miss_bounds_the_model_payload(monkeypatch, tmp_path):
    """Turn-hold expiry without a landed summary must not feed the model the whole transcript
    (#111988); a landed commit is still adopted byte-identical."""
    worker_started = threading.Event()
    release_worker = threading.Event()
    cleanup_done = threading.Event()
    session_id = "sess-bound"

    class StreamingCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id", session_id)
            self._session_db = kwargs.get("session_db")
            self._last_compaction_in_place = False
            self.context_compressor = SimpleNamespace(
                bind_session_state=MagicMock(),
                _last_compress_aborted=False,
                _last_aux_model_failure_model=None,
            )
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock(side_effect=cleanup_done.set)
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, commit_fence=None, **_kwargs):
            worker_started.set()
            release_worker.wait(timeout=10)
            return (messages, None)

    transcript = _make_bound_probe_transcript()
    db, runner, _adapter, event = _make_bound_runner(
        monkeypatch, tmp_path, StreamingCompressAgent,
        "compression:\n"
        "  enabled: true\n"
        f"  hygiene_hard_message_limit: {_HARD_LIMIT}\n"
        "  hygiene_timeout_seconds: 60\n"
        "  hygiene_total_ceiling_seconds: 600\n"
        "  hygiene_max_turn_hold_seconds: 0.3\n",
        transcript,
    )
    try:
        assert await asyncio.wait_for(runner._handle_message(event), timeout=15) == "ok"
        assert worker_started.wait(timeout=2)

        payload = _turn_payload(runner)
        assert len(payload) <= _HARD_LIMIT, (
            f"hygiene never landed but the model got {len(payload)} messages "
            f"(limit {_HARD_LIMIT}) — unbounded fail-open"
        )
        assert payload[0] is transcript[0], "the system/setup head must be preserved"
        assert payload[-1] is transcript[-1], "the newest tail must be preserved"
        assert payload[1].get("role") != "tool", "the kept tail must not start on an orphan tool result"

        # History on disk untouched: the loaded transcript still holds every row and nothing
        # rewrote it — only the payload about to be sent to the model was clipped.
        assert len(transcript) == len(_make_bound_probe_transcript())
        assert transcript[20]["content"].startswith("old-")
        runner.session_store.rewrite_transcript.assert_not_called()

        release_worker.set()
        await asyncio.wait_for(asyncio.to_thread(cleanup_done.wait), timeout=3)

        # Control: a LANDED hygiene commit is adopted as-is — its 25 summary rows exceed the
        # limit, and the bound must not touch them (only the unlanded path is clipped).
        db.create_session("sess-landed", "telegram")
        runner2, _adapter2, event2 = _make_cooldown_runner(
            monkeypatch, tmp_path, _LandedCompressAgent, db, "sess-landed"
        )
        runner2.session_store.load_transcript.return_value = _make_bound_probe_transcript()
        assert await asyncio.wait_for(runner2._handle_message(event2), timeout=15) == "ok"
        assert _turn_payload(runner2) is _LandedCompressAgent.last_compressed
    finally:
        release_worker.set()
        db.close()
