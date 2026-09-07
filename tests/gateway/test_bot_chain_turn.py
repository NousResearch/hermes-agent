import asyncio
import contextlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.bot_chain import (
    BotChainCancelled,
    BotChainResult,
    BotChainStep,
    BotChainRunner,
    parse_bot_chain_message,
    publish_bot_chain_history,
)
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionStore
from hermes_state import SessionDB
from hermes_cli.bot_profiles import BotProfile


@pytest.fixture(autouse=True)
def _routed_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


class _AsyncStore:
    def __init__(self):
        self._store = SimpleNamespace(bot_chain_publication_guard=lambda *_: contextlib.nullcontext())
        self.appended = []
        self.updated = []
        self.admissions = {}
        self.chain_names = {}
        self.settlements = []
        self.fail_user_append = False

    async def has_platform_message_id(self, session_id, message_id):
        return False

    async def admit_bot_chain_delivery(self, session_id, message_id, chain_name):
        key = (session_id, message_id)
        # The first admission binds the chain identity; later deliveries
        # reuse it and never overwrite it.
        self.chain_names.setdefault(key, chain_name)
        if key in self.admissions:
            return self.admissions[key]
        self.admissions[key] = "admitted"
        return "admitted"

    async def get_bot_chain_delivery(self, session_id, message_id):
        key = (session_id, message_id)
        if key not in self.admissions:
            return None
        return {
            "chain_name": self.chain_names.get(key),
            "state": self.admissions[key],
        }

    async def mark_bot_chain_delivery_running(self, session_id, message_id):
        return "claim-token"

    async def settle_bot_chain_delivery(
        self, session_id, message_id, *, outcome, detail="", owner_token=None
    ):
        self.admissions[(session_id, message_id)] = "settled"
        self.settlements.append((session_id, message_id, outcome))
        return True

    async def append_to_transcript(self, session_id, message):
        if self.fail_user_append and message.get("role") == "user":
            raise OSError("state.db write failed")
        self.appended.append((session_id, message))

    async def update_session(self, session_key, **kwargs):
        self.updated.append((session_key, kwargs))


class _DurableAsyncStore(_AsyncStore):
    """Gateway facade whose admission and transcript writes hit real SQLite."""

    def __init__(self, db):
        super().__init__()
        self.db = db
        self._store = SimpleNamespace(bot_chain_publication_guard=db.bot_chain_publication_guard)

    async def has_platform_message_id(self, session_id, message_id):
        return self.db.has_platform_message_id(session_id, message_id)

    async def admit_bot_chain_delivery(self, session_id, message_id, chain_name):
        return self.db.admit_bot_chain_delivery(session_id, message_id, chain_name)

    async def mark_bot_chain_delivery_running(self, session_id, message_id):
        return self.db.mark_bot_chain_delivery_running(session_id, message_id)

    async def settle_bot_chain_delivery(
        self, session_id, message_id, *, outcome, detail="", owner_token=None
    ):
        return self.db.settle_bot_chain_delivery(
            session_id,
            message_id,
            outcome=outcome,
            detail=detail,
            owner_token=owner_token,
        )

    async def get_bot_chain_delivery(self, session_id, message_id):
        return self.db.get_bot_chain_delivery(session_id, message_id)

    async def release_bot_chain_delivery_claim(
        self, session_id, message_id, owner_token=None
    ):
        return self.db.release_bot_chain_delivery_claim(
            session_id, message_id, owner_token
        )

    async def append_to_transcript(self, session_id, message):
        if self.fail_user_append and message.get("role") == "user":
            raise OSError("state.db write failed")
        self.db.append_message(
            session_id,
            role=message.get("role", "unknown"),
            content=message.get("content"),
            platform_message_id=message.get("message_id"),
            timestamp=message.get("timestamp"),
        )


def _profile(tmp_path, name="worker"):
    profile_home = tmp_path / ".hermes" / "profiles" / name
    profile_home.mkdir(parents=True, exist_ok=True)
    return BotProfile(
        name=name,
        path=profile_home,
        model="test/model",
        provider="nous",
        system_prompt="Work",
    )


def _persist_completed_bot_turn(profile, conversation_name, prompt, output):
    """Persist the exact model/history side effect a crashed run left behind."""
    db = SessionDB(Path(profile.path) / "state.db")
    try:
        db.create_session(
            "canonical",
            source="desktop",
            model=profile.model,
            profile_name=profile.name,
        )
        db.set_session_title("canonical", "Bot Chat")
        db.set_session_hidden("canonical", True)
        db.create_session(
            "completed-chain",
            source="cli",
            model=profile.model,
            model_config={"follow_profile_config": True},
            profile_name=profile.name,
        )
        db.set_session_title("completed-chain", conversation_name)
        db.set_session_hidden("completed-chain", True)
        db.append_messages_batch(
            "completed-chain",
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ],
        )
    finally:
        db.close()
    publish_bot_chain_history(profile, conversation_name)


def test_gateway_bot_chain_returns_one_labeled_reply_and_persists_retryable_turn(
    monkeypatch,
):
    async def _inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _inline_to_thread)
    first = BotProfile(
        name="deepseek",
        path=Path("/tmp/deepseek"),
        model="deepseek-v4",
        provider="deepseek",
        system_prompt="First",
    )
    second = BotProfile(
        name="default",
        path=Path("/tmp/default"),
        model="test/model",
        provider="nous",
        system_prompt="Second",
    )
    result = BotChainResult(
        prompt="do the task",
        steps=(
            BotChainStep(first, "do the task", "draft"),
            BotChainStep(second, "handoff", "final answer"),
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [first, second],
    )
    monkeypatch.setattr(BotChainRunner, "run", lambda *_args, **_kwargs: result)

    runner = object.__new__(GatewayRunner)
    async_store = _AsyncStore()
    runner.session_store = async_store._store
    runner._async_session_store = async_store
    state = SimpleNamespace(turn=SimpleNamespace(agent=None, started_ts=0.0))
    runner._session_state = lambda _key: state
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-7", thread_id="31")
    event = SimpleNamespace(
        text="$DeepSeek $Default do the task",
        message_id="telegram-42",
        internal=False,
        source=source,
    )
    session = SimpleNamespace(session_id="session-1", session_key="telegram:chat-7:31")
    request = parse_bot_chain_message(event.text)

    response = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )

    assert response == (
        "$deepseek:\ndraft\n\n$default (final):\nfinal answer"
    )
    assert [row[1]["role"] for row in async_store.appended] == [
        "user",
        "assistant",
    ]
    assert async_store.appended[0][1]["content"] == event.text
    assert async_store.appended[0][1]["message_id"] == "telegram-42"
    assert async_store.appended[1][1]["content"] == response
    assert async_store.updated == [
        (session.session_key, {"touch_activity": True})
    ]


def test_gateway_bot_chain_stop_suppresses_duplicate_delivery(monkeypatch):
    async def _inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _inline_to_thread)
    profile = BotProfile(
        name="worker",
        path=Path("/tmp/worker"),
        model="test/model",
        provider="nous",
        system_prompt="Work",
    )
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    def _cancelled(*_args, **_kwargs):
        raise BotChainCancelled("Bot chain stopped.")

    monkeypatch.setattr(BotChainRunner, "run", _cancelled)

    runner = object.__new__(GatewayRunner)
    async_store = _AsyncStore()
    runner.session_store = async_store._store
    runner._async_session_store = async_store
    state = SimpleNamespace(turn=SimpleNamespace(agent=None, started_ts=0.0))
    runner._session_state = lambda _key: state
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-43",
        internal=False,
    )
    session = SimpleNamespace(session_id="session-1", session_key="telegram:chat-7:31")
    request = parse_bot_chain_message(event.text)

    response = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )

    assert response is None
    assert async_store.appended[-1][1]["content"] == "Bot chain stopped."


def _wire_runner(monkeypatch, async_store):
    async def _inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _inline_to_thread)
    runner = object.__new__(GatewayRunner)
    runner.session_store = async_store._store
    runner._async_session_store = async_store
    state = SimpleNamespace(turn=SimpleNamespace(agent=None, started_ts=0.0))
    runner._session_state = lambda _key: state
    return runner


def test_recovery_read_failure_releases_receipt_without_execution(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "ingress.db")
    db.create_session("session-1", source="telegram")
    profile = _profile(tmp_path)
    target = SessionDB(profile.path / "state.db")
    target.close()
    monkeypatch.setattr("hermes_cli.bot_profiles.resolve_bot_chain", lambda _names: [profile])
    def no_execution(*args, **kwargs):
        pytest.fail("unreadable durable history must never replay a model turn")
    monkeypatch.setattr("agent.bot_chain.default_bot_turn_executor", lambda: no_execution)
    def unavailable(*args, **kwargs):
        raise OSError("state read unavailable")
    monkeypatch.setattr(SessionDB, "get_session_by_title", unavailable)
    runner = _wire_runner(monkeypatch, _DurableAsyncStore(db))
    event = SimpleNamespace(text="$worker task", message_id="recovery-deferred", internal=False)
    session = SimpleNamespace(session_id="session-1", session_key="telegram:test")
    try:
        response = asyncio.run(runner._handle_bot_chain_turn(
            event, session, session.session_key, parse_bot_chain_message(event.text),
        ))
        assert "recovery deferred" in response
        receipt = db.get_bot_chain_delivery("session-1", event.message_id)
        assert receipt["state"] == "admitted"
        assert receipt["owner_token"] is None
        assert not db.has_platform_message_id("session-1", event.message_id)
    finally:
        db.close()


def test_gateway_bot_chain_redelivery_after_transcript_failure_never_reexecutes(
    monkeypatch,
):
    """Reviewer regression (#100758 blocker 3): the FIRST gateway transcript
    append fails after the chain executed; the platform then redelivers the
    same message id. The admission receipt must forbid a second
    BotChainRunner execution (and therefore a second canonical-history
    publish)."""
    profile = BotProfile(
        name="worker",
        path=Path("/tmp/worker"),
        model="test/model",
        provider="nous",
        system_prompt="Work",
    )
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    run_calls = []

    def _run(self, profiles, prompt, **kwargs):
        run_calls.append(kwargs.get("conversation_name"))
        return BotChainResult(
            prompt=prompt,
            steps=(BotChainStep(profile, prompt, "done"),),
        )

    monkeypatch.setattr(BotChainRunner, "run", _run)

    async_store = _AsyncStore()
    async_store.fail_user_append = True  # the exact failure from the review
    runner = _wire_runner(monkeypatch, async_store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-99",
        internal=False,
    )
    session = SimpleNamespace(session_id="session-1", session_key="telegram:chat-7:31")
    request = parse_bot_chain_message(event.text)

    first = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )
    # The chain executed and the response is returned even though the
    # transcript write failed.
    assert first == "$worker (final):\ndone"
    assert async_store.settlements == [("session-1", "telegram-99", "completed")]

    # Platform redelivers the same message id.
    second = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )

    assert second is None
    # Exactly one execution, and the admission receipt carried the chain
    # identity that was used for the run.
    assert len(run_calls) == 1
    assert run_calls[0] is not None and run_calls[0].startswith("Bot Chain ")


def test_gateway_bot_chain_refuses_to_execute_without_durable_receipt(
    monkeypatch,
):
    """Admission write failure fails closed: no receipt, no execution."""

    class _BrokenAdmissionStore(_AsyncStore):
        async def admit_bot_chain_delivery(self, session_id, message_id, chain_name):
            raise OSError("state.db unavailable")

    run_calls = []
    monkeypatch.setattr(
        BotChainRunner, "run", lambda *a, **k: run_calls.append(k) or None
    )
    async_store = _BrokenAdmissionStore()
    runner = _wire_runner(monkeypatch, async_store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-100",
        internal=False,
    )
    session = SimpleNamespace(session_id="session-1", session_key="telegram:chat-7:31")
    request = parse_bot_chain_message(event.text)

    response = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )

    assert response is not None and "resend" in response
    assert run_calls == []


@pytest.mark.parametrize("claim_result", [False, "error"])
def test_gateway_executes_zero_turns_without_durable_running_claim(
    monkeypatch, claim_result
):
    """A lost or failed atomic claim may never cross into model execution."""

    class _UnclaimedStore(_AsyncStore):
        async def mark_bot_chain_delivery_running(self, session_id, message_id):
            if claim_result == "error":
                raise OSError("claim write failed")
            return False

    profile = BotProfile(
        name="worker",
        path=Path("/tmp/worker"),
        model="test/model",
        provider="nous",
        system_prompt="Work",
    )
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    run_calls = []
    monkeypatch.setattr(
        BotChainRunner,
        "run",
        lambda *_args, **_kwargs: run_calls.append(True),
    )
    store = _UnclaimedStore()
    runner = _wire_runner(monkeypatch, store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id=f"telegram-claim-{claim_result}",
        internal=False,
    )
    session = SimpleNamespace(
        session_id="session-1",
        session_key="telegram:chat-7:31",
    )

    asyncio.run(
        runner._handle_bot_chain_turn(
            event,
            session,
            session.session_key,
            parse_bot_chain_message(event.text),
        )
    )

    assert run_calls == []


def test_session_store_admission_without_owning_db_is_typed_failure(tmp_path):
    """No SQLite owner means no durable receipt, never synthetic admission."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    store = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    store._db = None
    try:
        with pytest.raises(Exception) as exc_info:
            store.admit_bot_chain_delivery(
                "session-without-owner",
                "telegram-no-db",
                "Bot Chain unavailable",
            )
        assert type(exc_info.value).__name__ == "BotChainAdmissionUnavailable"
    finally:
        store.close_all_db_handles()


def test_gateway_redelivery_after_admission_reuses_identity_and_executes_once(
    tmp_path, monkeypatch
):
    """A crash before execution must resume the reserved chain, not lose it."""
    db = SessionDB(tmp_path / "ingress.db")
    db.create_session("session-1", source="telegram")
    original_name = "Bot Chain original-admission"
    assert (
        db.admit_bot_chain_delivery("session-1", "telegram-admitted", original_name)
        == "admitted"
    )
    profile = _profile(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    conversation_names = []

    def _run(self, profiles, prompt, **kwargs):
        conversation_names.append(kwargs.get("conversation_name"))
        return BotChainResult(
            prompt=prompt,
            steps=(BotChainStep(profile, prompt, "done"),),
        )

    monkeypatch.setattr(BotChainRunner, "run", _run)
    store = _DurableAsyncStore(db)
    runner = _wire_runner(monkeypatch, store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-admitted",
        internal=False,
    )
    session = SimpleNamespace(
        session_id="session-1",
        session_key="telegram:chat-7:31",
    )
    try:
        response = asyncio.run(
            runner._handle_bot_chain_turn(
                event,
                session,
                session.session_key,
                parse_bot_chain_message(event.text),
            )
        )

        assert response == "$worker (final):\ndone"
        assert conversation_names == [original_name]
        receipt = db.get_bot_chain_delivery("session-1", "telegram-admitted")
        assert receipt["state"] == "settled"
        assert receipt["outcome"] == "completed"
        assert receipt["chain_name"] == original_name
    finally:
        db.close()


def test_gateway_recovers_published_turn_after_owner_crash_without_reexecution(
    tmp_path, monkeypatch
):
    """Published history is durable proof: recover its result, never rerun it."""
    ingress_path = tmp_path / "ingress.db"
    db = SessionDB(ingress_path)
    db.create_session("session-1", source="telegram")
    conversation_name = "Bot Chain published-before-settlement"
    assert (
        db.admit_bot_chain_delivery(
            "session-1", "telegram-published", conversation_name
        )
        == "admitted"
    )

    profile = _profile(tmp_path)
    _persist_completed_bot_turn(
        profile,
        conversation_name,
        "do the task",
        "durable answer",
    )
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    model_turns = []

    class _ForbiddenSecondTurn:
        def __call__(self, *_args, **_kwargs):
            model_turns.append(True)
            raise AssertionError("a durable completed turn must not run again")

    monkeypatch.setattr(
        "agent.bot_chain.default_bot_turn_executor",
        lambda: _ForbiddenSecondTurn(),
    )

    store = _DurableAsyncStore(db)
    runner = _wire_runner(monkeypatch, store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-published",
        internal=False,
    )
    session = SimpleNamespace(
        session_id="session-1",
        session_key="telegram:chat-7:31",
    )
    try:
        response = asyncio.run(
            runner._handle_bot_chain_turn(
                event,
                session,
                session.session_key,
                parse_bot_chain_message(event.text),
            )
        )

        assert response == "$worker (final):\ndurable answer"
        assert model_turns == []
        receipt = db.get_bot_chain_delivery("session-1", "telegram-published")
        assert receipt["state"] == "settled"
        assert receipt["outcome"] == "completed"
        assert receipt["chain_name"] == conversation_name
    finally:
        db.close()

    profile_db = SessionDB(Path(profile.path) / "state.db")
    try:
        canonical = profile_db.get_session_by_title("Bot Chat")
        messages = profile_db.get_messages_as_conversation(canonical["id"])
        assert [
            (message["role"], message["content"])
            for message in messages
        ] == [
            ("user", "do the task"),
            ("assistant", "durable answer"),
        ]
    finally:
        profile_db.close()


def test_gateway_settlement_failure_releases_claim_and_redelivery_recovers(
    tmp_path, monkeypatch
):
    """Stale-running regression (#100758): the chain executed and its step
    output persisted durably, but the settlement write failed while the
    gateway process stayed alive. Left in ``running`` under this live owner,
    the receipt would stand every redelivery down forever. The failed
    settlement must release the claim instead, and the redelivery must
    resume the admission, recover the durable step output, and settle
    WITHOUT re-executing the model turn."""
    db = SessionDB(tmp_path / "ingress.db")
    db.create_session("session-1", source="telegram")

    profile = _profile(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    model_turns = []

    def _executing_factory():
        def _execute(profile, prompt, control, *, conversation_name):
            # The real side effect of a completed turn: durable history in
            # the profile's own state.db, exactly what recovery later reads.
            _persist_completed_bot_turn(
                profile, conversation_name, prompt, "durable answer"
            )
            model_turns.append(conversation_name)
            return "durable answer"

        return _execute

    monkeypatch.setattr(
        "agent.bot_chain.default_bot_turn_executor", _executing_factory
    )

    store = _DurableAsyncStore(db)
    # The settlement write wedged for the first delivery (attempt + retry),
    # and works again afterwards.
    real_settle = store.settle_bot_chain_delivery
    settle_attempts = []

    async def _flaky_settle(
        session_id, message_id, *, outcome, detail="", owner_token=None
    ):
        settle_attempts.append(message_id)
        if len(settle_attempts) <= 2:
            raise OSError("state.db wedged")
        return await real_settle(
            session_id, message_id, outcome=outcome, detail=detail,
            owner_token=owner_token,
        )

    store.settle_bot_chain_delivery = _flaky_settle
    # The same wedge that broke settlement also breaks the transcript append,
    # so no message_id row exists to legacy-dedupe the redelivery.
    store.fail_user_append = True
    runner = _wire_runner(monkeypatch, store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-stale",
        internal=False,
    )
    session = SimpleNamespace(
        session_id="session-1",
        session_key="telegram:chat-7:31",
    )
    try:
        first = asyncio.run(
            runner._handle_bot_chain_turn(
                event,
                session,
                session.session_key,
                parse_bot_chain_message(event.text),
            )
        )

        assert first == "$worker (final):\ndurable answer"
        assert len(model_turns) == 1
        # The failed settlement released the claim: the receipt is NOT
        # wedged in "running" under this still-alive process.
        receipt = db.get_bot_chain_delivery("session-1", "telegram-stale")
        assert receipt["state"] == "admitted"
        chain_name = receipt["chain_name"]
        assert chain_name == model_turns[0]

        # The platform redelivers the same message; no model turn may run.
        class _ForbiddenSecondTurn:
            def __call__(self, *_args, **_kwargs):
                model_turns.append(True)
                raise AssertionError(
                    "a durable completed turn must not run again"
                )

        monkeypatch.setattr(
            "agent.bot_chain.default_bot_turn_executor",
            lambda: _ForbiddenSecondTurn(),
        )
        # The wedge cleared: settlement and transcript writes work again.
        store.fail_user_append = False
        second = asyncio.run(
            runner._handle_bot_chain_turn(
                event,
                session,
                session.session_key,
                parse_bot_chain_message(event.text),
            )
        )

        assert second == "$worker (final):\ndurable answer"
        assert len(model_turns) == 1
        receipt = db.get_bot_chain_delivery("session-1", "telegram-stale")
        assert receipt["state"] == "settled"
        assert receipt["outcome"] == "completed"
        assert receipt["chain_name"] == chain_name
    finally:
        db.close()


def test_gateway_redelivery_reclaims_dead_owner_and_recovers_without_reexecution(
    tmp_path, monkeypatch
):
    """A claim owner that DIED mid-execution (not merely failed to settle)
    is reclaimed under the original chain identity, and durable step output
    is recovered without re-running the model turn."""
    ingress_path = tmp_path / "ingress.db"
    db = SessionDB(ingress_path)
    db.create_session("session-1", source="telegram")
    conversation_name = "Bot Chain dead-owner"

    # A genuinely dead owner: a child process admits and claims, then exits.
    script = """
import sys
from pathlib import Path
from hermes_state import SessionDB

db = SessionDB(Path(sys.argv[1]))
try:
    assert db.admit_bot_chain_delivery('session-1', 'telegram-dead', sys.argv[2]) == 'admitted'
    assert db.mark_bot_chain_delivery_running('session-1', 'telegram-dead')
finally:
    db.close()
"""
    subprocess.run(
        [sys.executable, "-c", script, str(ingress_path), conversation_name],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )
    receipt = db.get_bot_chain_delivery("session-1", "telegram-dead")
    assert receipt["state"] == "running"

    profile = _profile(tmp_path)
    _persist_completed_bot_turn(
        profile,
        conversation_name,
        "do the task",
        "durable answer",
    )
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    model_turns = []

    class _ForbiddenTurn:
        def __call__(self, *_args, **_kwargs):
            model_turns.append(True)
            raise AssertionError("a durable completed turn must not run again")

    monkeypatch.setattr(
        "agent.bot_chain.default_bot_turn_executor",
        lambda: _ForbiddenTurn(),
    )

    store = _DurableAsyncStore(db)
    runner = _wire_runner(monkeypatch, store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-dead",
        internal=False,
    )
    session = SimpleNamespace(
        session_id="session-1",
        session_key="telegram:chat-7:31",
    )
    try:
        response = asyncio.run(
            runner._handle_bot_chain_turn(
                event,
                session,
                session.session_key,
                parse_bot_chain_message(event.text),
            )
        )

        assert response == "$worker (final):\ndurable answer"
        assert model_turns == []
        receipt = db.get_bot_chain_delivery("session-1", "telegram-dead")
        assert receipt["state"] == "settled"
        assert receipt["outcome"] == "completed"
        # The dead owner's original chain identity was reused, not replaced.
        assert receipt["chain_name"] == conversation_name
    finally:
        db.close()


def test_gateway_redelivery_with_receipt_bypasses_legacy_transcript_dedupe(
    tmp_path, monkeypatch
):
    """Released receipt + persisted transcript row: the legacy message_id
    dedupe must not stop the redelivery before the receipt state machine —
    the receipt would otherwise stay admitted (non-terminal) forever. The
    state machine resumes the admission, recovers the durable step output,
    and settles WITHOUT re-executing the model turn."""
    db = SessionDB(tmp_path / "ingress.db")
    db.create_session("session-1", source="telegram")

    profile = _profile(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    model_turns = []

    def _executing_factory():
        def _execute(profile, prompt, control, *, conversation_name):
            _persist_completed_bot_turn(
                profile, conversation_name, prompt, "durable answer"
            )
            model_turns.append(conversation_name)
            return "durable answer"

        return _execute

    monkeypatch.setattr(
        "agent.bot_chain.default_bot_turn_executor", _executing_factory
    )

    store = _DurableAsyncStore(db)
    # Settlement wedged for the first delivery (attempt + retry); the
    # transcript append SUCCEEDED, so a message_id row exists — the exact
    # state where the legacy dedupe used to swallow the redelivery.
    real_settle = store.settle_bot_chain_delivery
    settle_attempts = []

    async def _flaky_settle(
        session_id, message_id, *, outcome, detail="", owner_token=None
    ):
        settle_attempts.append(message_id)
        if len(settle_attempts) <= 2:
            raise OSError("state.db wedged")
        return await real_settle(
            session_id, message_id, outcome=outcome, detail=detail,
            owner_token=owner_token,
        )

    store.settle_bot_chain_delivery = _flaky_settle
    runner = _wire_runner(monkeypatch, store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-legacy",
        internal=False,
    )
    session = SimpleNamespace(
        session_id="session-1",
        session_key="telegram:chat-7:31",
    )
    try:
        first = asyncio.run(
            runner._handle_bot_chain_turn(
                event,
                session,
                session.session_key,
                parse_bot_chain_message(event.text),
            )
        )

        assert first == "$worker (final):\ndurable answer"
        assert len(model_turns) == 1
        # The transcript row exists (legacy dedupe would fire), and the
        # released receipt waits for its state machine, not for that row.
        assert db.has_platform_message_id("session-1", "telegram-legacy")
        receipt = db.get_bot_chain_delivery("session-1", "telegram-legacy")
        assert receipt["state"] == "admitted"
        chain_name = receipt["chain_name"]

        class _ForbiddenSecondTurn:
            def __call__(self, *_args, **_kwargs):
                model_turns.append(True)
                raise AssertionError(
                    "a durable completed turn must not run again"
                )

        monkeypatch.setattr(
            "agent.bot_chain.default_bot_turn_executor",
            lambda: _ForbiddenSecondTurn(),
        )
        second = asyncio.run(
            runner._handle_bot_chain_turn(
                event,
                session,
                session.session_key,
                parse_bot_chain_message(event.text),
            )
        )

        assert second == "$worker (final):\ndurable answer"
        assert len(model_turns) == 1
        receipt = db.get_bot_chain_delivery("session-1", "telegram-legacy")
        assert receipt["state"] == "settled"
        assert receipt["outcome"] == "completed"
        assert receipt["chain_name"] == chain_name
    finally:
        db.close()


def test_gateway_redelivery_reclaims_expired_foreign_generation_and_recovers(
    tmp_path, monkeypatch
):
    """A ``running`` claim left by a PRIOR runtime generation — a host
    identity the current process can never probe (container replaced,
    machine renamed, state directory restored) — becomes reclaimable once
    its lease expires. The redelivery resumes under the original chain
    identity, recovers the durable step output WITHOUT re-running the
    model turn, and settles the receipt."""
    db = SessionDB(tmp_path / "ingress.db")
    db.create_session("session-1", source="telegram")
    conversation_name = "Bot Chain stale-generation"
    assert (
        db.admit_bot_chain_delivery("session-1", "telegram-stale", conversation_name)
        == "admitted"
    )
    old_token = db.mark_bot_chain_delivery_running(
        "session-1", "telegram-stale"
    )
    assert old_token
    # The recording runtime is gone for good and unprobeable from here.
    import sqlite3

    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            "UPDATE bot_chain_deliveries SET owner_host = 'gone-host', "
            "owner_pid = -1, lease_expires_at = 0 WHERE session_id = 'session-1' AND "
            "platform_message_id = 'telegram-stale'"
        )
    receipt = db.get_bot_chain_delivery("session-1", "telegram-stale")
    assert receipt["state"] == "running"

    profile = _profile(tmp_path)
    _persist_completed_bot_turn(
        profile,
        conversation_name,
        "do the task",
        "durable answer",
    )
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    model_turns = []

    class _ForbiddenTurn:
        def __call__(self, *_args, **_kwargs):
            model_turns.append(True)
            raise AssertionError("a durable completed turn must not run again")

    monkeypatch.setattr(
        "agent.bot_chain.default_bot_turn_executor",
        lambda: _ForbiddenTurn(),
    )

    store = _DurableAsyncStore(db)
    runner = _wire_runner(monkeypatch, store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-stale",
        internal=False,
    )
    session = SimpleNamespace(
        session_id="session-1",
        session_key="telegram:chat-7:31",
    )
    try:
        response = asyncio.run(
            runner._handle_bot_chain_turn(
                event,
                session,
                session.session_key,
                parse_bot_chain_message(event.text),
            )
        )

        assert response == "$worker (final):\ndurable answer"
        assert model_turns == []
        receipt = db.get_bot_chain_delivery("session-1", "telegram-stale")
        assert receipt["state"] == "settled"
        assert receipt["outcome"] == "completed"
        # Resumed under the original chain identity, owned by the NEW claim.
        assert receipt["chain_name"] == conversation_name
        assert receipt["owner_token"] != old_token
    finally:
        db.close()


def test_gateway_claim_loss_mid_turn_cancels_and_stands_down(
    tmp_path, monkeypatch
):
    """When the execution claim is reclaimed mid-turn — a concurrent
    redelivery really takes over the receipt under a NEW owner_token — the
    stale owner cancels the chain, does not settle, writes no transcript
    rows, and returns no response. The REAL canonical-history publisher
    (publish_bot_chain_history, not a stub) must refuse to create/rename
    Bot Chat or publish the stale generation's output."""
    db = SessionDB(tmp_path / "ingress.db")
    db.create_session("session-1", source="telegram")

    profile = _profile(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    # Test-scale lease so the heartbeat fires immediately.
    monkeypatch.setattr(SessionDB, "BOT_CHAIN_CLAIM_LEASE_SECONDS", 0.3)

    class _ReclaimingStore(_DurableAsyncStore):
        def __init__(self, db):
            super().__init__(db)
            self.reclaimed_token = None

        async def renew_bot_chain_delivery_claim(
            self, session_id, message_id, owner_token
        ):
            if self.reclaimed_token is None:
                # A concurrent redelivery reclaims the claim for real: force
                # the lease expired, resume the admission, and win the fresh
                # claim under a new owner_token.
                import sqlite3

                with sqlite3.connect(self.db.db_path) as conn:
                    conn.execute(
                        "UPDATE bot_chain_deliveries SET lease_expires_at = 0 "
                        "WHERE session_id = ? AND platform_message_id = ?",
                        (session_id, message_id),
                    )
                assert (
                    self.db.admit_bot_chain_delivery(
                        session_id, message_id, "ignored-replacement-name"
                    )
                    == "admitted"
                )
                self.reclaimed_token = self.db.mark_bot_chain_delivery_running(
                    session_id, message_id
                )
                assert self.reclaimed_token
                assert self.reclaimed_token != owner_token
            return self.db.renew_bot_chain_delivery_claim(
                session_id, message_id, owner_token
            )

    store = _ReclaimingStore(db)
    publish_attempts = []

    def _execute(profile, prompt, control, *, conversation_name):
        # A slow in-flight model turn: it must observe the cancellation.
        assert control.cancel_event.wait(timeout=30), (
            "claim loss must cancel the running chain"
        )
        # The stale worker still holds its model output: durable rows exist
        # in the isolated chain session, and the turn reaches for the real
        # canonical-history publisher with the claim already lost.
        isolated_db = SessionDB(Path(profile.path) / "state.db")
        try:
            isolated_db.create_session(
                "chain-session",
                source="cli",
                model=profile.model,
                profile_name=profile.name,
            )
            isolated_db.set_session_title("chain-session", conversation_name)
            isolated_db.append_messages_batch(
                "chain-session",
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "stale output"},
                ],
            )
        finally:
            isolated_db.close()
        publish_attempts.append(conversation_name)
        with pytest.raises(BotChainCancelled):
            publish_bot_chain_history(profile, conversation_name, control=control)
        raise BotChainCancelled("Bot chain stopped.")

    monkeypatch.setattr(
        "agent.bot_chain.default_bot_turn_executor",
        lambda: _execute,
    )

    # Real asyncio.to_thread (NOT inlined): the heartbeat task must get
    # event-loop time while the turn runs in a worker thread.
    runner = object.__new__(GatewayRunner)
    runner.session_store = store._store
    runner._async_session_store = store
    state = SimpleNamespace(turn=SimpleNamespace(agent=None, started_ts=0.0))
    runner._session_state = lambda _key: state

    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-lost",
        internal=False,
    )
    session = SimpleNamespace(
        session_id="session-1",
        session_key="telegram:chat-7:31",
    )
    async def exercise():
        return await asyncio.wait_for(
            runner._handle_bot_chain_turn(
                event, session, session.session_key, parse_bot_chain_message(event.text)
            ),
            timeout=5,
        )

    try:
        response = asyncio.run(exercise())

        # The stale owner answered nothing and persisted nothing.
        assert response is None
        assert not db.has_platform_message_id("session-1", "telegram-lost")
        receipt = db.get_bot_chain_delivery("session-1", "telegram-lost")
        assert receipt is not None
        assert receipt["state"] == "running"  # left for the real owner
        assert receipt["owner_token"] == store.reclaimed_token
        assert receipt["outcome"] is None
        # The real publisher stood down: Bot Chat was neither created nor
        # renamed into, and the stale output was never projected anywhere.
        chain_name = receipt["chain_name"]
        assert publish_attempts == [chain_name]
        profile_db = SessionDB(Path(profile.path) / "state.db")
        try:
            assert profile_db.get_session_by_title("Bot Chat") is None
            isolated = profile_db.get_session_by_title(chain_name)
            assert isolated is not None  # never renamed into Bot Chat
            assert [
                (m["role"], m["content"])
                for m in profile_db.get_messages_as_conversation(
                    str(isolated["id"])
                )
            ] == [("user", "do the task"), ("assistant", "stale output")]
        finally:
            profile_db.close()
    finally:
        db.close()
