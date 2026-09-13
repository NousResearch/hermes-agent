"""Exception topics preserve destination policy and seed the session a reply resolves to."""

import asyncio
from concurrent.futures import Future
from copy import deepcopy
import socket
from types import SimpleNamespace

import pytest
import yaml

from cron import delivery_queue, incidents, jobs, scheduler
from cron.scheduler_delivery import _deliver_result
from cron.scheduler_prompt import _build_job_prompt
from gateway.config import Platform, load_gateway_config
from gateway.session import SessionSource, SessionStore, build_session_key


MARKER = "[CRON_BUSINESS_ERROR]"
BRIEF = "Mandatory input is unavailable; the report could not be completed."


class RecordingAdapter:
    supports_inchannel_continuable = True

    def __init__(self, store):
        self._session_store = store
        self.topics = []
        self.sent = []
        self.fail_create = False
        self.fail_send = False

    async def create_handoff_thread(self, chat_id, name):
        if self.fail_create:
            return None
        topic = str(9000 + len(self.topics))
        self.topics.append((chat_id, topic, name))
        return topic

    async def get_chat_info(self, chat_id):
        return {"type": "private" if int(chat_id) > 0 else "supergroup"}

    async def send(self, chat_id, content, *, metadata=None, **kwargs):
        self.sent.append((chat_id, content, metadata))
        if self.fail_send:
            return {"success": False, "error": "synthetic send failure"}
        return {"success": True, "message_id": str(len(self.sent))}


@pytest.fixture
def delivery_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER", raising=False)
    # No inherited home lanes, credentials or real network, including accidental fallback sends.
    from cron.scheduler_delivery import _HOME_TARGET_ENV_VARS
    for name in (*_HOME_TARGET_ENV_VARS.values(), "TELEGRAM_CRON_THREAD_ID"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name + "_THREAD_ID", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("network transport must be fake")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    cfg = {
        "cron": {"telegram_error_topics": True, "wrap_response": False},
        "platforms": {
            "telegram": {"enabled": True, "token": "synthetic", "home_channel": {
                "platform": "telegram", "chat_id": "12345", "thread_id": "71"}},
            "discord": {"enabled": True, "token": "synthetic", "home_channel": {
                "platform": "discord", "chat_id": "67890"}},
        },
    }

    def configure():
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))

    configure()
    store = SessionStore(tmp_path / "sessions", load_gateway_config())
    tg, discord = RecordingAdapter(store), RecordingAdapter(store)
    adapters = {Platform.TELEGRAM: tg, Platform.DISCORD: discord}
    # Run real router/adapter coroutines locally; no gateway service or socket wakeup is needed.
    loop = SimpleNamespace(is_running=lambda: True)

    def schedule(coro, loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except Exception as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", schedule)
    with jobs.use_cron_store(tmp_path):
        yield cfg, configure, store, adapters, loop
    store.close_all_db_handles()


@pytest.mark.parametrize("business", [False, True], ids=["engine", "business"])
@pytest.mark.parametrize("lane", ["telegram", "all", "origin", "explicit"])
@pytest.mark.parametrize("chat_id", ["12345", "-10012345"], ids=["dm", "forum"])
@pytest.mark.parametrize("queued", [False, True], ids=["live", "queued"])
def test_error_notifications_get_fresh_reply_context(
    delivery_env, business, lane, chat_id, queued, monkeypatch,
):
    cfg, configure, store, adapters, loop = delivery_env
    tg = adapters[Platform.TELEGRAM]
    cfg["platforms"]["telegram"]["home_channel"]["chat_id"] = chat_id
    # Even a configured flat surface and an old topic must yield a new topic on failure.
    cfg["platforms"]["telegram"]["extra"] = {"cron_continuable_surface": "in_channel"}
    configure()
    job = {
        "id": "abcdef123456", "name": "Synthetic report", "prompt": "Produce a report.",
        "deliver": f"telegram:{chat_id}:71" if lane == "explicit" else lane,
        "attach_to_session": False,
    }
    if lane == "origin":
        job["origin"] = {"platform": "telegram", "chat_id": chat_id,
                         "thread_id": "71", "chat_type": "dm" if int(chat_id) > 0 else "group"}
    # Business outcomes stay on deliver, even when the execution-failure lane is local.
    if business:
        job["failure_deliver"] = "local"
    original = deepcopy(job)
    content = MARKER + "\n" + BRIEF if business else BRIEF

    for run in range(2):
        if queued:
            execution_id = f"synthetic-{run}"
            job["execution_id"] = execution_id
            monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution_id)
            monkeypatch.setattr(delivery_queue, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0)
            assert _deliver_result(job, content, for_failure=not business) is None
            assert len(tg.topics) == run  # Only the gateway may open the topic.
            assert delivery_queue.drain(lambda j, c, f: _deliver_result(
                j, c, adapters=adapters, loop=loop, for_failure=f)) == 1
            assert delivery_queue.get_status(execution_id)["status"] == "delivered"
        else:
            assert _deliver_result(
                job, content, adapters=adapters, loop=loop, for_failure=not business) is None

        assert len(tg.topics) == run + 1
        sent_chat, sent_text, metadata = tg.sent[-1]
        topic = tg.topics[-1][1]
        assert sent_chat == chat_id and sent_text == BRIEF
        assert metadata["thread_id"] == topic and topic != "71"
        assert metadata.get("direct_messages_topic_id") is None
        assert metadata["telegram_dm_topic_created_for_send"] is True
        reply = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id,
                              chat_type="dm" if int(chat_id) > 0 else "thread",
                              thread_id=topic, user_id="reader")
        key = build_session_key(reply)
        assert key in store._entries, "reply must find the seeded session, not create an empty one"
        transcript = store.load_transcript(store._entries[key].session_id)
        assert any(row["role"] == "user" and BRIEF in row["content"] for row in transcript)
        assert all(MARKER not in row["content"] for row in transcript)

    assert tg.topics[0][1] != tg.topics[1][1]
    assert {k: job[k] for k in original} == original
    discord = adapters[Platform.DISCORD]
    assert len(discord.sent) == (2 if lane == "all" else 0)
    assert discord.topics == []
    assert MARKER in _build_job_prompt(job)


@pytest.mark.parametrize("case", [
    "normal", "quoted", "historical", "default_off", "disabled", "attached_normal",
    "local", "failure_local", "acked", "silent", "no_message", "business_silent",
    "business_no_message", "business_empty", "muted_push", "non_telegram",
    "create_failure", "send_failure", "standalone", "failure_override", "wrapped",
    "shared_topic_route", "relay",
    "native_engine", "native_business", "native_missing_topic", "empty_engine_notice",
])
def test_error_topics_preserve_existing_policy(delivery_env, case, monkeypatch):
    cfg, configure, store, adapters, loop = delivery_env
    tg, discord = adapters[Platform.TELEGRAM], adapters[Platform.DISCORD]
    job = {"id": "abcdef123456", "name": "Synthetic report", "prompt": "Produce a report.",
           "deliver": "telegram:12345", "attach_to_session": False}
    content, success, error = MARKER + "\n" + BRIEF, True, None
    expected_topics, expected_sends = 1, 1
    if case in {"normal", "attached_normal"}:
        content = BRIEF  # Error-like words alone must never classify the outcome.
        expected_topics = 0
    if case == "attached_normal":
        job["attach_to_session"] = True
        job["origin"] = {"platform": "telegram", "chat_id": "12345", "chat_type": "dm"}
        expected_topics = 1
    if case == "quoted":
        content, expected_topics = f"> {MARKER}\n{BRIEF}", 0
    if case == "historical":
        content, expected_topics = f"Previous run:\n{MARKER}\n{BRIEF}", 0
    if case == "default_off":
        cfg["cron"].pop("telegram_error_topics")
        expected_topics = 0
    if case == "disabled":
        cfg["cron"]["telegram_error_topics"] = False
        expected_topics = 0
    if case == "local":
        job["deliver"] = "local"
        expected_topics = expected_sends = 0
    if case in {"failure_local", "acked", "failure_override"}:
        success, error = False, "synthetic engine failure"
    if case == "failure_local":
        job["failure_deliver"] = "local"
        expected_topics = expected_sends = 0
    if case == "failure_override":
        job.update(deliver="discord:67890", failure_deliver="telegram:12345:71")
    if case == "acked":
        incident, _ = incidents.upsert_incident(job["id"], error)
        incidents.ack_incident(incident)
        expected_topics = expected_sends = 0
    silent = {"silent": "[SILENT]", "no_message": "NO_MESSAGE",
              "business_silent": MARKER + "\n[SILENT]\nNothing new.",
              "business_no_message": MARKER + "\nNO_MESSAGE",
              "business_empty": MARKER}
    if case in silent:
        content = silent[case]
        expected_topics = expected_sends = 0
    if case == "muted_push":
        cfg["cron"]["delivery"] = {"notify": False}
    if case == "non_telegram":
        job["deliver"] = "discord:67890"
        expected_topics = expected_sends = 0
    if case == "create_failure":
        tg.fail_create = True
        expected_topics = expected_sends = 0
    if case == "send_failure":
        tg.fail_send = True
    if case == "standalone":
        expected_topics = expected_sends = 0
    if case == "empty_engine_notice":
        expected_topics = expected_sends = 0
    if case == "wrapped":
        cfg["cron"]["wrap_response"] = True
    if case == "shared_topic_route":
        from cron.scheduler_preflight import SharedRouteAdapters
        from gateway.profile_routing import ProfileRoute

        job["deliver"] = "telegram:12345:71"
        adapters = SharedRouteAdapters(adapters, [ProfileRoute(
            name="topic", platform="telegram", chat_id="12345", thread_id="71", profile="test")])
        expected_topics = expected_sends = 0
    if case == "relay":
        cfg["platforms"]["telegram"]["enabled"] = False
        cfg["platforms"]["relay"] = {"enabled": True}
        tg.fronts_platform = lambda platform: platform == Platform.TELEGRAM
        adapters = {Platform.RELAY: tg}
        expected_topics = expected_sends = 0
    native = case.startswith("native_")
    if native:
        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter
        from telegram.error import BadRequest

        # Exercise the native adapter down to Telegram API kwargs, without connecting a bot.
        tg = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
        tg._session_store = store
        tg.topics, tg.sent = [], []
        chat_id = "12345" if case == "native_engine" else "-10012345"
        job["deliver"] = f"telegram:{chat_id}:71"
        if case == "native_engine":
            success, error = False, "synthetic engine failure"

        async def create_forum_topic(**kwargs):
            tg.topics.append((str(kwargs["chat_id"]), "9000", kwargs["name"]))
            return SimpleNamespace(message_thread_id=9000)

        async def send_message(**kwargs):
            tg.sent.append((str(kwargs["chat_id"]), kwargs["text"], {
                "thread_id": str(kwargs.get("message_thread_id")),
                "notify": not kwargs.get("disable_notification", False),
            }))
            if case == "native_missing_topic":
                raise BadRequest("Message thread not found")
            return SimpleNamespace(message_id=100)

        tg._bot = SimpleNamespace(create_forum_topic=create_forum_topic, send_message=send_message)
        adapters[Platform.TELEGRAM] = tg
    configure()
    original = deepcopy(job)
    fallback = []

    async def standalone(*args, **kwargs):
        fallback.append((args, kwargs))
        return {"success": True, "message_id": "fake"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", standalone)
    # Isolate the optional LLM summarizer; all delivery, ack, config, session and file I/O is real.
    monkeypatch.setattr(scheduler, "_summarize_cron_failure_for_delivery", lambda j, e: BRIEF)
    outcome = scheduler._RunDelivery(job=job, success=success, error=error)
    if case == "empty_engine_notice":
        outcome.delivery_error = _deliver_result(
            job, "   ", adapters=adapters, loop=loop, for_failure=True)
    else:
        scheduler._save_compose_deliver(
            outcome, scheduler._FireOwnership(job, None), content, content,
            adapters=None if case == "standalone" else adapters,
            loop=None if case == "standalone" else loop, verbose=False, execution_token=None,
        )
    assert len(tg.topics) == expected_topics
    assert len(tg.sent) == expected_sends
    assert fallback == [], "exception topic failures must never open a standalone/root fallback"
    assert bool(outcome.delivery_error) == (case in {
        "create_failure", "send_failure", "standalone", "shared_topic_route", "relay",
        "native_missing_topic", "empty_engine_notice"})
    assert {k: job[k] for k in original} == original
    if tg.sent:
        expected_content = content if case in {"quoted", "historical"} else BRIEF
        if native:
            expected_content = tg.format_message(expected_content)
            assert tg.sent[0][2]["thread_id"] == tg.topics[0][1]
            reply = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id,
                                  chat_type="dm" if int(chat_id) > 0 else "thread",
                                  thread_id="9000", user_id="reader")
            if case == "native_missing_topic":
                assert build_session_key(reply) not in store._entries
            else:
                entry = store._entries[build_session_key(reply)]
                assert any(BRIEF in row["content"] for row in store.load_transcript(entry.session_id))
        if case == "wrapped":
            assert BRIEF in tg.sent[0][1] and MARKER not in tg.sent[0][1]
        else:
            assert tg.sent[0][1] == expected_content
        assert tg.sent[0][2]["notify"] is (case != "muted_push")
    assert len(discord.sent) == (1 if case == "non_telegram" else 0)
    assert discord.topics == []
    if discord.sent:
        assert discord.sent[0][1] == BRIEF
    prompt = _build_job_prompt(job)
    assert (MARKER in prompt) is (case not in {"default_off", "disabled"})
