from __future__ import annotations

import asyncio


import pytest

IDS = {
    "Demian": "UDEMIAN01",
    "Hans": "UHANS0001",
    "Wendy": "UWENDY001",
    "Tesla": "UTESLA001",
    "Turing": "UTURING01",
    "Mason": "UMASON001",
    "Watson": "UWATSON01",
}

def _base_mapping():
    return {
        "enabled": True,
        "mode": "work",
        "channel_allowlist": ["C0BNBDNC745"],
        "sinclair_user_id": "U0BC6GDPRAQ",
        "bot_registry": IDS,
        "db_path": "/tmp/tsk60-router.db",
    }


def test_lounge_spontaneous_flag_defaults_disabled():
    from gateway.work_router.config import RouterConfig

    config = RouterConfig.from_mapping(_base_mapping())

    assert config.lounge_spontaneous_enabled is False


@pytest.mark.parametrize(
    "enabled",
    [True, False],
)
def test_lounge_spontaneous_flag_is_explicit(enabled):
    from gateway.work_router.config import RouterConfig

    mapping = _base_mapping()
    mapping["lounge_spontaneous"] = {"enabled": enabled}

    config = RouterConfig.from_mapping(mapping)

    assert config.lounge_spontaneous_enabled is enabled


@pytest.mark.parametrize(
    "text",
    [
        "해야지",
        "대응해야지",
        "이건 진행해야지",
        "메일을 작성해야지",
        "검토해보자",
        "이거 어떻게 생각해?",
        "이 문제 한번 보자",
    ],
)
def test_discussion_language_does_not_become_execution(text):
    from gateway.work_router.models import classify_lounge_execution_intent

    result = classify_lounge_execution_intent(text)

    assert result.kind == "discussion"


@pytest.mark.parametrize(
    "text",
    [
        "진행해",
        "착수해",
        "업무요청으로 넘겨",
        "메일 작성해줘",
        "이 내용 수정해줘",
        "관련 자료 찾아줘",
    ],
)
def test_explicit_execution_language_enters_execution_path(text):
    from gateway.work_router.models import classify_lounge_execution_intent

    result = classify_lounge_execution_intent(text)

    assert result.kind == "execute"


@pytest.mark.parametrize(
    "text",
    [
        "이거 해",
        "처리 가능?",
        "한번 해볼까",
    ],
)
def test_ambiguous_language_fails_closed(text):
    from gateway.work_router.models import classify_lounge_execution_intent

    result = classify_lounge_execution_intent(text)

    assert result.kind == "clarify"


def test_execution_classifier_is_deterministic():
    from gateway.work_router.models import classify_lounge_execution_intent

    text = "이 내용 수정해줘"

    first = classify_lounge_execution_intent(text)
    second = classify_lounge_execution_intent(text)

    assert first == second


def test_plain_lounge_conversation_remains_discussion():
    from gateway.work_router.models import classify_lounge_execution_intent

    assert classify_lounge_execution_intent("오늘 분위기 괜찮네").kind == "discussion"


LOUNGE_CHANNEL_ID = "C0BNBDNC745"
SINCLAIR_USER_ID = "U0BC6GDPRAQ"


def _service_config(tmp_path, *, spontaneous_enabled):
    from gateway.work_router.config import RouterConfig

    mapping = _base_mapping()
    mapping["db_path"] = str(tmp_path / "router.db")
    mapping["channel_allowlist"] = [LOUNGE_CHANNEL_ID, "C0BDCURRS9W"]
    mapping["lounge_spontaneous"] = {"enabled": spontaneous_enabled}
    return RouterConfig.from_mapping(mapping)


def _lounge_event(event_id, text, **overrides):
    from gateway.work_router.models import CanonicalEvent

    values = {
        "event_id": event_id,
        "channel_id": LOUNGE_CHANNEL_ID,
        "thread_ts": "1789000000.000100",
        "text": text,
        "author_kind": "human",
        "author_user_id": SINCLAIR_USER_ID,
        "metadata": {
            "slack_team_id": "TWORKCOMPANY",
            "slack_thread_context": "Sinclair: 앞선 Lounge 공유 대화",
        },
    }
    values.update(overrides)
    return CanonicalEvent(**values)


async def _process(router, event, *, sender=None):
    assert router.store.enqueue_after_ack(event, ack_completed=True)
    resolved_sender = sender
    if resolved_sender is None:
        async def successful_sender(**kwargs):
            return {"success": True, "message_id": f"sent-{event.event_id}"}
        resolved_sender = successful_sender
    return await router.process_once(
        event.event_id,
        worker_id=f"worker-{event.event_id}",
        sender=resolved_sender,
    )


def test_flag_off_keeps_existing_route_and_starts_no_candidate(tmp_path):
    from gateway.work_router.rules import route
    from gateway.work_router.service import WorkRouter

    calls = []
    router = WorkRouter(
        _service_config(tmp_path, spontaneous_enabled=False),
        lounge_candidate_submitter=lambda **kwargs: calls.append(kwargs),
    )
    event = _lounge_event("EvTsk60FlagOff", "이 문제 검토해보자")
    before = router.store.get_thread(event.channel_id, event.thread_ts)
    expected = route(before, event, router.registry, handoff_count=0)

    result = asyncio.run(_process(router, event))

    assert result.decision == expected
    assert calls == []
    assert router.store.get_work_execution(event.event_id) is None


def test_explicit_staff_call_keeps_existing_route_and_shared_context(tmp_path):
    from gateway.work_router.rules import route
    from gateway.work_router.service import WorkRouter

    calls = []
    router = WorkRouter(
        _service_config(tmp_path, spontaneous_enabled=True),
        lounge_candidate_submitter=lambda **kwargs: calls.append(kwargs),
    )
    event = _lounge_event(
        "EvTsk60Single",
        "Hans 이 견적 관점만 봐줘",
        mentioned_user_ids=(IDS["Hans"],),
    )
    before = router.store.get_thread(event.channel_id, event.thread_ts)
    expected = route(before, event, router.registry, handoff_count=0)

    result = asyncio.run(_process(router, event))

    assert result.decision is not None
    assert result.decision.action.kind == expected.action.kind
    assert result.decision.action.target_profile == "Hans"
    assert "앞선 Lounge 공유 대화" in result.decision.action.context
    assert calls == []


def test_explicit_multi_team_call_keeps_existing_route(tmp_path):
    from gateway.work_router.rules import route
    from gateway.work_router.service import WorkRouter

    calls = []
    router = WorkRouter(
        _service_config(tmp_path, spontaneous_enabled=True),
        lounge_candidate_submitter=lambda **kwargs: calls.append(kwargs),
    )
    event = _lounge_event(
        "EvTsk60Multi",
        "Hans Tesla 둘 다 의견 줘",
        mentioned_user_ids=(IDS["Hans"], IDS["Tesla"]),
    )
    before = router.store.get_thread(event.channel_id, event.thread_ts)
    expected = route(before, event, router.registry, handoff_count=0)

    result = asyncio.run(_process(router, event))

    assert result.decision.action.kind == "multi_team_dispatch"
    assert result.decision.action.target_profile == expected.action.target_profile
    assert result.decision.action.content == expected.action.content
    assert "앞선 Lounge 공유 대화" in result.decision.action.context
    assert calls == []


def test_everyone_call_keeps_existing_multi_team_route(tmp_path):
    from gateway.work_router.service import WorkRouter

    calls = []
    router = WorkRouter(
        _service_config(tmp_path, spontaneous_enabled=True),
        lounge_candidate_submitter=lambda **kwargs: calls.append(kwargs),
    )
    event = _lounge_event("EvTsk60Everyone", "모두들 자기 의견 줘")

    result = asyncio.run(_process(router, event))

    assert result.decision.action.kind == "multi_team_dispatch"
    assert calls == []


def test_meeting_control_precedes_lounge_admission(tmp_path):
    from gateway.work_router.service import WorkRouter

    calls = []
    router = WorkRouter(
        _service_config(tmp_path, spontaneous_enabled=True),
        lounge_candidate_submitter=lambda **kwargs: calls.append(kwargs),
    )
    event = _lounge_event(
        "EvTsk60Meeting",
        "회의 시작 안건: Lounge 정책 검토",
        channel_id="C0BDCURRS9W",
    )

    result = asyncio.run(_process(router, event))

    assert result.decision.action.kind == "meeting_preflight"
    assert calls == []


def test_unnamed_discussion_submits_candidate_without_blocking_demian(tmp_path):
    from gateway.work_router.service import WorkRouter

    async def scenario():
        child_started = asyncio.Event()
        release_child = asyncio.Event()
        candidate_calls = []
        sends = []

        async def slow_child(**kwargs):
            candidate_calls.append(kwargs)
            child_started.set()
            await release_child.wait()

        async def sender(**kwargs):
            sends.append(kwargs)
            return {"success": True, "message_id": "demian-fast-lane"}

        router = WorkRouter(
            _service_config(tmp_path, spontaneous_enabled=True),
            lounge_candidate_submitter=slow_child,
        )
        event = _lounge_event("EvTsk60Discussion", "이 문제 검토해보자")

        result = await asyncio.wait_for(
            _process(router, event, sender=sender),
            timeout=2.0,
        )
        await asyncio.wait_for(child_started.wait(), timeout=2.0)

        assert not release_child.is_set()
        assert result.decision.action.target_profile == "Demian"
        assert len(sends) == 1
        assert len(candidate_calls) == 1
        assert "앞선 Lounge 공유 대화" in candidate_calls[0]["context"]
        assert router.store.get_work_execution(event.event_id) is None

        release_child.set()
        await asyncio.sleep(0)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "text",
    [
        "진행해",
        "착수해",
        "업무요청으로 넘겨",
        "메일 작성해줘",
        "이 내용 수정해줘",
        "관련 자료 찾아줘",
    ],
)
def test_service_execution_uses_existing_path_without_candidate(tmp_path, text):
    from gateway.work_router.rules import route
    from gateway.work_router.service import WorkRouter

    calls = []
    router = WorkRouter(
        _service_config(tmp_path, spontaneous_enabled=True),
        lounge_candidate_submitter=lambda **kwargs: calls.append(kwargs),
    )
    event = _lounge_event(f"EvTsk60Execute{text}", text)
    before = router.store.get_thread(event.channel_id, event.thread_ts)
    expected = route(before, event, router.registry, handoff_count=0)

    result = asyncio.run(_process(router, event))

    assert result.decision == expected
    assert calls == []


@pytest.mark.parametrize("text", ["이거 해", "처리 가능?", "한번 해볼까"])
def test_service_clarifies_once_and_creates_no_work(tmp_path, text):
    from gateway.work_router.service import WorkRouter

    router = WorkRouter(_service_config(tmp_path, spontaneous_enabled=True))
    event = _lounge_event(f"EvTsk60Clarify{text}", text)

    first = asyncio.run(_process(router, event))
    second = asyncio.run(
        router.process_once(
            event.event_id,
            worker_id=f"worker-replay-{event.event_id}",
            sender=lambda **kwargs: None,
        )
    )

    assert first.decision.action.response_kind == "lounge_execution_clarification"
    assert first.decision.action.target_profile == "Demian"
    assert "진행" in first.decision.action.content
    assert second.status == "duplicate_or_not_pending"
    assert router.store.get_work_execution(event.event_id) is None


def test_bot_echo_never_submits_lounge_candidate(tmp_path):
    from gateway.work_router.service import WorkRouter

    calls = []
    router = WorkRouter(
        _service_config(tmp_path, spontaneous_enabled=True),
        lounge_candidate_submitter=lambda **kwargs: calls.append(kwargs),
    )
    event = _lounge_event(
        "EvTsk60BotEcho",
        "검토해보자",
        author_kind="bot",
        author_profile="Hans",
        author_user_id=IDS["Hans"],
    )

    result = asyncio.run(_process(router, event))

    assert result.decision.action.kind == "silence"
    assert calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"author_user_id": "UOTHER001"},
        {"channel_id": "COTHER001"},
    ],
)
def test_untrusted_principal_or_channel_never_submits_candidate(tmp_path, overrides):
    from gateway.work_router.service import WorkRouter

    calls = []
    router = WorkRouter(
        _service_config(tmp_path, spontaneous_enabled=True),
        lounge_candidate_submitter=lambda **kwargs: calls.append(kwargs),
    )
    event = _lounge_event("EvTsk60Untrusted", "검토해보자", **overrides)

    asyncio.run(_process(router, event))

    assert calls == []


def test_unsupported_lounge_subtype_is_silent(tmp_path):
    from gateway.work_router.service import WorkRouter

    calls = []
    router = WorkRouter(
        _service_config(tmp_path, spontaneous_enabled=True),
        lounge_candidate_submitter=lambda **kwargs: calls.append(kwargs),
    )
    event = router.canonicalize_slack_event(
        {
            "channel": LOUNGE_CHANNEL_ID,
            "ts": "1789000000.000200",
            "text": "검토해보자",
            "user": SINCLAIR_USER_ID,
            "subtype": "message_changed",
        },
        {"event_id": "EvTsk60Unsupported", "team_id": "TWORKCOMPANY"},
    )
    assert event is not None

    result = asyncio.run(_process(router, event))

    assert result.decision.action.kind == "silence"
    assert result.decision.action.reason == "unsupported_lounge_event_subtype"
    assert calls == []
