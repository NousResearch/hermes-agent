"""Contracts for non-blocking decisions emitted by autonomous cron runs."""

import json
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _envelope(cards, visible="Completed analysis."):
    payload = json.dumps({"version": 1, "cards": cards}, ensure_ascii=False)
    return f"{visible}\n\n```hermes-deferred-decisions\n{payload}\n```"


def _session_binding(
    *,
    chat_id="chat-a",
    chat_type="group",
    user_id="user-a",
    thread_id=None,
):
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        thread_id=thread_id,
    )
    return source.to_dict(), build_session_key(source)


def _stored_records(tmp_path):
    path = tmp_path / "cron" / "deferred_decisions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def test_parser_strips_one_valid_terminal_control_block():
    from cron.deferred_decisions import parse_deferred_decisions

    parsed = parse_deferred_decisions(
        _envelope([
            {
                "question": "Which rollout should continue?",
                "choices": ["Canary", "Regional", "Pause"],
            },
            {
                "question": "When should the next review run?",
                "choices": ["In one hour", "Tomorrow"],
            },
        ])
    )

    assert parsed is not None
    assert parsed.visible_text == "Completed analysis."
    assert [card.question for card in parsed.cards] == [
        "Which rollout should continue?",
        "When should the next review run?",
    ]
    assert parsed.cards[0].choices == ("Canary", "Regional", "Pause")


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 2, "cards": [{"question": "Q?", "choices": ["A", "B"]}]},
        {"version": 1, "cards": []},
        {"version": 1, "cards": [{"question": "Q?", "choices": ["A"]}]},
        {"version": 1, "cards": [{"question": "Q?", "choices": ["A", "A"]}]},
        {"version": 1, "cards": [{"question": "Q?", "choices": ["A", "B"], "extra": True}]},
        {"version": 1, "cards": [{"question": " Q?", "choices": ["A", "B"]}]},
        {"version": 1, "cards": [{"question": "Q?", "choices": ["A\nB", "C"]}]},
        {"version": 1, "cards": [{"question": "Q" * 501, "choices": ["A", "B"]}]},
        {"version": 1, "cards": [{"question": "Q?", "choices": ["A" * 101, "B"]}]},
        {
            "version": 1,
            "cards": [
                {"question": f"Q{i}?", "choices": ["A", "B"]}
                for i in range(4)
            ],
        },
    ],
)
def test_parser_rejects_malformed_or_unbounded_payloads(payload):
    from cron.deferred_decisions import parse_deferred_decisions

    raw = _envelope(payload.get("cards", []))
    raw = raw.replace(
        json.dumps({"version": 1, "cards": payload.get("cards", [])}, ensure_ascii=False),
        json.dumps(payload, ensure_ascii=False),
    )
    assert parse_deferred_decisions(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        "Ordinary cron output.",
        "```hermes-deferred-decisions\nnot json\n```",
        _envelope([{"question": "Q?", "choices": ["A", "B"]}]) + "\ntrailing text",
        _envelope([{"question": "Q?", "choices": ["A", "B"]}]).replace(
            "```hermes-deferred-decisions", "```json"
        ),
    ],
)
def test_parser_leaves_non_protocol_output_untouched(raw):
    from cron.deferred_decisions import parse_deferred_decisions

    assert parse_deferred_decisions(raw) is None


def test_explicit_origin_attach_and_live_capability_are_all_required():
    from cron.deferred_decisions import delivery_is_eligible

    origin = {"platform": "telegram", "chat_id": "chat-a", "thread_id": "topic-a"}
    target = {"platform": "telegram", "chat_id": "chat-a", "thread_id": "topic-a"}
    capable = type("Capable", (), {"send_deferred_decision": lambda self: None})()

    assert delivery_is_eligible(
        {"deliver": "origin", "origin": origin, "attach_to_session": True},
        target,
        capable,
    )
    assert not delivery_is_eligible(
        {"deliver": "origin", "origin": origin, "attach_to_session": False},
        target,
        capable,
    )
    assert not delivery_is_eligible(
        {"deliver": "telegram", "origin": origin, "attach_to_session": True},
        target,
        capable,
    )
    assert not delivery_is_eligible(
        {"deliver": "origin", "origin": origin, "attach_to_session": True},
        {**target, "thread_id": "topic-b"},
        capable,
    )
    assert not delivery_is_eligible(
        {"deliver": "origin", "origin": origin, "attach_to_session": True},
        target,
        object(),
    )


def test_only_explicitly_continuable_cron_prompts_receive_protocol_guidance():
    from cron.scheduler import _build_job_prompt

    ordinary = _build_job_prompt({"id": "a1b2c3d4e5f6", "prompt": "Run report."})
    continuable = _build_job_prompt({
        "id": "a1b2c3d4e5f6",
        "prompt": "Run report.",
        "attach_to_session": True,
    })

    assert "hermes-deferred-decisions" not in ordinary
    assert "hermes-deferred-decisions" in continuable
    assert "Finish the scheduled work autonomously" in continuable


def test_durable_choice_claim_survives_reconstruction_and_is_single_use(tmp_path, monkeypatch):
    from cron import deferred_decisions as decisions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = datetime.now(timezone.utc)
    session_source, session_key = _session_binding(thread_id="topic-a")
    record = decisions.DeferredDecisionRecord(
        job_id="a1b2c3d4e5f6",
        decision_id="1122334455667788",
        job_name="Deployment review",
        card_index=0,
        platform="telegram",
        chat_id="chat-a",
        thread_id="topic-a",
        user_id="user-a",
        question="Proceed?",
        choices=("Proceed", "Pause"),
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=1)).isoformat(),
        context_ready=True,
        session_source=session_source,
        session_key=session_key,
        message_id="message-a",
    )
    decisions.save_records([record])

    # Clear only process memory: the second read must reconstruct from disk.
    decisions._reset_for_tests()
    claimed = decisions.claim_choice(
        job_id=record.job_id,
        decision_id=record.decision_id,
        card_index=0,
        choice_index=1,
        platform="telegram",
        chat_id="chat-a",
        thread_id="topic-a",
        user_id="user-a",
        message_id="message-a",
    )

    assert claimed is not None
    assert claimed.choice == "Pause"
    assert claimed.record.job_name == "Deployment review"
    assert decisions.acknowledge_choice(claimed)
    assert decisions.claim_choice(
        job_id=record.job_id,
        decision_id=record.decision_id,
        card_index=0,
        choice_index=1,
        platform="telegram",
        chat_id="chat-a",
        thread_id="topic-a",
        user_id="user-a",
        message_id="message-a",
    ) is None


def test_only_one_concurrent_claim_wins_and_release_returns_it_to_pending():
    from cron import deferred_decisions as decisions

    session_source, session_key = _session_binding()
    record = decisions.register_cards(
        job={"id": "a1b2c3d4e5f6", "name": "Review"},
        cards=[decisions.DeferredDecisionCard("Proceed?", ("Proceed", "Pause"))],
        platform="telegram",
        chat_id="chat-a",
        thread_id=None,
        user_id="user-a",
        context_ready=True,
        session_source=session_source,
        session_key=session_key,
    )[0]
    assert decisions.bind_message(record, "message-a")
    claim_args = {
        "job_id": record.job_id,
        "decision_id": record.decision_id,
        "card_index": 0,
        "choice_index": 0,
        "platform": "telegram",
        "chat_id": "chat-a",
        "thread_id": None,
        "user_id": "user-a",
        "message_id": "message-a",
    }

    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _index: decisions.claim_choice(**claim_args), range(8)))

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert decisions.release_choice(winners[0])
    reclaimed = decisions.claim_choice(**claim_args)
    assert reclaimed is not None
    assert decisions.acknowledge_choice(reclaimed)


def test_pending_record_pressure_rejects_new_cards_without_evicting_old_pending(
    monkeypatch,
):
    from cron import deferred_decisions as decisions

    monkeypatch.setattr(decisions, "MAX_PENDING_RECORDS", 5)
    session_source, session_key = _session_binding()
    records = []
    for index in range(5):
        registered = decisions.register_cards(
            job={"id": f"{index + 1:012x}", "name": f"Review {index}"},
            cards=[
                decisions.DeferredDecisionCard(
                    f"Proceed with {index}?", ("Proceed", "Pause")
                )
            ],
            platform="telegram",
            chat_id="chat-a",
            thread_id=None,
            user_id="user-a",
            context_ready=True,
            session_source=session_source,
            session_key=session_key,
        )
        assert len(registered) == 1
        assert decisions.bind_message(registered[0], f"message-{index}")
        records.extend(registered)

    overflow = decisions.register_cards(
        job={"id": "ffffffffffff", "name": "Overflow"},
        cards=[decisions.DeferredDecisionCard("Overflow?", ("Proceed", "Pause"))],
        platform="telegram",
        chat_id="chat-a",
        thread_id=None,
        user_id="user-a",
        context_ready=True,
        session_source=session_source,
        session_key=session_key,
    )

    assert overflow == []
    oldest = records[0]
    claimed = decisions.claim_choice(
        job_id=oldest.job_id,
        decision_id=oldest.decision_id,
        card_index=0,
        choice_index=1,
        platform="telegram",
        chat_id="chat-a",
        thread_id=None,
        user_id="user-a",
        message_id="message-0",
    )
    assert claimed is not None
    assert claimed.choice == "Pause"
    assert decisions.acknowledge_choice(claimed)


def test_corrupt_durable_record_types_fail_closed(tmp_path, monkeypatch):
    from cron import deferred_decisions as decisions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_path = tmp_path / "cron" / "deferred_decisions.json"
    state_path.parent.mkdir(parents=True)
    session_source, session_key = _session_binding()
    state_path.write_text(
        json.dumps({
            "version": 1,
            "records": [{
                "job_id": 123,
                "decision_id": "1122334455667788",
                "job_name": "Review",
                "card_index": 0,
                "platform": "telegram",
                "chat_id": "chat-a",
                "thread_id": None,
                "user_id": "user-a",
                "question": "Proceed?",
                "choices": ["Proceed", "Pause"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                "context_ready": True,
                "session_source": session_source,
                "session_key": session_key,
                "message_id": None,
                "claimed": False,
                "claim_token": None,
                "claim_expires_at": None,
            }],
        }),
        encoding="utf-8",
    )

    assert decisions.claim_choice(
        job_id="a1b2c3d4e5f6",
        decision_id="1122334455667788",
        card_index=0,
        choice_index=0,
        platform="telegram",
        chat_id="chat-a",
        thread_id=None,
        user_id="user-a",
    ) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"job_id": "ffffffffffff"},
        {"decision_id": "ffffffffffffffff"},
        {"card_index": 1},
        {"choice_index": 9},
        {"platform": "discord"},
        {"chat_id": "chat-b"},
        {"thread_id": "topic-b"},
        {"user_id": "user-b"},
    ],
)
def test_tampered_choice_claim_fails_closed(tmp_path, monkeypatch, overrides):
    from cron import deferred_decisions as decisions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = datetime.now(timezone.utc)
    session_source, session_key = _session_binding()
    record = decisions.DeferredDecisionRecord(
        job_id="a1b2c3d4e5f6",
        decision_id="1122334455667788",
        job_name="Review",
        card_index=0,
        platform="telegram",
        chat_id="chat-a",
        thread_id=None,
        user_id="user-a",
        question="Proceed?",
        choices=("Proceed", "Pause"),
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=1)).isoformat(),
        context_ready=True,
        session_source=session_source,
        session_key=session_key,
        message_id="message-a",
    )
    decisions.save_records([record])
    args = {
        "job_id": record.job_id,
        "decision_id": record.decision_id,
        "card_index": 0,
        "choice_index": 0,
        "platform": "telegram",
        "chat_id": "chat-a",
        "thread_id": None,
        "user_id": "user-a",
        "message_id": "message-a",
    }
    args.update(overrides)
    assert decisions.claim_choice(**args) is None


def test_expired_or_unmirrored_record_fails_closed(tmp_path, monkeypatch):
    from cron import deferred_decisions as decisions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = datetime.now(timezone.utc)
    session_source, session_key = _session_binding()
    for context_ready, expires_at in (
        (False, now + timedelta(hours=1)),
        (True, now - timedelta(seconds=1)),
    ):
        record = decisions.DeferredDecisionRecord(
            job_id="a1b2c3d4e5f6",
            decision_id="1122334455667788",
            job_name="Review",
            card_index=0,
            platform="telegram",
            chat_id="chat-a",
            thread_id=None,
            user_id="user-a",
            question="Proceed?",
            choices=("Proceed", "Pause"),
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            context_ready=context_ready,
            session_source=session_source,
            session_key=session_key,
            message_id="message-a",
        )
        decisions.save_records([record])
        assert decisions.claim_choice(
            job_id=record.job_id,
            decision_id=record.decision_id,
            card_index=0,
            choice_index=0,
            platform="telegram",
            chat_id="chat-a",
            thread_id=None,
            user_id="user-a",
            message_id="message-a",
        ) is None


def test_callback_data_is_bounded_for_every_valid_index():
    from cron.deferred_decisions import callback_data

    for card_index in range(3):
        for choice_index in range(4):
            value = callback_data(
                "a1b2c3d4e5f6", "1122334455667788", card_index, choice_index
            )
            assert len(value.encode("utf-8")) <= 64
            assert value == f"cd:a1b2c3d4e5f6:1122334455667788:{card_index}:{choice_index}"


def test_recurring_deliveries_keep_callbacks_bound_to_their_own_choices(
    tmp_path, monkeypatch
):
    from cron import deferred_decisions as decisions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {"id": "a1b2c3d4e5f6", "name": "Review"}
    session_source, session_key = _session_binding()
    first = decisions.register_cards(
        job=job,
        cards=[decisions.DeferredDecisionCard("First?", ("First yes", "First no"))],
        platform="telegram",
        chat_id="chat-a",
        thread_id=None,
        user_id="user-a",
        context_ready=True,
        session_source=session_source,
        session_key=session_key,
    )[0]
    second = decisions.register_cards(
        job=job,
        cards=[decisions.DeferredDecisionCard("Second?", ("Second yes", "Second no"))],
        platform="telegram",
        chat_id="chat-a",
        thread_id=None,
        user_id="user-a",
        context_ready=True,
        session_source=session_source,
        session_key=session_key,
    )[0]
    assert decisions.bind_message(first, "message-first")
    assert decisions.bind_message(second, "message-second")

    assert first.decision_id != second.decision_id
    old_claim = decisions.claim_choice(
        job_id=job["id"],
        decision_id=first.decision_id,
        card_index=0,
        choice_index=1,
        platform="telegram",
        chat_id="chat-a",
        thread_id=None,
        user_id="user-a",
        message_id="message-first",
    )
    new_claim = decisions.claim_choice(
        job_id=job["id"],
        decision_id=second.decision_id,
        card_index=0,
        choice_index=1,
        platform="telegram",
        chat_id="chat-a",
        thread_id=None,
        user_id="user-a",
        message_id="message-second",
    )
    assert old_claim is not None and old_claim.choice == "First no"
    assert new_claim is not None and new_claim.choice == "Second no"


def _run_live_delivery(
    content,
    *,
    mirror_ok=True,
    card_ok=True,
    attach=True,
    loop_mode="running",
    main_ok=True,
):
    from cron.scheduler import _deliver_result
    from gateway.config import Platform
    from gateway.platforms.base import SendResult

    adapter = MagicMock(spec=["send", "send_deferred_decision"])
    adapter.send = AsyncMock(
        return_value=SendResult(
            success=main_ok,
            message_id="m1" if main_ok else None,
            error=None if main_ok else "main send failed",
        )
    )
    adapter.send_deferred_decision = AsyncMock(
        return_value=SendResult(success=card_ok, message_id="c1" if card_ok else None)
    )
    pconfig = MagicMock(enabled=True, extra={})
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: pconfig}
    loop = None if loop_mode == "missing" else MagicMock()
    if loop is not None:
        loop.is_running.return_value = loop_mode == "running"
    standalone_calls = []

    async def standalone_send(
        platform,
        platform_config,
        chat_id,
        text,
        *,
        thread_id=None,
        media_files=None,
    ):
        standalone_calls.append({
            "platform": platform,
            "config": platform_config,
            "chat_id": chat_id,
            "text": text,
            "thread_id": thread_id,
            "media_files": media_files,
        })
        return {}

    def run_now(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 - preserve future behavior
            future.set_exception(exc)
        return future

    session_source, session_key = _session_binding(
        chat_id="1200",
        chat_type="group",
        user_id="700",
        thread_id="44",
    )
    job = {
        "id": "a1b2c3d4e5f6",
        "name": "Release review",
        "deliver": "origin",
        "origin": {**session_source, "session_key": session_key},
        "attach_to_session": attach,
    }
    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}
    ), patch(
        "asyncio.run_coroutine_threadsafe", side_effect=run_now
    ), patch(
        "gateway.mirror.mirror_to_session", return_value=mirror_ok
    ) as mirror_mock, patch(
        "tools.send_message_tool._send_to_platform", side_effect=standalone_send
    ), patch(
        "cron.scheduler._interpreter_shutting_down", return_value=False
    ):
        result = _deliver_result(
            job,
            content,
            adapters={Platform.TELEGRAM: adapter},
            loop=loop,
        )
    return result, adapter, mirror_mock.call_args, standalone_calls


def test_delivery_strips_protocol_mirrors_clean_output_then_sends_cards():
    content = _envelope([
        {"question": "Proceed?", "choices": ["Proceed", "Pause"]},
        {"question": "Review when?", "choices": ["Today", "Tomorrow"]},
    ])

    result, adapter, mirror_call, standalone_calls = _run_live_delivery(content)

    assert result is None
    assert standalone_calls == []
    first_text = adapter.send.await_args_list[0].args[1]
    assert first_text == "Completed analysis."
    assert "hermes-deferred-decisions" not in first_text
    assert "Decision: Proceed?" in mirror_call.args[2]
    assert "1. Proceed" in mirror_call.args[2]
    assert "hermes-deferred-decisions" not in mirror_call.args[2]
    assert adapter.send_deferred_decision.await_count == 2
    assert [
        call.kwargs["question"]
        for call in adapter.send_deferred_decision.await_args_list
    ] == ["Proceed?", "Review when?"]


def test_unavailable_context_falls_back_to_numbered_text_and_no_buttons():
    content = _envelope([
        {"question": "Proceed?", "choices": ["Proceed", "Pause"]},
    ])

    result, adapter, _, standalone_calls = _run_live_delivery(
        content, mirror_ok=False
    )

    assert result is None
    assert standalone_calls == []
    adapter.send_deferred_decision.assert_not_awaited()
    assert adapter.send.await_count == 2
    fallback = adapter.send.await_args_list[1].args[1]
    assert "Decision: Proceed?" in fallback
    assert "1. Proceed" in fallback
    assert "2. Pause" in fallback


def test_card_send_failure_falls_back_without_losing_decision():
    content = _envelope([
        {"question": "Proceed?", "choices": ["Proceed", "Pause"]},
    ])

    result, adapter, _, standalone_calls = _run_live_delivery(
        content, card_ok=False
    )

    assert result is None
    assert standalone_calls == []
    adapter.send_deferred_decision.assert_awaited_once()
    assert adapter.send.await_count == 2
    assert "Decision: Proceed?" in adapter.send.await_args_list[1].args[1]


@pytest.mark.parametrize("loop_mode", ["missing", "stopped"])
def test_missing_or_stopped_live_loop_uses_readable_standalone_fallback(
    loop_mode,
    tmp_path,
):
    content = _envelope([
        {"question": "Proceed?", "choices": ["Proceed", "Pause"]},
    ])

    result, adapter, _, standalone_calls = _run_live_delivery(
        content,
        loop_mode=loop_mode,
    )

    assert result is None
    adapter.send.assert_not_awaited()
    adapter.send_deferred_decision.assert_not_awaited()
    assert len(standalone_calls) == 1
    delivered = standalone_calls[0]["text"]
    assert "Completed analysis." in delivered
    assert "Decision: Proceed?" in delivered
    assert "1. Proceed" in delivered
    assert "2. Pause" in delivered
    assert "hermes-deferred-decisions" not in delivered
    assert _stored_records(tmp_path) == []


def test_live_main_send_failure_preserves_decisions_in_standalone_fallback(tmp_path):
    content = _envelope([
        {"question": "Proceed?", "choices": ["Proceed", "Pause"]},
    ])

    result, adapter, _, standalone_calls = _run_live_delivery(
        content,
        main_ok=False,
    )

    assert result is None
    adapter.send.assert_awaited_once()
    adapter.send_deferred_decision.assert_not_awaited()
    assert len(standalone_calls) == 1
    delivered = standalone_calls[0]["text"]
    assert "Completed analysis." in delivered
    assert "Decision: Proceed?" in delivered
    assert "1. Proceed" in delivered
    assert "2. Pause" in delivered
    assert "hermes-deferred-decisions" not in delivered
    assert _stored_records(tmp_path) == []


def _deferred_card_send_kwargs(adapter):
    from cron.deferred_decisions import DeferredDecisionCard

    session_source, session_key = _session_binding(
        chat_id="1200",
        chat_type="group",
        user_id="700",
        thread_id="44",
    )
    card = DeferredDecisionCard("Proceed?", ("Proceed", "Pause"))
    return card, {
        "job": {"id": "a1b2c3d4e5f6", "name": "Release review"},
        "cards": [card],
        "adapter": adapter,
        "chat_id": "1200",
        "thread_id": "44",
        "user_id": "700",
        "platform_name": "telegram",
        "metadata": {"thread_id": "44"},
        "loop": MagicMock(),
        "context_ready": True,
        "session_source": session_source,
        "session_key": session_key,
    }


def _claim_stored_card(record, *, message_id=None):
    from cron import deferred_decisions as decisions

    return decisions.claim_choice(
        job_id=record["job_id"],
        decision_id=record["decision_id"],
        card_index=record["card_index"],
        choice_index=0,
        platform="telegram",
        chat_id="1200",
        thread_id="44",
        user_id="700",
        message_id=message_id,
    )


def test_card_timeout_already_in_flight_avoids_fallback_and_late_success_binds(
    tmp_path,
):
    from cron.scheduler import _send_deferred_decision_cards
    from gateway.platforms.base import SendResult

    adapter = MagicMock(spec=["send_deferred_decision"])
    adapter.send_deferred_decision = AsyncMock()
    scheduled = MagicMock()
    scheduled.result.side_effect = TimeoutError
    scheduled.cancel.return_value = False
    done_callbacks = []
    scheduled.add_done_callback.side_effect = done_callbacks.append

    def schedule(coro, _loop):
        coro.close()
        return scheduled

    card, kwargs = _deferred_card_send_kwargs(adapter)
    with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=schedule):
        failed = _send_deferred_decision_cards(**kwargs)

    assert failed == []
    assert len(done_callbacks) == 1
    [record] = _stored_records(tmp_path)
    assert _claim_stored_card(record, message_id="card-1") is None

    late = Future()
    late.set_result(SendResult(success=True, message_id="card-1"))
    done_callbacks[0](late)

    rebound = _stored_records(tmp_path)[0]
    claimed = _claim_stored_card(rebound, message_id="card-1")
    assert claimed is not None
    assert claimed.choice == "Proceed"
    assert card.question == claimed.record.question


def test_card_timeout_already_in_flight_late_failure_revokes_callback(tmp_path):
    from cron.scheduler import _send_deferred_decision_cards
    from gateway.platforms.base import SendResult

    adapter = MagicMock(spec=["send_deferred_decision"])
    adapter.send_deferred_decision = AsyncMock()
    scheduled = MagicMock()
    scheduled.result.side_effect = TimeoutError
    scheduled.cancel.return_value = False
    done_callbacks = []
    scheduled.add_done_callback.side_effect = done_callbacks.append

    def schedule(coro, _loop):
        coro.close()
        return scheduled

    _, kwargs = _deferred_card_send_kwargs(adapter)
    with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=schedule):
        failed = _send_deferred_decision_cards(**kwargs)

    assert failed == []
    [record] = _stored_records(tmp_path)
    late = Future()
    late.set_result(SendResult(success=False, error="late failure"))
    done_callbacks[0](late)

    assert _stored_records(tmp_path) == []
    assert _claim_stored_card(record, message_id="card-1") is None


def test_card_timeout_before_dispatch_cancels_record_and_uses_fallback(tmp_path):
    from cron.scheduler import _send_deferred_decision_cards

    adapter = MagicMock(spec=["send_deferred_decision"])
    adapter.send_deferred_decision = AsyncMock()
    scheduled = MagicMock()
    scheduled.result.side_effect = TimeoutError
    scheduled.cancel.return_value = True

    def schedule(coro, _loop):
        coro.close()
        return scheduled

    card, kwargs = _deferred_card_send_kwargs(adapter)
    with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=schedule):
        failed = _send_deferred_decision_cards(**kwargs)

    assert failed == [card]
    assert _stored_records(tmp_path) == []


def test_malformed_envelope_is_delivered_as_ordinary_text():
    malformed = (
        "Completed analysis.\n\n```hermes-deferred-decisions\n"
        '{"version":1,"cards":[{"question":"Q?","choices":["Only one"]}]}\n```'
    )

    result, adapter, _, standalone_calls = _run_live_delivery(malformed)

    assert result is None
    assert standalone_calls == []
    adapter.send_deferred_decision.assert_not_awaited()
    assert adapter.send.await_count == 1
    assert adapter.send.await_args.args[1] == malformed


def test_valid_envelope_on_ineligible_delivery_becomes_readable_plain_text():
    content = _envelope([
        {"question": "Proceed?", "choices": ["Proceed", "Pause"]},
    ])

    result, adapter, _, standalone_calls = _run_live_delivery(
        content, attach=False
    )

    assert result is None
    assert standalone_calls == []
    adapter.send_deferred_decision.assert_not_awaited()
    delivered = adapter.send.await_args.args[1]
    assert "Completed analysis." in delivered
    assert "Decision: Proceed?" in delivered
    assert "1. Proceed" in delivered
    assert "hermes-deferred-decisions" not in delivered
