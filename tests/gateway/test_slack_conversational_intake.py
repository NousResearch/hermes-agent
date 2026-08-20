"""Behavior contracts for private Slack conversational intake."""

from __future__ import annotations

import concurrent.futures
import asyncio
import json
import stat

import pytest

from gateway.config import Platform
from gateway.config import PlatformConfig
from gateway.session import build_session_key
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.hermes_intake import (
    SlackIntakeStore,
    build_intake_prompt,
    build_intake_source,
    build_promotion_payload,
    parse_resolved_brief,
)


def _source(**overrides):
    value = {
        "team_id": "T123",
        "channel_id": "C456",
        "channel_name": "private-source",
        "message_ts": "123.456",
        "message_text": "Private source text",
        "author_id": "UAUTHOR",
        "author_name": "A. User",
        "submitter_id": "UJERRY",
        "submitter_name": "Jerry",
        "permalink": "https://example.slack.com/archives/C456/p123456",
    }
    value.update(overrides)
    return value


async def _record_async(target, value):
    target.append(value)


def test_store_dedupes_callback_delivery_but_new_invocation_gets_new_intake(tmp_path):
    store = SlackIntakeStore(tmp_path / "intakes.db")

    first = store.reserve("delivery-1", "coding", _source())
    retry = store.reserve("delivery-1", "coding", _source())
    deliberate = store.reserve("delivery-2", "coding", _source())

    assert retry.intake_id == first.intake_id
    assert deliberate.intake_id != first.intake_id
    assert store.get(first.intake_id).state == "reserved"


def test_private_source_database_is_owner_only(tmp_path):
    path = tmp_path / "intakes.db"

    SlackIntakeStore(path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for suffix in ("-wal", "-shm"):
        artifact = path.with_name(path.name + suffix)
        if artifact.exists():
            assert stat.S_IMODE(artifact.stat().st_mode) == 0o600


def test_lineage_rows_do_not_duplicate_raw_source_text(tmp_path):
    store = SlackIntakeStore(tmp_path / "intakes.db")
    record = store.reserve("delivery-1", "coding", _source())
    store.bind_thread(record.intake_id, "D123", "200.001")
    store.bind_session(record.intake_id, "slack:T123:D123:200.001", "session-1")

    lineage = store.lineage(record.intake_id)
    assert "Private source text" not in json.dumps(lineage)
    assert lineage["dm_channel_id"] == "D123"
    assert lineage["thread_ts"] == "200.001"


def test_synthetic_source_matches_organic_dm_thread_identity():
    synthetic = build_intake_source(
        profile="coding",
        team_id="T123",
        dm_channel_id="D123",
        thread_ts="200.001",
        user_id="UJERRY",
        user_name="Jerry",
    )

    assert synthetic.platform is Platform.SLACK
    assert synthetic.chat_type == "dm"
    assert synthetic.scope_id == "T123"
    assert synthetic.chat_id == "D123"
    assert synthetic.thread_id == "200.001"
    assert synthetic.profile == "coding"

    adapter = _adapter_for_source_identity()
    organic = adapter.build_source(
        chat_id="D123",
        chat_type="dm",
        user_id="UJERRY",
        user_name="Jerry",
        thread_id="200.001",
        scope_id="T123",
    )
    assert build_session_key(synthetic) == build_session_key(organic)


def _adapter_for_source_identity():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="fake"))

    class Runner:
        @staticmethod
        def _profile_name_for_source(source):
            return "coding"

    adapter.gateway_runner = Runner()  # type: ignore[assignment]
    return adapter


def test_initial_prompt_is_human_originated_and_states_no_card_exists():
    prompt = build_intake_prompt(_source(), intake_id="i_123")

    assert "Private source text" in prompt
    assert "No card exists yet" in prompt
    assert "Create card" in prompt
    assert "i_123" in prompt
    assert "## Outcome" in prompt


def test_resolved_brief_is_parsed_from_latest_assistant_draft():
    brief = parse_resolved_brief(
        [{"role": "user", "content": "raw private source"}, {
            "role": "assistant",
            "content": (
                "## Outcome\nInvestigate checkout failures\n"
                "## Why\nCustomers cannot pay\n"
                "## Scope\n- Checkout API\n"
                "## Non-goals\n- Checkout redesign\n"
                "## Acceptance criteria\n- Root cause documented\n"
                "## Decisions\n- Triage first\n"
                "## Constraints\n- No customer data\n"
                "## Unresolved questions\n- Which deploy?"
            ),
        }]
    )

    assert brief["outcome"] == "Investigate checkout failures"
    assert brief["scope"] == ["Checkout API"]
    assert brief["unresolved_questions"] == ["Which deploy?"]


def test_promotion_payload_is_synthesized_lineage_not_raw_transcript():
    payload = build_promotion_payload(
        intake_id="i_123",
        promotion_key="slack-intake:promotion-1",
        source=_source(),
        session_key="slack:T123:D123:200.001",
        session_id="session-1",
        dm_channel_id="D123",
        thread_ts="200.001",
        brief={
            "outcome": "Investigate checkout failures",
            "why": "Customers cannot pay",
            "scope": ["Checkout API"],
            "non_goals": ["Redesign checkout"],
            "acceptance_criteria": ["Root cause is documented"],
            "decisions": ["Triage first"],
            "constraints": ["Do not expose customer data"],
            "unresolved_questions": ["Which deploy introduced it?"],
        },
        raw_transcript="SECRET TRANSCRIPT MUST NOT LEAK",
    )

    assert payload["triage"] is True
    assert "assignee" not in payload
    assert payload["idempotency_key"] == "slack-intake:promotion-1"
    assert "SECRET TRANSCRIPT MUST NOT LEAK" not in payload["body"]
    for heading in (
        "Outcome", "Why", "Scope", "Non-goals", "Acceptance criteria",
        "Decisions", "Constraints", "Unresolved questions", "Lineage",
    ):
        assert heading in payload["body"]


def test_concurrent_promotion_binding_converges_on_one_card(tmp_path):
    path = tmp_path / "intakes.db"
    initial = SlackIntakeStore(path)
    intake = initial.reserve("delivery-1", "coding", _source())

    def bind(card_id: str) -> str:
        return SlackIntakeStore(path).bind_card(intake.intake_id, "hrms", card_id).card_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(bind, ["t_first", "t_second"]))

    assert len(set(results)) == 1
    assert initial.get(intake.intake_id).card_id == results[0]


def test_concurrent_thread_binding_preserves_first_seed(tmp_path):
    path = tmp_path / "intakes.db"
    initial = SlackIntakeStore(path)
    intake = initial.reserve("delivery-1", "coding", _source())

    def bind(thread_ts: str) -> str | None:
        return SlackIntakeStore(path).bind_thread(
            intake.intake_id, "D123", thread_ts
        ).thread_ts

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(bind, ["200.001", "200.002"]))

    assert len(set(results)) == 1
    assert initial.get(intake.intake_id).thread_ts == results[0]


def test_concurrent_session_binding_preserves_first_session(tmp_path):
    path = tmp_path / "intakes.db"
    initial = SlackIntakeStore(path)
    intake = initial.reserve("delivery-1", "coding", _source())

    def bind(session_id: str) -> str | None:
        return SlackIntakeStore(path).bind_session(
            intake.intake_id, f"key:{session_id}", session_id
        ).session_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(bind, ["session-1", "session-2"]))

    assert len(set(results)) == 1
    assert initial.get(intake.intake_id).session_id == results[0]


def test_stale_stage_claim_is_recoverable_after_crash_lease(tmp_path):
    store = SlackIntakeStore(tmp_path / "intakes.db")
    intake = store.reserve("delivery-1", "coding", _source())
    first_owner = store.claim_stage(
        intake.intake_id, from_states=("reserved",), claimed_state="seeding"
    )
    assert first_owner
    assert not store.claim_stage(
        intake.intake_id, from_states=("reserved",), claimed_state="seeding"
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE slack_intakes SET updated_at = updated_at - 120, "
            "claim_pid = 2147483647 WHERE intake_id = ?",
            (intake.intake_id,),
        )

    second_owner = store.claim_stage(
        intake.intake_id, from_states=("reserved",), claimed_state="seeding"
    )
    assert second_owner and second_owner != first_owner
    assert not store.complete_stage(
        intake.intake_id, owner_token=first_owner, state="thread_bound"
    )
    assert store.complete_stage(
        intake.intake_id, owner_token=second_owner, state="thread_bound"
    )


def test_stale_stage_claim_does_not_overtake_live_owner(tmp_path):
    store = SlackIntakeStore(tmp_path / "intakes.db")
    intake = store.reserve("delivery-1", "coding", _source())
    owner = store.claim_stage(
        intake.intake_id, from_states=("reserved",), claimed_state="dispatching"
    )
    assert owner
    with store._connect() as conn:
        conn.execute(
            "UPDATE slack_intakes SET updated_at = updated_at - 120 WHERE intake_id = ?",
            (intake.intake_id,),
        )

    assert not store.claim_stage(
        intake.intake_id, from_states=("reserved",), claimed_state="dispatching"
    )


def _adapter(tmp_path, *, enabled=True):
    adapter = SlackAdapter(
        PlatformConfig(
            enabled=True,
            token="fake",
            extra={"conversational_intake_enabled": enabled},
        )
    )
    adapter._slack_intake_store = SlackIntakeStore(tmp_path / "intakes.db")
    adapter.set_authorization_check(
        lambda user_id, chat_type=None, chat_id=None: user_id == "UJERRY"
    )
    return adapter


class _TranscriptStore:
    @staticmethod
    def load_transcript(session_id):
        assert session_id == "session-1"
        return [{
            "role": "assistant",
            "content": (
                "## Outcome\nInvestigate checkout failures\n"
                "## Why\nCustomers cannot pay\n"
                "## Scope\n- Checkout API\n"
                "## Non-goals\n- Checkout redesign\n"
                "## Acceptance criteria\n- Root cause documented\n"
                "## Decisions\n- Triage first\n"
                "## Constraints\n- No customer data\n"
                "## Unresolved questions\n- None"
            ),
        }]


@pytest.mark.asyncio
async def test_shortcut_starts_one_normal_dm_thread_turn_and_zero_cards(tmp_path):
    adapter = _adapter(tmp_path)
    seen = []
    async def handle(event):
        seen.append(event)
        return "Hermes response"

    adapter.set_message_handler(handle)

    class Client:
        async def chat_getPermalink(self, **kwargs):
            return {"permalink": "https://example.slack.com/source"}

        async def conversations_open(self, **kwargs):
            return {"channel": {"id": "D123"}}

        async def chat_postMessage(self, **kwargs):
            return {"ts": "200.001"}

    body = {
        "trigger_id": "delivery-1",
        "team": {"id": "T123"},
        "user": {"id": "UJERRY", "name": "Jerry"},
        "channel": {"id": "C456", "name": "private-source"},
        "message": {"ts": "123.456", "user": "UAUTHOR", "text": "Private source"},
    }
    acked = []
    await adapter._handle_send_to_hermes_shortcut(
        lambda **kwargs: _record_async(acked, kwargs), body, Client()
    )
    if adapter._background_tasks:
        await asyncio.gather(*tuple(adapter._background_tasks))

    assert acked == [{}]
    assert len(seen) == 1
    assert seen[0].internal is False
    assert seen[0].message_id.startswith("slack-intake:")
    assert seen[0].source.chat_type == "dm"
    assert seen[0].source.chat_id == "D123"
    assert seen[0].source.thread_id == "200.001"
    with adapter._slack_intake_store._connect() as conn:
        row = conn.execute("SELECT * FROM slack_intakes").fetchone()
    assert row["card_id"] is None


@pytest.mark.asyncio
async def test_retry_after_thread_binding_dispatches_without_second_seed(tmp_path):
    adapter = _adapter(tmp_path)
    source = _source(message_text="Private source")
    intake = adapter._slack_intake_store.reserve("delivery-1", "", source)
    adapter._slack_intake_store.bind_thread(intake.intake_id, "D123", "200.001")
    seen = []
    adapter.set_message_handler(lambda event: _record_async(seen, event))

    class Client:
        async def conversations_open(self, **kwargs):
            raise AssertionError("retry must reuse the bound DM")

        async def chat_postMessage(self, **kwargs):
            if "thread_ts" not in kwargs:
                raise AssertionError("retry must not post a second seed")
            return {"ts": "200.002"}

    await adapter._start_slack_conversational_intake(
        invocation_key="delivery-1", source=source, client=Client()
    )
    if adapter._background_tasks:
        await asyncio.gather(*tuple(adapter._background_tasks))

    assert len(seen) == 1
    assert adapter._slack_intake_store.get(intake.intake_id).state == "active"


@pytest.mark.asyncio
async def test_concurrent_shortcut_retries_share_one_seed_and_first_turn(tmp_path):
    adapter = _adapter(tmp_path)
    second = _adapter(tmp_path)
    source = _source(message_text="Private source")
    seed_calls = []
    seen = []

    async def handler(event):
        seen.append(event.message_id)
        await asyncio.sleep(0)
        return None

    adapter.set_message_handler(handler)
    second.set_message_handler(handler)

    class Client:
        async def conversations_open(self, **kwargs):
            return {"channel": {"id": "D123"}}

        async def chat_postMessage(self, **kwargs):
            if "thread_ts" not in kwargs:
                seed_calls.append(kwargs["client_msg_id"])
                await asyncio.sleep(0)
                return {"ts": "200.001"}
            return {"ts": "controls"}

    await asyncio.gather(
        adapter._start_slack_conversational_intake(
            invocation_key="concurrent-1", source=source, client=Client()
        ),
        second._start_slack_conversational_intake(
            invocation_key="concurrent-1", source=source, client=Client()
        ),
    )

    assert len(seed_calls) == 1
    assert len(seen) == 1


def test_intake_source_uses_active_gateway_profile(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "coding")

    assert adapter._slack_intake_profile() == "coding"


def test_intake_source_prefers_gateway_profile_route(tmp_path):
    adapter = _adapter(tmp_path)

    class Runner:
        @staticmethod
        def _profile_name_for_source(source):
            assert source.chat_id == "D123"
            return "victory-live"

    adapter.gateway_runner = Runner()  # type: ignore[assignment]

    assert adapter._slack_intake_profile(
        build_intake_source(
            profile="",
            team_id="T123",
            dm_channel_id="D123",
            thread_ts="200.001",
            user_id="UJERRY",
            user_name="Jerry",
        )
    ) == "victory-live"


@pytest.mark.asyncio
async def test_failed_first_turn_stays_retryable_with_stable_event_id(tmp_path):
    adapter = _adapter(tmp_path)
    source = _source(message_text="Private source")
    seen = []

    async def fail(event):
        seen.append(event.message_id)
        raise RuntimeError("delivery failed")

    adapter.set_message_handler(fail)

    class Client:
        async def conversations_open(self, **kwargs):
            return {"channel": {"id": "D123"}}

        async def chat_postMessage(self, **kwargs):
            return {"ts": "200.001"}

    await adapter._start_slack_conversational_intake(
        invocation_key="delivery-1", source=source, client=Client()
    )

    intake = adapter._slack_intake_store.reserve("delivery-1", "", source)
    assert intake.state == "failed_retryable"
    assert seen == [intake.first_event_id]


@pytest.mark.asyncio
async def test_feature_flag_off_blocks_new_intake_without_persisting_source(tmp_path):
    adapter = _adapter(tmp_path, enabled=False)
    opened = []

    class Client:
        async def views_open(self, **kwargs):
            opened.append(kwargs)

    await adapter._handle_send_to_hermes_shortcut(
        lambda **kwargs: _record_async([], kwargs),
        {
            "trigger_id": "delivery-off",
            "team": {"id": "T123"},
            "user": {"id": "UJERRY"},
            "channel": {"id": "C456"},
            "message": {"ts": "123.456", "text": "must not persist"},
        },
        Client(),
    )

    assert "not enabled" in json.dumps(opened).lower()
    with adapter._slack_intake_store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM slack_intakes").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_create_card_action_reauthorizes_and_reuses_promoted_card(tmp_path):
    adapter = _adapter(tmp_path)
    source = _source(submitter_id="UJERRY", message_text="Investigate checkout failures")
    intake = adapter._slack_intake_store.reserve("delivery-1", "coding", source)
    adapter._slack_intake_store.bind_thread(intake.intake_id, "D123", "200.001")
    adapter._slack_intake_store.bind_session(
        intake.intake_id, "slack:T123:D123:200.001", "session-1"
    )
    adapter._session_store = _TranscriptStore()
    created = []

    class IntakeClient:
        async def create_task(self, payload):
            created.append(payload)
            return {"id": "t_only_one", "status": "triage"}

    adapter._hermes_intake_client = IntakeClient()
    acked = []
    updates = []

    class Client:
        async def chat_update(self, **kwargs):
            updates.append(kwargs)

    body = {
        "user": {"id": "UJERRY"},
        "team": {"id": "T123"},
        "channel": {"id": "D123"},
        "message": {"ts": "200.002", "thread_ts": "200.001"},
        "actions": [{"value": intake.intake_id}],
    }
    ack = lambda **kwargs: _record_async(acked, kwargs)

    await adapter._handle_slack_intake_create_card(ack, body, Client())
    await adapter._handle_slack_intake_create_card(ack, body, Client())

    assert len(created) == 1
    assert created[0]["triage"] is True
    assert "assignee" not in created[0]
    assert adapter._slack_intake_store.get(intake.intake_id).card_id == "t_only_one"
    assert "Creating card" in json.dumps(updates[0])
    assert all("t_only_one" in json.dumps(update) for update in updates[1:])
    assert all("hermes_intake_open_card" not in json.dumps(update) for update in updates)


@pytest.mark.asyncio
async def test_create_card_action_rejects_same_dm_wrong_thread(tmp_path):
    adapter = _adapter(tmp_path)
    intake = adapter._slack_intake_store.reserve("delivery-1", "coding", _source())
    adapter._slack_intake_store.bind_thread(intake.intake_id, "D123", "200.001")
    adapter._slack_intake_store.bind_session(
        intake.intake_id, "slack:T123:D123:200.001", "session-1"
    )
    created = []

    class IntakeClient:
        async def create_task(self, payload):
            created.append(payload)
            return {"id": "must-not-exist"}

    adapter._hermes_intake_client = IntakeClient()
    await adapter._handle_slack_intake_create_card(
        lambda **kwargs: _record_async([], kwargs),
        {
            "user": {"id": "UJERRY"},
            "team": {"id": "T123"},
            "channel": {"id": "D123"},
            "message": {"ts": "999.001", "thread_ts": "999.001"},
            "actions": [{"value": intake.intake_id}],
        },
        object(),
    )

    assert created == []


@pytest.mark.asyncio
async def test_unknown_promotion_outcome_keeps_same_key_and_renders_retry(tmp_path):
    adapter = _adapter(tmp_path)
    source = _source(submitter_id="UJERRY", message_text="Investigate checkout failures")
    intake = adapter._slack_intake_store.reserve("delivery-1", "coding", source)
    adapter._slack_intake_store.bind_thread(intake.intake_id, "D123", "200.001")
    adapter._slack_intake_store.bind_session(
        intake.intake_id, "slack:T123:D123:200.001", "session-1"
    )
    adapter._session_store = _TranscriptStore()
    keys = []

    class IntakeClient:
        async def create_task(self, payload):
            keys.append(payload["idempotency_key"])
            raise RuntimeError("acknowledgment lost")

    adapter._hermes_intake_client = IntakeClient()  # type: ignore[assignment]
    updates = []

    class Client:
        async def chat_update(self, **kwargs):
            updates.append(kwargs)

    body = {
        "user": {"id": "UJERRY"},
        "team": {"id": "T123"},
        "channel": {"id": "D123"},
        "message": {"ts": "200.002", "thread_ts": "200.001"},
        "actions": [{"value": intake.intake_id}],
    }

    await adapter._handle_slack_intake_create_card(
        lambda **kwargs: _record_async([], kwargs), body, Client()
    )

    assert keys == [intake.promotion_key]
    assert adapter._slack_intake_store.get(intake.intake_id).state == "failed_retryable"
    assert "Retry creation" in json.dumps(updates)