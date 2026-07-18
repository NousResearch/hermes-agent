"""Durable proposal store: TTL, origin binding, replay, restart safety."""

import os
import stat

import pytest

from plugins.sol_food.limits import (
    FOOD_DEDUP_RETENTION_SECONDS,
    FOOD_PROPOSAL_TTL_SECONDS,
)
from plugins.sol_food.proposal import (
    ACTION_CANCEL,
    ACTION_CONFIRM,
    Candidate,
    ProposalError,
    ProposalState,
)
from plugins.sol_food.store import CallbackOutcome, FoodProposalStore

BOT = "999000"
OWNER = "208214988"
THREAD = 1


class Clock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


def candidates(n: int = 2):
    return [
        Candidate(
            label=f"synthetic option {i}",
            items=({"plant_key": f"synthetic_item_{i}", "is_plant": True},),
        )
        for i in range(n)
    ]


async def make_proposal(store, update_id=1000, message_id=42):
    return await store.create(
        bot_id=BOT,
        owner_chat_id=OWNER,
        thread_id=THREAD,
        origin_update_id=update_id,
        origin_message_id=message_id,
        candidates=candidates(),
    )


def resolve_kwargs(token, update_id=2000, message_id=77, **overrides):
    kwargs = dict(
        token=token,
        update_id=update_id,
        bot_id=BOT,
        chat_id=OWNER,
        thread_id=THREAD,
        callback_message_id=message_id,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def store(tmp_path, clock):
    return FoodProposalStore(tmp_path / "state", clock=clock)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_create_and_single_active(self, store):
        proposal, tokens = await make_proposal(store)
        assert proposal.state is ProposalState.PENDING
        assert set(tokens) == {"choice:0", "choice:1", ACTION_CONFIRM, "edit", ACTION_CANCEL}
        with pytest.raises(ProposalError) as excinfo:
            await make_proposal(store, update_id=1001)
        assert excinfo.value.reason_code == "food_store_active_proposal_exists"

    @pytest.mark.asyncio
    async def test_state_files_are_owner_private(self, store, tmp_path):
        await make_proposal(store)
        state_dir = tmp_path / "state"
        assert stat.S_IMODE(os.stat(state_dir).st_mode) == 0o700
        state_file = state_dir / "sol-food-proposals.json"
        assert stat.S_IMODE(os.stat(state_file).st_mode) == 0o600

    @pytest.mark.asyncio
    async def test_atomic_state_publish_fsyncs_parent_after_replace(
        self, tmp_path, clock, monkeypatch
    ):
        events = []
        real_fsync = os.fsync
        real_replace = os.replace

        def tracked_fsync(fd):
            mode = os.fstat(fd).st_mode
            events.append("fsync_dir" if stat.S_ISDIR(mode) else "fsync_file")
            return real_fsync(fd)

        def tracked_replace(src, dst):
            events.append("replace")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "fsync", tracked_fsync)
        monkeypatch.setattr(os, "replace", tracked_replace)
        durable_store = FoodProposalStore(tmp_path / "durable", clock=clock)

        await make_proposal(durable_store)

        assert events.index("fsync_file") < events.index("replace")
        assert events.index("replace") < events.index("fsync_dir")

    @pytest.mark.asyncio
    async def test_ttl_boundary_minus_one_still_valid(self, store, clock):
        _proposal, tokens = await make_proposal(store)
        clock.now += FOOD_PROPOSAL_TTL_SECONDS - 1
        outcome = await store.resolve_callback(
            **resolve_kwargs(tokens[ACTION_CONFIRM], message_id=None)
        )
        assert outcome.kind == CallbackOutcome.KIND_ACTION

    @pytest.mark.asyncio
    async def test_ttl_boundary_at_limit_expired(self, tmp_path, clock):
        store = FoodProposalStore(tmp_path / "ttl-at", clock=clock)
        _proposal, tokens = await make_proposal(store)
        clock.now += FOOD_PROPOSAL_TTL_SECONDS  # exactly at expiry
        outcome = await store.resolve_callback(
            **resolve_kwargs(tokens[ACTION_CONFIRM], message_id=None)
        )
        assert outcome.kind == CallbackOutcome.KIND_DENIED
        assert outcome.reason_code == "food_store_expired"

    @pytest.mark.asyncio
    async def test_ttl_boundary_past_limit_expired(self, tmp_path, clock):
        store = FoodProposalStore(tmp_path / "ttl-past", clock=clock)
        _proposal, tokens = await make_proposal(store)
        clock.now += FOOD_PROPOSAL_TTL_SECONDS + 1
        outcome = await store.resolve_callback(
            **resolve_kwargs(tokens[ACTION_CONFIRM], message_id=None)
        )
        assert outcome.kind == CallbackOutcome.KIND_DENIED
        assert outcome.reason_code == "food_store_expired"

    @pytest.mark.asyncio
    async def test_awaiting_commit_remains_active_after_ttl(self, store, clock):
        proposal, tokens = await make_proposal(store)
        proposal, _ = await store.edit_new_version(
            proposal.proposal_id, candidates(1)
        )
        confirm = next(
            token
            for token, record in proposal.tokens.items()
            if record["action"] == ACTION_CONFIRM and not record["consumed"]
        )
        consumed = await store.resolve_callback(
            **resolve_kwargs(confirm, update_id=1999, message_id=None)
        )
        assert consumed.kind == CallbackOutcome.KIND_ACTION
        clock.now += FOOD_PROPOSAL_TTL_SECONDS + 1

        active = await store.active_proposal(OWNER, THREAD)
        assert active is not None
        assert active.awaiting_commit is True
        with pytest.raises(ProposalError) as excinfo:
            await make_proposal(store, update_id=2001)
        assert excinfo.value.reason_code == "food_store_active_proposal_exists"

    @pytest.mark.asyncio
    async def test_expired_proposal_content_scrubbed(self, store, clock):
        proposal, _tokens = await make_proposal(store)
        clock.now += FOOD_PROPOSAL_TTL_SECONDS + 1
        await store.sweep()
        stored = await store.get(proposal.proposal_id)
        assert stored.state is ProposalState.EXPIRED
        assert stored.candidates == []


class TestCallbackValidation:
    @pytest.mark.asyncio
    async def test_unknown_token_denied(self, store):
        await make_proposal(store)
        outcome = await store.resolve_callback(
            **resolve_kwargs("sf1:" + "A" * 22, message_id=None)
        )
        assert outcome.kind == CallbackOutcome.KIND_DENIED
        assert outcome.reason_code == "food_store_unknown_token"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "override",
        [
            {"chat_id": "31337"},
            {"bot_id": "other-bot"},
            {"thread_id": 2},
            {"thread_id": None},
        ],
    )
    async def test_foreign_origin_denied(self, store, override):
        _proposal, tokens = await make_proposal(store)
        outcome = await store.resolve_callback(
            **resolve_kwargs(tokens[ACTION_CONFIRM], message_id=None, **override)
        )
        assert outcome.kind == CallbackOutcome.KIND_DENIED
        assert outcome.reason_code == "food_store_foreign_origin"

    @pytest.mark.asyncio
    async def test_wrong_presentation_message_denied(self, store):
        proposal, tokens = await make_proposal(store)
        await store.record_presentation(proposal.proposal_id, 555)
        outcome = await store.resolve_callback(
            **resolve_kwargs(tokens[ACTION_CONFIRM], message_id=556)
        )
        assert outcome.kind == CallbackOutcome.KIND_DENIED
        assert outcome.reason_code == "food_store_bad_presentation"

    @pytest.mark.asyncio
    async def test_stale_version_denied_after_edit(self, store):
        proposal, tokens = await make_proposal(store)
        old_token = tokens[ACTION_CONFIRM]
        _p2, _tokens2 = await store.edit_new_version(
            proposal.proposal_id, candidates(1)
        )
        outcome = await store.resolve_callback(
            **resolve_kwargs(old_token, message_id=None)
        )
        assert outcome.kind == CallbackOutcome.KIND_DENIED
        # Old tokens are invalidated on edit (consumed) — a stale-version
        # token can never fire an action.
        assert outcome.reason_code in (
            "food_store_stale_version",
            "food_store_already_resolved",
        )

    @pytest.mark.asyncio
    async def test_edit_preserves_lifetime(self, store, clock):
        proposal, _tokens = await make_proposal(store)
        clock.now += FOOD_PROPOSAL_TTL_SECONDS - 10
        _p2, tokens2 = await store.edit_new_version(
            proposal.proposal_id, candidates(1)
        )
        clock.now += 11  # past the ORIGINAL expiry
        outcome = await store.resolve_callback(
            **resolve_kwargs(tokens2[ACTION_CONFIRM], update_id=2500, message_id=None)
        )
        assert outcome.kind == CallbackOutcome.KIND_DENIED
        assert outcome.reason_code == "food_store_expired"


class TestReplayAndConsumption:
    @pytest.mark.asyncio
    async def test_action_consumes_once(self, store):
        proposal, tokens = await make_proposal(store)
        token = tokens[ACTION_CONFIRM]
        outcome = await store.resolve_callback(**resolve_kwargs(token, message_id=None))
        assert outcome.kind == CallbackOutcome.KIND_ACTION
        assert outcome.action == ACTION_CONFIRM
        # A NEW human callback (different update) on the consumed token:
        # denied as already resolved, no new mutation.
        second = await store.resolve_callback(
            **resolve_kwargs(token, update_id=2001, message_id=None)
        )
        assert second.kind == CallbackOutcome.KIND_DENIED
        assert second.reason_code == "food_store_already_resolved"

    @pytest.mark.asyncio
    async def test_duplicate_update_is_replay_with_receipt(self, store):
        proposal, tokens = await make_proposal(store)
        token = tokens[ACTION_CONFIRM]
        outcome = await store.resolve_callback(**resolve_kwargs(token, message_id=None))
        assert outcome.kind == CallbackOutcome.KIND_ACTION
        await store.record_receipt(proposal.proposal_id, 2000, "a" * 64)
        # Exact same Telegram update delivered again: transport replay.
        replay = await store.resolve_callback(**resolve_kwargs(token, message_id=None))
        assert replay.kind == CallbackOutcome.KIND_REPLAY
        assert replay.receipt_ref == "a" * 64

    @pytest.mark.asyncio
    async def test_concurrent_double_tap_single_effect(self, store):
        import asyncio

        proposal, tokens = await make_proposal(store)
        token = tokens[ACTION_CONFIRM]
        results = await asyncio.gather(
            store.resolve_callback(**resolve_kwargs(token, update_id=3000, message_id=None)),
            store.resolve_callback(**resolve_kwargs(token, update_id=3001, message_id=None)),
        )
        kinds = sorted(r.kind for r in results)
        assert kinds == [CallbackOutcome.KIND_ACTION, CallbackOutcome.KIND_DENIED]

    @pytest.mark.asyncio
    async def test_awaiting_commit_cannot_be_edited_under_store_lock(self, store):
        proposal, tokens = await make_proposal(store)
        await store.edit_new_version(proposal.proposal_id, [candidates(1)[0]])
        proposal = await store.get(proposal.proposal_id)
        confirm_token = next(
            token
            for token, record in proposal.tokens.items()
            if (
                record["action"] == ACTION_CONFIRM
                and not record["consumed"]
                and record["version"] == proposal.version
            )
        )
        outcome = await store.resolve_callback(
            **resolve_kwargs(confirm_token, message_id=None)
        )
        assert outcome.kind == CallbackOutcome.KIND_ACTION
        before = await store.get(proposal.proposal_id)
        before_version = before.version
        before_hash = before.version_hash

        with pytest.raises(ProposalError) as excinfo:
            await store.edit_new_version(proposal.proposal_id, candidates(2))

        assert excinfo.value.reason_code == "food_store_commit_pending"
        after = await store.get(proposal.proposal_id)
        assert after.awaiting_commit is True
        assert after.version == before_version
        assert after.version_hash == before_hash

    @pytest.mark.asyncio
    async def test_cancel_scrubs_content(self, store):
        proposal, tokens = await make_proposal(store)
        outcome = await store.resolve_callback(
            **resolve_kwargs(tokens[ACTION_CANCEL], message_id=None)
        )
        assert outcome.action == ACTION_CANCEL
        await store.mark_terminal(proposal.proposal_id, ProposalState.CANCELLED)
        stored = await store.get(proposal.proposal_id)
        assert stored.candidates == []
        assert stored.state is ProposalState.CANCELLED


class TestRestartRecovery:
    @pytest.mark.asyncio
    async def test_consumption_survives_restart(self, tmp_path, clock):
        store = FoodProposalStore(tmp_path / "state", clock=clock)
        proposal, tokens = await make_proposal(store)
        token = tokens[ACTION_CONFIRM]
        outcome = await store.resolve_callback(**resolve_kwargs(token, message_id=None))
        assert outcome.kind == CallbackOutcome.KIND_ACTION
        await store.record_receipt(proposal.proposal_id, 2000, "b" * 64)

        # New process: same directory, fresh instance.
        reborn = FoodProposalStore(tmp_path / "state", clock=clock)
        replay = await reborn.resolve_callback(**resolve_kwargs(token, message_id=None))
        assert replay.kind == CallbackOutcome.KIND_REPLAY
        assert replay.receipt_ref == "b" * 64
        fresh_tap = await reborn.resolve_callback(
            **resolve_kwargs(token, update_id=2002, message_id=None)
        )
        assert fresh_tap.kind == CallbackOutcome.KIND_DENIED

    @pytest.mark.asyncio
    async def test_pending_proposal_survives_restart(self, tmp_path, clock):
        store = FoodProposalStore(tmp_path / "state", clock=clock)
        proposal, tokens = await make_proposal(store)
        reborn = FoodProposalStore(tmp_path / "state", clock=clock)
        outcome = await reborn.resolve_callback(
            **resolve_kwargs(tokens[ACTION_CONFIRM], message_id=None)
        )
        assert outcome.kind == CallbackOutcome.KIND_ACTION

    @pytest.mark.asyncio
    async def test_corrupt_state_fails_closed(self, tmp_path, clock):
        store = FoodProposalStore(tmp_path / "state", clock=clock)
        _proposal, tokens = await make_proposal(store)
        state_file = tmp_path / "state" / "sol-food-proposals.json"
        state_file.write_text("{ not json", encoding="utf-8")
        reborn = FoodProposalStore(tmp_path / "state", clock=clock)
        outcome = await reborn.resolve_callback(
            **resolve_kwargs(tokens[ACTION_CONFIRM], message_id=None)
        )
        # All tokens dead — deny, never guess.
        assert outcome.kind == CallbackOutcome.KIND_DENIED


class TestDedupRetention:
    @pytest.mark.asyncio
    async def test_48h_boundary(self, store, clock):
        proposal, tokens = await make_proposal(store)
        token = tokens[ACTION_CONFIRM]
        await store.resolve_callback(**resolve_kwargs(token, message_id=None))
        await store.record_receipt(proposal.proposal_id, 2000, "c" * 64)

        # limit-1: replay still recognized.
        clock.now += FOOD_DEDUP_RETENTION_SECONDS - 1
        replay = await store.resolve_callback(**resolve_kwargs(token, message_id=None))
        assert replay.kind == CallbackOutcome.KIND_REPLAY

        # limit+1 (relative to the consumption timestamp): linkage pruned;
        # the update is no longer recognized, and the token is gone with
        # the pruned proposal — fail closed, never a second effect.
        clock.now += 2
        gone = await store.resolve_callback(**resolve_kwargs(token, message_id=None))
        assert gone.kind == CallbackOutcome.KIND_DENIED
