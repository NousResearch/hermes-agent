"""Tests for the first-class control-plane authority substrate.

These pin the behaviours the adversarial review flagged as BLOCKers:

* a work verdict is orthogonal to its ACK — a failed/skipped ACK never
  downgrades a ``GO``/``BLOCK`` verdict, and a missing subscription is
  *not* ``NEED_MORE``;
* ``passive_sent`` and ``active_wake`` are distinct, never conflated;
* stale compaction / stale todo / stale history cannot authorize a
  board/cron/send mutation — the gate fail-closes and *demotes* them;
* a lane rejects a cross-lane (e.g. Warroom) mutation unless an explicit
  approval or a pre-approved route is present.

Lane / board names like ``#hermes-main``, ``#warroom``, ``#research``
appear ONLY as test fixtures — the module under test is lane-agnostic.
"""

from __future__ import annotations

import pytest

from hermes_cli.control_plane_contracts import (
    AckMode,
    AckStatus,
    AuthorizationRecord,
    ContextContract,
    DeliveryAckState,
    DeliveryEnvelope,
    LaneContract,
    MutationAction,
    OriginReturnContract,
    ResumePacket,
    ScopedTodo,
    StaleAuthorityError,
    WorkVerdict,
    authorize_mutation_from_env,
    authorize_pre_action_mutation,
    validate_resume_packet_authority,
)

# --------------------------------------------------------------------------
# Fixtures — lane-specific names live here only, never in the module.
# --------------------------------------------------------------------------

HERMES_MAIN = "#hermes-main"
WARROOM = "#warroom"
RESEARCH = "#research"
CURRENT_EPOCH = 100


def _lane(
    lane: str = HERMES_MAIN,
    epoch: int = CURRENT_EPOCH,
    *,
    allowed: frozenset[str] = frozenset(),
    allow_cross: bool = True,
) -> LaneContract:
    return LaneContract(
        lane=lane,
        authority_epoch=epoch,
        allowed_mutation_lanes=allowed,
        allow_cross_lane_with_approval=allow_cross,
    )


def _packet(
    *,
    lane: str = HERMES_MAIN,
    context_epoch: int = CURRENT_EPOCH,
    compaction_epoch: int | None = None,
    history_epoch: int | None = None,
    todos: tuple[ScopedTodo, ...] = (),
) -> ResumePacket:
    return ResumePacket(
        packet_id="rp_1",
        lane=lane,
        context=ContextContract(
            context_id="ctx_1", lane=lane, authority_epoch=context_epoch
        ),
        scoped_todos=todos,
        compaction_epoch=compaction_epoch,
        history_epoch=history_epoch,
    )


# --------------------------------------------------------------------------
# work_verdict vs ACK orthogonality
# --------------------------------------------------------------------------


def test_failed_ack_does_not_downgrade_go_verdict():
    env = DeliveryEnvelope("t_a2e271f5", WorkVerdict.GO)
    failed = env.with_ack(DeliveryAckState.failed("no_live_gateway_runner"))

    # The ACK failed, but the work verdict is byte-for-byte preserved.
    assert failed.work_verdict is WorkVerdict.GO
    assert failed.work_complete is True
    assert failed.ack.status is AckStatus.FAILED
    assert failed.ack.error == "no_live_gateway_runner"
    assert failed.needs_more_work is False
    # original envelope untouched (frozen / immutable)
    assert env.ack.status is AckStatus.PENDING


def test_failed_ack_keeps_block_verdict_intact():
    env = DeliveryEnvelope("t_x", WorkVerdict.BLOCK).with_ack(
        DeliveryAckState.failed("no_target")
    )
    assert env.work_verdict is WorkVerdict.BLOCK
    assert env.work_complete is True
    assert env.ack_outstanding is True


def test_projection_keeps_work_and_ack_facets_separate():
    env = DeliveryEnvelope("t_x", WorkVerdict.GO).with_ack(
        DeliveryAckState.failed("no_subscription")
    )
    proj = env.project()
    assert proj["work"] == {"verdict": "GO", "complete": True}
    assert proj["ack"]["status"] == "FAILED"
    # the work facet carries no ack key and vice versa
    assert "verdict" not in proj["ack"]
    assert "status" not in proj["work"]


# --------------------------------------------------------------------------
# missing subscription is SKIPPED, never NEED_MORE
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", [WorkVerdict.GO, WorkVerdict.BLOCK])
def test_missing_subscription_is_skipped_ack_not_need_more(verdict):
    env = DeliveryEnvelope("t_x", verdict).with_ack(
        DeliveryAckState.skipped("no_subscription")
    )
    # The ACK is skipped-with-reason...
    assert env.ack.status is AckStatus.SKIPPED_WITH_REASON
    assert env.ack.reason == "no_subscription"
    # ...but the WORK verdict stays GO/BLOCK and is NOT NEED_MORE.
    assert env.work_verdict is verdict
    assert env.needs_more_work is False
    assert env.work_complete is True


def test_skipped_ack_requires_a_reason():
    with pytest.raises(ValueError):
        DeliveryAckState(status=AckStatus.SKIPPED_WITH_REASON)


def test_failed_ack_requires_an_error():
    with pytest.raises(ValueError):
        DeliveryAckState(status=AckStatus.FAILED)


# --------------------------------------------------------------------------
# passive_sent vs active_wake are distinct
# --------------------------------------------------------------------------


def test_passive_sent_is_distinct_from_active_wake():
    passive = DeliveryAckState.passive_sent()
    active = DeliveryAckState.active_wake()

    assert passive.mode is AckMode.PASSIVE_SENT
    assert active.mode is AckMode.ACTIVE_WAKE
    assert passive != active

    assert passive.is_passive_sent and not passive.is_active_wake
    assert active.is_active_wake and not active.is_passive_sent
    # both count as delivered, but the mode is never lost
    assert passive.delivered and active.delivered
    assert AckMode.PASSIVE_SENT.value == "passive_sent"
    assert AckMode.ACTIVE_WAKE.value == "active_wake"


def test_sent_ack_must_record_a_mode():
    with pytest.raises(ValueError):
        DeliveryAckState(status=AckStatus.SENT)


# --------------------------------------------------------------------------
# resume packet authority — stale sources are demoted
# --------------------------------------------------------------------------


def test_current_resume_packet_authority_is_authorized():
    rec = validate_resume_packet_authority(_packet(), _lane())
    assert rec.authorized is True
    assert bool(rec) is True
    assert rec.demoted is False


@pytest.mark.parametrize(
    "kwargs, expected_source_prefix",
    [
        ({"compaction_epoch": 42}, "compaction:"),
        ({"history_epoch": 42}, "history:"),
        ({"context_epoch": 42}, "context:"),
    ],
)
def test_stale_source_blocks_and_demotes_resume_packet(kwargs, expected_source_prefix):
    rec = validate_resume_packet_authority(_packet(**kwargs), _lane())
    assert rec.authorized is False
    assert rec.demoted is True
    assert any(s.startswith(expected_source_prefix) for s in rec.blocked_sources)


def test_stale_scoped_todo_blocks_resume_packet():
    stale = ScopedTodo(todo_id="td_old", lane=HERMES_MAIN, authority_epoch=10)
    rec = validate_resume_packet_authority(_packet(todos=(stale,)), _lane())
    assert rec.authorized is False
    assert "todo:td_old" in rec.blocked_sources


def test_cross_lane_resume_packet_is_blocked():
    rec = validate_resume_packet_authority(_packet(lane=RESEARCH), _lane(HERMES_MAIN))
    assert rec.authorized is False
    assert any("resume_packet:lane=" in s for s in rec.blocked_sources)


# --------------------------------------------------------------------------
# authorize_pre_action_mutation — fail-closed gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("action_kind", ["board", "cron", "send"])
def test_stale_compaction_cannot_authorize_mutation(action_kind):
    action = MutationAction(kind=action_kind, target_lane=HERMES_MAIN)
    rec = authorize_pre_action_mutation(
        action, _lane(), _packet(compaction_epoch=1)
    )
    assert rec.authorized is False
    assert rec.demoted is True
    assert any(s.startswith("compaction:") for s in rec.blocked_sources)
    with pytest.raises(StaleAuthorityError):
        rec.raise_for_status()


@pytest.mark.parametrize("action_kind", ["board", "cron", "send"])
def test_stale_history_cannot_authorize_mutation(action_kind):
    action = MutationAction(kind=action_kind, target_lane=HERMES_MAIN)
    rec = authorize_pre_action_mutation(action, _lane(), _packet(history_epoch=1))
    assert rec.authorized is False
    assert rec.demoted is True
    assert any(s.startswith("history:") for s in rec.blocked_sources)


def test_stale_call_site_todo_cannot_authorize_mutation():
    action = MutationAction(kind="board", target_lane=HERMES_MAIN)
    stale = ScopedTodo(todo_id="td_stale", lane=HERMES_MAIN, authority_epoch=5)
    rec = authorize_pre_action_mutation(
        action, _lane(), _packet(), scoped_todos=(stale,)
    )
    assert rec.authorized is False
    assert rec.demoted is True
    assert "todo:td_stale" in rec.blocked_sources


def test_current_authority_authorizes_in_lane_mutation():
    action = MutationAction(kind="board", target_lane=HERMES_MAIN)
    rec = authorize_pre_action_mutation(action, _lane(), _packet())
    assert rec.authorized is True
    rec.raise_for_status()  # does not raise


# --------------------------------------------------------------------------
# lane contract — cross-lane (Warroom) mutation requires approval/route
# --------------------------------------------------------------------------


def test_hermes_main_lane_rejects_warroom_mutation_without_approval():
    # generic check: the #hermes-main lane may not mutate the #warroom
    # board with no explicit approval and no pre-approved route.
    action = MutationAction(kind="board", target_lane=WARROOM)
    rec = authorize_pre_action_mutation(action, _lane(HERMES_MAIN), _packet())
    assert rec.authorized is False
    assert rec.blocked_sources == (f"cross_lane:{WARROOM}",)


def test_cross_lane_mutation_allowed_with_explicit_approval():
    action = MutationAction(kind="board", target_lane=WARROOM)
    rec = authorize_pre_action_mutation(
        action, _lane(HERMES_MAIN), _packet(), explicit_approval=True
    )
    assert rec.authorized is True


def test_cross_lane_mutation_allowed_via_preapproved_route():
    action = MutationAction(kind="board", target_lane=WARROOM)
    lane = _lane(HERMES_MAIN, allowed=frozenset({WARROOM}))
    rec = authorize_pre_action_mutation(action, lane, _packet())
    assert rec.authorized is True


def test_explicit_approval_denied_when_lane_forbids_cross_lane():
    action = MutationAction(kind="board", target_lane=WARROOM)
    lane = _lane(HERMES_MAIN, allow_cross=False)
    rec = authorize_pre_action_mutation(
        action, lane, _packet(), explicit_approval=True
    )
    assert rec.authorized is False


def test_stale_authority_blocks_even_with_explicit_approval():
    # explicit approval covers cross-lane routing, NOT stale authority.
    action = MutationAction(kind="cron", target_lane=WARROOM)
    rec = authorize_pre_action_mutation(
        action, _lane(HERMES_MAIN), _packet(compaction_epoch=1),
        explicit_approval=True,
    )
    assert rec.authorized is False
    assert rec.demoted is True


# --------------------------------------------------------------------------
# OriginReturnContract — structured, immutable, stable return_id
# --------------------------------------------------------------------------


def test_origin_return_contract_target_and_stable_return_id():
    c = OriginReturnContract(platform="discord", chat_id="1497895797579190357")
    assert c.target == "discord:1497895797579190357"
    assert c.return_id.startswith("ret_")
    # return_id is deterministic / byte-stable
    again = OriginReturnContract(platform="discord", chat_id="1497895797579190357")
    assert c.return_id == again.return_id


def test_origin_return_contract_thread_id_changes_target_and_id():
    base = OriginReturnContract(platform="telegram", chat_id="12345")
    threaded = OriginReturnContract(platform="telegram", chat_id="12345", thread_id="678")
    assert threaded.target == "telegram:12345:678"
    assert threaded.return_id != base.return_id


def test_origin_return_contract_from_origin_dict():
    c = OriginReturnContract.from_origin_dict(
        {"platform": "discord", "chat_id": "999", "thread_id": None}
    )
    assert c is not None
    assert c.platform == "discord" and c.chat_id == "999"
    assert c.source == "legacy_prose"
    assert OriginReturnContract.from_origin_dict(None) is None
    assert OriginReturnContract.from_origin_dict({"platform": "x"}) is None


def test_origin_return_contract_to_dict_carries_no_body_text():
    c = OriginReturnContract(platform="discord", chat_id="999", lane=HERMES_MAIN)
    d = c.to_dict()
    assert set(d) == {
        "return_id", "platform", "chat_id", "thread_id", "lane", "source",
    }
    # only routing metadata — no message body / secret fields
    assert d["return_id"] == c.return_id


def test_origin_return_contract_from_prose_materializes_structured_contract():
    body = "Origin/return_to: discord:1497895797579190357 (#hermes-main)\n\nDo the thing."
    c = OriginReturnContract.from_prose(body, lane=HERMES_MAIN)
    assert c is not None
    assert c.platform == "discord"
    assert c.chat_id == "1497895797579190357"
    assert c.source == "legacy_prose"
    assert c.lane == HERMES_MAIN
    assert OriginReturnContract.from_prose("no directive here") is None
    assert OriginReturnContract.from_prose(None) is None


# --------------------------------------------------------------------------
# Environment / session-contract bridge — authorize_mutation_from_env
# --------------------------------------------------------------------------


def test_env_bridge_noop_when_no_authority_contract_configured():
    # No HERMES_CONTROL_PLANE_LANE => explicit authorized record so
    # existing CLI / tests that never set the env are unaffected.
    rec = authorize_mutation_from_env("send", env={})
    assert rec.authorized is True
    assert rec.reason == "no authority contract configured"


def test_env_bridge_authorizes_current_in_lane_mutation():
    env = {
        "HERMES_CONTROL_PLANE_LANE": HERMES_MAIN,
        "HERMES_CONTROL_PLANE_EPOCH": "100",
    }
    rec = authorize_mutation_from_env("board", env=env)
    assert rec.authorized is True


@pytest.mark.parametrize(
    "stale_var",
    [
        "HERMES_RESUME_COMPACTION_EPOCH",
        "HERMES_RESUME_HISTORY_EPOCH",
        "HERMES_RESUME_CONTEXT_EPOCH",
    ],
)
def test_env_bridge_blocks_stale_resume_authority(stale_var):
    env = {
        "HERMES_CONTROL_PLANE_LANE": HERMES_MAIN,
        "HERMES_CONTROL_PLANE_EPOCH": "100",
        stale_var: "1",  # stale relative to epoch 100
    }
    rec = authorize_mutation_from_env("board", env=env)
    assert rec.authorized is False
    assert rec.demoted is True


def test_env_bridge_blocks_stale_scoped_todo():
    env = {
        "HERMES_CONTROL_PLANE_LANE": HERMES_MAIN,
        "HERMES_CONTROL_PLANE_EPOCH": "100",
        "HERMES_RESUME_TODO_EPOCHS": "100,3",  # second todo is stale
    }
    rec = authorize_mutation_from_env("cron", env=env)
    assert rec.authorized is False
    assert rec.demoted is True


def test_env_bridge_blocks_cross_lane_mutation_without_approval():
    env = {
        "HERMES_CONTROL_PLANE_LANE": HERMES_MAIN,
        "HERMES_CONTROL_PLANE_EPOCH": "100",
        "HERMES_CONTROL_PLANE_TARGET_LANE": WARROOM,
    }
    rec = authorize_mutation_from_env("board", env=env)
    assert rec.authorized is False
    assert rec.blocked_sources == (f"cross_lane:{WARROOM}",)


def test_env_bridge_allows_cross_lane_mutation_with_explicit_approval():
    env = {
        "HERMES_CONTROL_PLANE_LANE": HERMES_MAIN,
        "HERMES_CONTROL_PLANE_EPOCH": "100",
        "HERMES_CONTROL_PLANE_TARGET_LANE": WARROOM,
        "HERMES_CONTROL_PLANE_APPROVAL": "1",
    }
    rec = authorize_mutation_from_env("board", env=env)
    assert rec.authorized is True


def test_env_bridge_allows_cross_lane_mutation_via_preapproved_route():
    env = {
        "HERMES_CONTROL_PLANE_LANE": HERMES_MAIN,
        "HERMES_CONTROL_PLANE_EPOCH": "100",
        "HERMES_CONTROL_PLANE_TARGET_LANE": WARROOM,
        "HERMES_CONTROL_PLANE_ALLOWED_LANES": f"{RESEARCH},{WARROOM}",
    }
    rec = authorize_mutation_from_env("board", env=env)
    assert rec.authorized is True


def test_env_bridge_explicit_param_target_lane_overrides_env():
    env = {
        "HERMES_CONTROL_PLANE_LANE": HERMES_MAIN,
        "HERMES_CONTROL_PLANE_EPOCH": "100",
    }
    # target_lane passed by the caller still triggers the cross-lane block.
    rec = authorize_mutation_from_env("board", target_lane=WARROOM, env=env)
    assert rec.authorized is False
