from agent.memory_policy import (
    WriteClass,
    assign_topic_key,
    classify_write_candidate,
    resolve_conflict,
    transition_freshness,
)
from agent.memory_records import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RecordStatus,
    SalienceTier,
    TrustTier,
)


def make_record(content: str, *, topic_key: str, trust_tier: TrustTier, created_at: str) -> MemoryRecord:
    return MemoryRecord(
        record_id=content.lower().replace(" ", "-"),
        memory_type=MemoryType.PROFILE,
        scope=MemoryScope.OPERATOR,
        topic_key=topic_key,
        content=content,
        source="test",
        source_kind="explicit_user_statement",
        trust_tier=trust_tier,
        salience_tier=SalienceTier.HIGH,
        status=RecordStatus.ACTIVE,
        created_at=created_at,
    )


def test_explicit_remember_is_must_write():
    result = classify_write_candidate(
        target="user",
        content="Remember this: I prefer detailed architecture writeups.",
        source_kind="explicit_user_statement",
        explicit_remember=True,
        explicit_correction=False,
    )

    assert result.write_class is WriteClass.MUST_WRITE


def test_transient_chatter_is_do_not_write():
    result = classify_write_candidate(
        target="memory",
        content="Let me think through this step by step and maybe try a few things.",
        source_kind="model_inference",
        explicit_remember=False,
        explicit_correction=False,
    )

    assert result.write_class is WriteClass.DO_NOT_WRITE


def test_plausible_inferred_workspace_fact_is_may_write():
    result = classify_write_candidate(
        target="memory",
        content="The repository likely deploys with make ship.",
        source_kind="model_inference",
        explicit_remember=False,
        explicit_correction=False,
    )

    assert result.write_class is WriteClass.MAY_WRITE
    assert result.trust_tier is TrustTier.INFERRED
    assert result.ambiguity_flag is True


def test_assign_topic_key_for_operator_preference():
    topic_key = assign_topic_key(
        target="user",
        content="User prefers British spelling.",
        scope=MemoryScope.OPERATOR,
    )

    assert topic_key is not None
    assert topic_key.startswith("preference:")


def test_assign_topic_key_for_workspace_deploy_fact():
    assert (
        assign_topic_key(
            target="memory",
            content="This repo deploys with make ship.",
            scope=MemoryScope.WORKSPACE,
        )
        == "workspace:deploy-command"
    )


def test_stale_transition_after_review_deadline():
    record = make_record(
        "User prefers concise replies.",
        topic_key="preference:reply-style",
        trust_tier=TrustTier.USER_ASSERTED,
        created_at="2026-01-01T00:00:00Z",
    )
    record.review_after = "2026-02-01T00:00:00Z"

    updated = transition_freshness(record, now="2026-03-01T00:00:00Z")

    assert updated.status is RecordStatus.STALE


def test_newer_explicit_correction_supersedes_older_equal_trust_record():
    old = make_record(
        "User prefers concise replies.",
        topic_key="preference:reply-style",
        trust_tier=TrustTier.USER_ASSERTED,
        created_at="2026-01-01T00:00:00Z",
    )
    new = make_record(
        "User wants fuller replies for design work.",
        topic_key="preference:reply-style",
        trust_tier=TrustTier.USER_ASSERTED,
        created_at="2026-03-01T00:00:00Z",
    )

    result = resolve_conflict(old, new, explicit_correction=True)

    assert result.winner.record_id == new.record_id
    assert result.loser_status is RecordStatus.SUPERSEDED


def test_scoped_refinement_keeps_both_records_and_marks_newer_scope_narrowed():
    old = make_record(
        "User prefers concise replies.",
        topic_key="preference:reply-style",
        trust_tier=TrustTier.USER_ASSERTED,
        created_at="2026-01-01T00:00:00Z",
    )
    new = make_record(
        "User prefers fuller replies for design work.",
        topic_key="preference:reply-style",
        trust_tier=TrustTier.USER_ASSERTED,
        created_at="2026-03-01T00:00:00Z",
    )

    result = resolve_conflict(old, new, explicit_correction=False)

    assert result.reason == "scoped_refinement_keep_both"
    assert result.winner.record_id == new.record_id
    assert result.winner.status is RecordStatus.ACTIVE
    assert result.winner.supersedes is None
    assert result.winner.metadata["scope_narrowed"] is True
    assert result.winner.metadata["scope_refinement_of"] == old.record_id
    assert result.loser.record_id == old.record_id
    assert result.loser_status is RecordStatus.ACTIVE
    assert result.loser.status is RecordStatus.ACTIVE


def test_lower_trust_conflict_is_disputed_instead_of_auto_superseding():
    old = make_record(
        "Project deploys with make ship.",
        topic_key="workspace:deploy-command",
        trust_tier=TrustTier.OBSERVED,
        created_at="2026-03-01T00:00:00Z",
    )
    new = make_record(
        "Project deploys with ./release.sh.",
        topic_key="workspace:deploy-command",
        trust_tier=TrustTier.INFERRED,
        created_at="2026-03-05T00:00:00Z",
    )
    old.scope = MemoryScope.WORKSPACE
    new.scope = MemoryScope.WORKSPACE

    result = resolve_conflict(old, new, explicit_correction=False)

    assert result.winner.record_id == old.record_id
    assert result.loser_status is RecordStatus.DISPUTED



def test_supersession_compacts_existing_chain_to_single_predecessor_hop():
    old = make_record(
        "Project deploys with make ship.",
        topic_key="workspace:deploy-command",
        trust_tier=TrustTier.INFERRED,
        created_at="2026-03-01T00:00:00Z",
    )
    new = make_record(
        "Project deploys with ./release.sh.",
        topic_key="workspace:deploy-command",
        trust_tier=TrustTier.OBSERVED,
        created_at="2026-03-05T00:00:00Z",
    )
    old.scope = MemoryScope.WORKSPACE
    new.scope = MemoryScope.WORKSPACE
    old.supersedes = "legacy-deploy-record"

    result = resolve_conflict(old, new, explicit_correction=False)

    assert result.winner.record_id == new.record_id
    assert result.loser_status is RecordStatus.SUPERSEDED
    assert result.winner.supersedes == "legacy-deploy-record"



def test_supersession_avoids_creating_circular_links():
    old = make_record(
        "User prefers concise replies.",
        topic_key="preference:reply-style",
        trust_tier=TrustTier.USER_ASSERTED,
        created_at="2026-01-01T00:00:00Z",
    )
    new = make_record(
        "User wants fuller replies.",
        topic_key="preference:reply-style",
        trust_tier=TrustTier.USER_ASSERTED,
        created_at="2026-03-01T00:00:00Z",
    )
    old.supersedes = new.record_id

    result = resolve_conflict(old, new, explicit_correction=True)

    assert result.winner.record_id == new.record_id
    assert result.loser_status is RecordStatus.SUPERSEDED
    assert result.winner.supersedes is None
    assert result.loser.supersedes == new.record_id
