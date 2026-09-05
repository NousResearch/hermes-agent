from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gateway.hermes_tag import (
    Fact,
    FactStore,
    IncompleteScope,
    ScopeRef,
    StorageError,
    Sensitivity,
    new_id,
    utc_text,
)


def _fact(scope, *, value="active", authority=10, confidence=1.0, **overrides):
    values = dict(
        fact_id=new_id("fact"),
        subject="project:hermes",
        predicate="status",
        value=value,
        scope=scope,
        source_type="github_api",
        source_id="NousResearch/hermes-agent",
        source_revision="sha-1",
        confidence=confidence,
        authority=authority,
        sensitivity=Sensitivity.INTERNAL,
        tags=("project",),
    )
    values.update(overrides)
    return Fact(**values)


def test_observe_and_get_fact(ledger, scope):
    store = FactStore(ledger)
    fact = _fact(scope)
    assert store.observe(fact) == fact
    assert store.get(fact.fact_id).content_hash == fact.content_hash


def test_duplicate_observation_is_idempotent(ledger, scope):
    store = FactStore(ledger)
    first = _fact(scope)
    second = _fact(scope, valid_from=first.valid_from)
    observed = store.observe(first)
    duplicate = store.observe(second)
    assert duplicate.fact_id == observed.fact_id


def test_cross_channel_fact_is_not_visible(ledger, scope):
    store = FactStore(ledger)
    other = ScopeRef(
        profile=scope.profile,
        platform=scope.platform,
        scope_id=scope.scope_id,
        chat_id="C_OTHER",
        principal_id=scope.principal_id,
        continuity_id=scope.continuity_id,
    )
    store.observe(_fact(other))
    result = store.query(scope)
    assert result.facts == ()
    assert result.omitted_count == 0


def test_duplicate_content_with_conflicting_metadata_is_rejected(ledger, scope):
    store = FactStore(ledger)
    first = _fact(scope)
    store.observe(first)
    with pytest.raises(StorageError, match="conflicting metadata"):
        store.observe(
            _fact(
                scope,
                confidence=0.5,
                valid_from=first.valid_from,
            )
        )


def test_cross_tenant_fact_is_not_visible(ledger, scope):
    store = FactStore(ledger)
    other = ScopeRef(
        profile=scope.profile,
        platform=scope.platform,
        scope_id="T_OTHER",
        chat_id=scope.chat_id,
        principal_id=scope.principal_id,
        continuity_id=scope.continuity_id,
    )
    store.observe(_fact(other))
    assert store.query(scope).facts == ()


def test_cross_principal_fact_is_not_visible(ledger, identity_store, scope):
    other_principal = identity_store.create_principal("Other")
    other = ScopeRef(
        profile=scope.profile,
        platform=scope.platform,
        scope_id=scope.scope_id,
        chat_id=scope.chat_id,
        principal_id=other_principal.principal_id,
    )
    store = FactStore(ledger)
    store.observe(_fact(other))
    assert store.query(scope).facts == ()


def test_sensitivity_filter_runs_before_render(ledger, scope):
    store = FactStore(ledger)
    store.observe(_fact(scope, value="visible", sensitivity=Sensitivity.INTERNAL))
    store.observe(
        _fact(
            scope,
            value="secret-token",
            source_revision="sha-secret",
            sensitivity=Sensitivity.SECRET,
        )
    )
    result = store.query(scope, sensitivity_ceiling=Sensitivity.INTERNAL)
    assert [fact.value for fact in result.facts] == ["visible"]
    assert "secret-token" not in result.rendered_text
    assert result.omitted_count == 1


def test_channel_level_fact_flows_into_thread(ledger, scope):
    channel_scope = ScopeRef(
        profile=scope.profile,
        platform=scope.platform,
        scope_id=scope.scope_id,
        chat_id=scope.chat_id,
        principal_id=scope.principal_id,
        thread_id=None,
        continuity_id=scope.continuity_id,
    )
    store = FactStore(ledger)
    store.observe(_fact(channel_scope, value="channel-policy"))
    assert store.query(scope).facts[0].value == "channel-policy"


def test_explicit_workspace_wildcard_fact_can_flow_to_channel(ledger, scope):
    wildcard = ScopeRef(
        profile=scope.profile,
        platform=scope.platform,
        scope_id=scope.scope_id,
        chat_id="*",
        principal_id=scope.principal_id,
    )
    store = FactStore(ledger)
    store.observe(_fact(wildcard, value="workspace-policy"))
    assert store.query(scope).facts[0].value == "workspace-policy"


def test_top_authority_wins_over_lower_authority(ledger, scope):
    store = FactStore(ledger)
    store.observe(_fact(scope, value="old", authority=10, source_revision="low"))
    store.observe(_fact(scope, value="current", authority=20, source_revision="high"))
    result = store.query(scope)
    assert [item.value for item in result.facts] == ["current"]
    assert result.conflicts == ()


def test_equal_authority_conflict_is_preserved(ledger, scope):
    store = FactStore(ledger)
    first = store.observe(_fact(scope, value="open", source_revision="one"))
    second = store.observe(_fact(scope, value="closed", source_revision="two"))
    result = store.query(scope)
    assert {item.value for item in result.facts} == {"open", "closed"}
    assert result.conflicts == (tuple(sorted((first.fact_id, second.fact_id))),)


def test_equal_value_does_not_create_conflict(ledger, scope):
    store = FactStore(ledger)
    store.observe(
        _fact(scope, value="open", confidence=0.5, source_revision="one")
    )
    best = store.observe(
        _fact(scope, value="open", confidence=0.9, source_revision="two")
    )
    result = store.query(scope)
    assert [item.fact_id for item in result.facts] == [best.fact_id]
    assert result.conflicts == ()


def test_supersession_deactivates_prior_fact(ledger, scope):
    store = FactStore(ledger)
    old = store.observe(_fact(scope, value="old"))
    new = store.observe(
        _fact(
            scope,
            value="new",
            source_revision="sha-2",
            supersedes=old.fact_id,
        )
    )
    assert [item.fact_id for item in store.query(scope).facts] == [new.fact_id]


def test_supersession_cannot_cross_scope(ledger, scope):
    store = FactStore(ledger)
    old = store.observe(_fact(scope, value="old"))
    other = ScopeRef(
        profile=scope.profile,
        platform=scope.platform,
        scope_id=scope.scope_id,
        chat_id="C_OTHER",
        principal_id=scope.principal_id,
    )
    with pytest.raises(IncompleteScope, match="same scope"):
        store.observe(
            _fact(
                other,
                value="new",
                source_revision="sha-2",
                supersedes=old.fact_id,
            )
        )


def test_expired_fact_is_omitted(ledger, scope):
    now = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    store = FactStore(ledger)
    store.observe(
        _fact(
            scope,
            valid_from=utc_text(now - timedelta(hours=2)),
            valid_until=utc_text(now - timedelta(hours=1)),
        )
    )
    result = store.query(scope, now=now)
    assert result.facts == ()
    assert result.omitted_count == 1


def test_tag_filter_requires_all_requested_tags(ledger, scope):
    store = FactStore(ledger)
    store.observe(_fact(scope, tags=("project", "security")))
    assert len(store.query(scope, tags=("project", "security")).facts) == 1
    assert store.query(scope, tags=("project", "missing")).facts == ()


def test_fact_count_bound_reports_omission(ledger, scope):
    store = FactStore(ledger)
    for index in range(3):
        store.observe(
            _fact(
                scope,
                subject=f"subject:{index}",
                source_revision=f"sha-{index}",
            )
        )
    result = store.query(scope, max_facts=2)
    assert len(result.facts) == 2
    assert result.omitted_count == 1


def test_character_bound_never_emits_partial_fact(ledger, scope):
    store = FactStore(ledger)
    store.observe(_fact(scope, value="x" * 1000))
    result = store.query(scope, max_chars=256)
    assert result.facts == ()
    assert result.rendered_text == ""
    assert result.omitted_count == 1


def test_rendered_context_contains_provenance(ledger, scope):
    store = FactStore(ledger)
    fact = store.observe(_fact(scope))
    rendered = store.query(scope).rendered_text
    assert fact.fact_id in rendered
    assert "github_api" in rendered
    assert "sha-1" in rendered
    assert '"authority":10' in rendered
