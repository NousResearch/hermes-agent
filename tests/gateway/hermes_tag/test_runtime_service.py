from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from gateway.hermes_tag import (
    ConfigurationError,
    ContinuityMode,
    DecisionOutcome,
    Fact,
    HermesTagConfig,
    HermesTagKernel,
    HermesTagService,
    IncompleteScope,
    LeaseAuthority,
    LeaseError,
    PolicyRule,
    ReplayDetected,
    RuntimeAuthority,
    Sensitivity,
    bind_admission,
    bind_authority,
    capture_authority,
    clear_runtime_context,
    current_admission,
    current_decision,
    current_lease,
    new_id,
)


def _source(**overrides):
    value = {
        "platform": "slack",
        "profile": "default",
        "team_id": "T1",
        "channel_id": "C1",
        "thread_ts": "TH1",
        "user_id": "U1",
        "display_name": "Axl",
    }
    value.update(overrides)
    return value


def test_disabled_service_touches_no_state(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config=None,
    )
    assert service.enabled is False
    assert service.ledger is None
    assert list(tmp_path.rglob("*.db")) == []


def test_enabled_service_uses_profile_local_database(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="worker",
        raw_config={"enabled": True},
    )
    assert service.enabled is True
    assert service.ledger is not None
    assert service.ledger.path == (
        tmp_path / "profiles" / "worker" / "hermes-tag.db"
    ).resolve()
    assert service.ledger.path.exists()


def test_secret_reference_is_resolved_not_persisted(tmp_path: Path):
    secret = b"super-secret-signing-material-1234567890"
    seen = []

    def resolve(reference: str):
        seen.append(reference)
        return secret

    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={
            "enabled": True,
            "leases": {"signing_secret_ref": "vault://hermes/tag"},
        },
        secret_resolver=resolve,
    )
    assert seen == ["vault://hermes/tag"]
    assert service.require_kernel().leases is not None
    assert secret not in service.ledger.path.read_bytes()


def test_missing_secret_resolver_fails_closed(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="no secret resolver"):
        HermesTagService.build(
            hermes_home=tmp_path,
            profile="default",
            raw_config={
                "enabled": True,
                "leases": {"signing_secret_ref": "vault://missing"},
            },
        )


def test_short_secret_fails_closed(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="32 bytes"):
        HermesTagService.build(
            hermes_home=tmp_path,
            profile="default",
            raw_config={
                "enabled": True,
                "leases": {"signing_secret_ref": "vault://short"},
            },
            secret_resolver=lambda _ref: b"short",
        )


def test_shadow_admission_returns_bounded_error(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={"enabled": True, "shadow": True},
    )
    outcome = service.shadow_admit_session_source(
        _source(user_id=None), event_id="slack-event-1"
    )
    assert outcome.admission is None
    assert outcome.error_class == "IncompleteScope"


def test_shadow_admission_contains_non_kernel_errors(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={
            "enabled": True,
            "shadow": True,
            "continuity": {"mode": "project"},
        },
    )
    outcome = service.shadow_admit_session_source(
        _source(), event_id="slack-event-project"
    )
    assert outcome.admission is None
    assert outcome.error_class == "ValueError"


def test_non_shadow_admission_propagates_failure(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={"enabled": True, "shadow": False},
    )
    with pytest.raises(IncompleteScope):
        service.shadow_admit_session_source(_source(user_id=None))


def test_admission_resolves_guest_and_continuity(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={
            "enabled": True,
            "continuity": {"mode": "workspace"},
        },
    )
    result = service.admit_session_source(_source(), event_id="slack-event-1")
    assert result.admission.principal.guest is True
    assert result.admission.scope.scope_id == "T1"
    assert result.admission.scope.chat_id == "C1"
    assert result.admission.continuity_id == result.continuity.continuity_id
    assert result.admission.shadow is True


def test_duplicate_provider_event_never_creates_second_turn(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={"enabled": True},
    )
    first = service.admit_session_source(_source(), event_id="slack-event-1")
    with pytest.raises(ReplayDetected, match="already admitted"):
        service.admit_session_source(_source(), event_id="slack-event-1")
    connection = service.ledger.connection()
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM hermes_tag_turn_events WHERE event_id='slack-event-1'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == 1
    assert first.admission.admission_id


def test_failed_admission_releases_event_reservation(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={
            "enabled": True,
            "shadow": False,
            "continuity": {"mode": "project"},
        },
    )
    source = _source()
    with pytest.raises(ValueError, match="project"):
        service.admit_session_source(source, event_id="slack-event-retry")
    result = service.admit_session_source(
        source,
        event_id="slack-event-retry",
        continuity_mode=ContinuityMode.ISOLATED,
    )
    assert result.admission.admission_id


def test_context_is_ephemeral_and_source_backed(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={
            "enabled": True,
            "context": {"enabled": True, "sensitivity_ceiling": "internal"},
            "continuity": {"mode": "principal"},
        },
    )
    first = service.admit_session_source(_source(), event_id="event-1")
    fact = Fact(
        fact_id=new_id("fact"),
        subject="campaign:slack",
        predicate="status",
        value="in progress",
        scope=first.admission.scope,
        source_type="github_api",
        source_id="NousResearch/hermes-agent#79772",
        source_revision="4a5b6dd",
        confidence=1.0,
        authority=100,
        sensitivity=Sensitivity.INTERNAL,
    )
    service.require_kernel().observe_fact(fact)
    second_source = _source(channel_id="C2", thread_ts="TH2")
    second = service.admit_session_source(second_source, event_id="event-2")
    assert second.admission.context.facts == ()  # channel-scoped fact does not leak
    third = service.admit_session_source(_source(), event_id="event-3")
    assert [item.fact_id for item in third.admission.context.facts] == [fact.fact_id]
    assert "github_api" in third.admission.context.rendered_text


def test_two_profiles_have_independent_principal_stores(tmp_path: Path):
    default = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={"enabled": True},
    )
    worker = HermesTagService.build(
        hermes_home=tmp_path,
        profile="worker",
        raw_config={"enabled": True},
    )
    one = default.admit_session_source(_source(profile="default"), event_id="evt-1")
    two = worker.admit_session_source(_source(profile="worker"), event_id="evt-1")
    assert one.admission.principal.principal_id != two.admission.principal.principal_id
    assert default.ledger.path != worker.ledger.path


def _admissions(tmp_path: Path):
    service = HermesTagService.build(
        hermes_home=tmp_path,
        profile="default",
        raw_config={"enabled": True},
    )
    one = service.admit_session_source(_source(user_id="U1"), event_id="evt-1")
    two = service.admit_session_source(
        _source(user_id="U2", channel_id="C2", thread_ts="TH2"),
        event_id="evt-2",
    )
    return one.admission, two.admission


def test_runtime_context_restores_after_exit(tmp_path: Path):
    one, two = _admissions(tmp_path)
    clear_runtime_context()
    with bind_admission(one):
        assert current_admission(required=True) == one
        with bind_admission(two):
            assert current_admission(required=True) == two
        assert current_admission(required=True) == one
    assert current_admission() is None


def test_runtime_context_required_fails_closed():
    clear_runtime_context()
    with pytest.raises(LeaseError, match="no Hermes Tag turn admission"):
        current_admission(required=True)


@pytest.mark.asyncio
async def test_contextvars_do_not_bleed_between_tasks(tmp_path: Path):
    one, two = _admissions(tmp_path)

    async def observe(admission):
        with bind_admission(admission):
            await asyncio.sleep(0)
            return current_admission(required=True).principal.principal_id

    first, second = await asyncio.gather(observe(one), observe(two))
    assert first == one.principal.principal_id
    assert second == two.principal.principal_id
    assert first != second
    assert current_admission() is None


def test_capture_authority_binds_all_three_objects(tmp_path: Path, ledger):
    one, _ = _admissions(tmp_path)
    kernel = HermesTagKernel(
        ledger,
        HermesTagConfig.from_mapping({"enabled": True}),
        rules=(PolicyRule("allow", DecisionOutcome.ALLOW, reason="allow"),),
        lease_authority=LeaseAuthority(b"a" * 32),
    )
    authorization = kernel.authorize(
        one.principal,
        capability="context.read",
        action="read",
        resource="context",
        arguments={},
        scope=one.scope,
    )
    authority = RuntimeAuthority(
        admission=one,
        decision=authorization.decision,
        lease=authorization.lease,
    )
    with bind_authority(authority):
        captured = capture_authority()
        assert captured == authority
        assert current_decision() == authorization.decision
        assert current_lease() == authorization.lease
    assert current_decision() is None
    assert current_lease() is None
