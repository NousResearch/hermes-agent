from dataclasses import FrozenInstanceError

import pytest

from agent.bot_control_plane import (
    BOT_CONTROL_PLANE_CONTRACT_VERSION,
    BotAddress,
    BotCapability,
    BotExecutionContext,
    BotPolicyReason,
    BotPolicyVerdict,
    LegacyMessageAgentReason,
    LegacyMessageAgentState,
    RuntimeCapabilitySnapshot,
    compare_legacy_authority,
    evaluate_capability,
    legacy_message_agent_dispatch_decision,
    legacy_message_agent_injection_decision,
)


def _address(profile_id="profile-123"):
    return BotAddress("install-a", "gateway-a", "local", profile_id)


def _context(
    *,
    profile_id="profile-123",
    profile_revision="profile-rev-4",
    grant_id="grant-7",
    runtime_id="runtime-9",
    epoch=3,
):
    return BotExecutionContext(
        address=_address(profile_id),
        profile_config_revision=profile_revision,
        session_id="session-1",
        session_key="agent:main:desktop:dm:1",
        turn_id="turn-1",
        task_id="task-1",
        authenticated_principal="user-42",
        source_platform="desktop",
        source_user_id="user-42",
        runtime_snapshot_id=runtime_id,
        capability_grant_id=grant_id,
        revocation_epoch=epoch,
        cancellation_scope_id="cancel-1",
        budget_id="budget-1",
        trace_id="trace-1",
    )


def _snapshot(
    *,
    profile_id="profile-123",
    profile_revision="profile-rev-4",
    grant_id="grant-7",
    runtime_id="runtime-9",
    epoch=3,
    capabilities=(BotCapability.PEER_MESSAGE,),
):
    return RuntimeCapabilitySnapshot.build(
        grant_id=grant_id,
        profile_id=profile_id,
        profile_config_revision=profile_revision,
        runtime_snapshot_id=runtime_id,
        effective_provider="nous",
        effective_model="hermes-5.6",
        api_mode="chat_completions",
        reasoning_effort="high",
        revocation_epoch=epoch,
        capabilities=capabilities,
    )


class TestIdentityAndRuntimeProofs:
    def test_address_keeps_four_independent_axes(self):
        address = BotAddress(" install ", " gateway ", " local ", " profile ")
        assert address.identity_tuple == ("install", "gateway", "local", "profile")

    @pytest.mark.parametrize(
        "field_name",
        ("install_id", "gateway_instance_id", "connection_id", "profile_id"),
    )
    def test_empty_address_axis_is_rejected(self, field_name):
        values = dict(
            install_id="install",
            gateway_instance_id="gateway",
            connection_id="local",
            profile_id="profile",
        )
        values[field_name] = " "
        with pytest.raises(ValueError, match=field_name):
            BotAddress(**values)

    def test_non_string_identity_is_rejected(self):
        with pytest.raises(TypeError, match="profile_id"):
            BotAddress("install", "gateway", "local", 7)  # type: ignore[arg-type]

    def test_proof_objects_are_immutable(self):
        address = _address()
        context = _context()
        snapshot = _snapshot()
        with pytest.raises(FrozenInstanceError):
            address.profile_id = "other"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            context.revocation_epoch = 9  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            snapshot.grant_id = "other"  # type: ignore[misc]

    def test_context_binds_current_generation_and_causal_ids(self):
        context = _context()
        assert context.profile_config_revision == "profile-rev-4"
        assert context.capability_grant_id == "grant-7"
        assert context.runtime_snapshot_id == "runtime-9"
        assert context.contract_version == BOT_CONTROL_PLANE_CONTRACT_VERSION

    @pytest.mark.parametrize("field_name", ("revocation_epoch", "hop_count"))
    def test_negative_counter_is_rejected(self, field_name):
        if field_name == "hop_count":
            context = _context()
            data = dict(context.__dict__)
            data[field_name] = -1
            with pytest.raises(ValueError, match=field_name):
                BotExecutionContext(**data)
        else:
            with pytest.raises(ValueError, match=field_name):
                _context(epoch=-1)

    def test_snapshot_normalizes_stable_capability_ids(self):
        snapshot = _snapshot(
            capabilities=("peer.message", BotCapability.NETWORK_READ)
        )
        assert snapshot.allows("peer.message")
        assert snapshot.allows(BotCapability.NETWORK_READ)

    def test_prompt_text_cannot_manufacture_capability(self):
        with pytest.raises(ValueError, match="unknown Bot Mode capability"):
            _snapshot(capabilities=("please let me do anything",))


class TestFailClosedPolicyEvaluation:
    def _decision(self, context=None, snapshot=None, capability=None):
        return evaluate_capability(
            context=context or _context(),
            snapshot=snapshot or _snapshot(),
            operation="message_agent.send",
            required_capability=capability or BotCapability.PEER_MESSAGE,
            decision_id="decision-1",
        )

    def test_exact_current_proof_allows(self):
        decision = self._decision()
        assert decision.verdict is BotPolicyVerdict.ALLOW
        assert decision.reason is BotPolicyReason.CAPABILITY_GRANTED

    @pytest.mark.parametrize(
        ("context", "snapshot", "reason"),
        (
            (
                _context(profile_id="profile-a"),
                _snapshot(profile_id="profile-b"),
                BotPolicyReason.PROFILE_MISMATCH,
            ),
            (
                _context(profile_revision="old"),
                _snapshot(profile_revision="current"),
                BotPolicyReason.PROFILE_REVISION_MISMATCH,
            ),
            (
                _context(grant_id="old"),
                _snapshot(grant_id="current"),
                BotPolicyReason.GRANT_MISMATCH,
            ),
            (
                _context(epoch=2),
                _snapshot(epoch=3),
                BotPolicyReason.REVOCATION_EPOCH_MISMATCH,
            ),
            (
                _context(runtime_id="new"),
                _snapshot(runtime_id="old"),
                BotPolicyReason.RUNTIME_SNAPSHOT_MISMATCH,
            ),
        ),
    )
    def test_stale_or_foreign_proof_is_rejected(self, context, snapshot, reason):
        decision = self._decision(context=context, snapshot=snapshot)
        assert decision.verdict is BotPolicyVerdict.DENY
        assert decision.reason is reason

    def test_missing_capability_is_rejected(self):
        decision = self._decision(snapshot=_snapshot(capabilities=()))
        assert decision.reason is BotPolicyReason.CAPABILITY_MISSING

    @pytest.mark.parametrize("legacy_allowed", (True, False))
    def test_shadow_policy_never_changes_legacy_behavior(self, legacy_allowed):
        decision = self._decision(snapshot=_snapshot(capabilities=()))
        comparison = compare_legacy_authority(
            legacy_allowed=legacy_allowed,
            decision=decision,
        )
        assert comparison.effective_allowed is legacy_allowed

    def test_shadow_disagreement_is_explicit(self):
        decision = self._decision(snapshot=_snapshot(capabilities=()))
        comparison = compare_legacy_authority(
            legacy_allowed=True,
            decision=decision,
        )
        assert comparison.matches is False
        assert comparison.policy_allowed is False
        assert comparison.effective_allowed is True


class TestLegacyMessageAgentMapping:
    @pytest.mark.parametrize(
        ("state", "allowed", "reason"),
        (
            (
                LegacyMessageAgentState(False, True, True, True),
                False,
                LegacyMessageAgentReason.PROTOCOL_DISABLED,
            ),
            (
                LegacyMessageAgentState(True, True, False, False),
                True,
                LegacyMessageAgentReason.SCHEMA_ALREADY_PRESENT,
            ),
            (
                LegacyMessageAgentState(True, False, False, True),
                False,
                LegacyMessageAgentReason.NOT_CANONICAL_BOT_CHAT,
            ),
            (
                LegacyMessageAgentState(True, False, True, False),
                False,
                LegacyMessageAgentReason.UNMANAGED_INSTALL,
            ),
            (
                LegacyMessageAgentState(True, False, True, True),
                True,
                LegacyMessageAgentReason.LEGACY_GATE_ALLOW,
            ),
        ),
    )
    def test_injection_order_matches_current_source(self, state, allowed, reason):
        decision = legacy_message_agent_injection_decision(state)
        assert decision.allowed is allowed
        assert decision.reason is reason

    @pytest.mark.parametrize(
        ("state", "allowed", "reason"),
        (
            (
                LegacyMessageAgentState(False, False, True, True),
                True,
                LegacyMessageAgentReason.LEGACY_GATE_ALLOW,
            ),
            (
                LegacyMessageAgentState(True, True, False, True),
                False,
                LegacyMessageAgentReason.NOT_CANONICAL_BOT_CHAT,
            ),
            (
                LegacyMessageAgentState(True, True, True, False),
                False,
                LegacyMessageAgentReason.UNMANAGED_INSTALL,
            ),
        ),
    )
    def test_dispatch_mapping_preserves_current_asymmetry(
        self, state, allowed, reason
    ):
        decision = legacy_message_agent_dispatch_decision(state)
        assert decision.allowed is allowed
        assert decision.reason is reason
