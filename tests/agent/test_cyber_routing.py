"""Tests for AgentCyber task-route classification."""

from agent.cyber_routing import (
    CyberRoute,
    ProviderPreference,
    classify_cyber_route,
)


def test_general_tasks_stay_on_general_route_without_secret_hosted_guard():
    decision = classify_cyber_route("Summarize these release notes and draft a changelog.")

    assert decision.route == CyberRoute.GENERAL
    assert decision.provider_preference == ProviderPreference.DEFAULT
    assert decision.requires_hosted_secret_confirmation is False
    assert decision.reason == "ordinary general task"


def test_credential_sensitive_tasks_prefer_local_and_guard_hosted_fallback():
    decision = classify_cyber_route(
        "Retrieve the documented Proxmox password from my approved notes and use it directly."
    )

    assert decision.route == CyberRoute.CREDENTIALS_SENSITIVE
    assert decision.provider_preference == ProviderPreference.LOCAL_OPEN_WEIGHT
    assert decision.requires_hosted_secret_confirmation is True
    assert "credential" in decision.reason


def test_break_glass_lockout_routes_to_incident_recovery():
    decision = classify_cyber_route("I'm locked out of VM 112; emergency access, get me back in.")

    assert decision.route == CyberRoute.IR_BREAKGLASS
    assert decision.provider_preference == ProviderPreference.LOCAL_OPEN_WEIGHT
    assert decision.requires_hosted_secret_confirmation is True
    assert "lockout" in decision.reason


def test_access_recovery_phrases_with_lab_hosts_are_breakglass():
    for prompt in (
        "recover access to VM 112",
        "restore VM112 access",
        "get back into the Proxmox host",
    ):
        decision = classify_cyber_route(prompt)
        assert decision.route == CyberRoute.IR_BREAKGLASS
        assert decision.provider_preference == ProviderPreference.LOCAL_OPEN_WEIGHT
        assert decision.requires_hosted_secret_confirmation is True


def test_malware_exploit_osint_and_destructive_routes_are_distinct():
    assert classify_cyber_route("Analyze this worm sample in the sandbox.").route == CyberRoute.MALWARE_RE
    assert classify_cyber_route("Test the exploit against owned lab VM 112.").route == CyberRoute.CYBER_LAB
    assert classify_cyber_route("Collect OSINT on this public company domain.").route == CyberRoute.OSINT
    assert classify_cyber_route("Wipe the disk and reset the firewall on the lab box.").route == CyberRoute.DESTRUCTIVE_HIGH_RISK


def test_explicit_operator_overrides_adjust_provider_preference_without_hiding_route():
    local_decision = classify_cyber_route("Use local model to analyze this malware sample.")
    azure_decision = classify_cyber_route("Use Azure to summarize this incident timeline.")
    cyber_decision = classify_cyber_route("Use cyber route for this credential recovery task.")

    assert local_decision.route == CyberRoute.MALWARE_RE
    assert local_decision.provider_preference == ProviderPreference.LOCAL_OPEN_WEIGHT
    assert local_decision.explicit_override == "local"

    assert azure_decision.route == CyberRoute.IR_BREAKGLASS
    assert azure_decision.provider_preference == ProviderPreference.HOSTED
    assert azure_decision.explicit_override == "azure"

    assert cyber_decision.route == CyberRoute.CREDENTIALS_SENSITIVE
    assert cyber_decision.provider_preference == ProviderPreference.LOCAL_OPEN_WEIGHT
    assert cyber_decision.explicit_override == "cyber"
