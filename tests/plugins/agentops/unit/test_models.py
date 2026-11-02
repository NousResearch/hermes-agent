from __future__ import annotations

from plugins.agentops.control.models import AuthorityMode, ControlPlaneHealth


def test_health_serializes_only_observe_only_authority():
    health = ControlPlaneHealth(
        ready=True,
        authority_mode=AuthorityMode.OBSERVE_ONLY,
        safe_start_reasons=("config_missing",),
        store_available=False,
        audit_chain_valid=None,
        event_count=0,
        spool_depth=0,
    )

    assert health.to_dict() == {
        "ready": True,
        "authority_mode": "observe_only",
        "safe_start_reasons": ["config_missing"],
        "store_available": False,
        "audit_chain_valid": None,
        "event_count": 0,
        "spool_depth": 0,
        "spool_bytes": 0,
        "spool_quarantine_bytes": 0,
        "spool_healthy": True,
        "global_write_enabled": False,
    }
