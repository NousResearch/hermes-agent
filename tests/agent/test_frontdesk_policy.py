"""Temporary shim: legacy import path agent.frontdesk_policy still works."""

from agent.frontdesk_policy import (
    FrontdeskPolicyDecision,
    FrontdeskRecommendation,
    classify_request,
    fingerprint,
)
from agent.concierge_policy import ConciergePolicyDecision, ConciergeRecommendation


def test_shim_types_are_aliases():
    assert FrontdeskPolicyDecision is ConciergePolicyDecision
    assert FrontdeskRecommendation is ConciergeRecommendation


def test_shim_classify_and_fingerprint():
    d = classify_request("stop")
    assert d.is_stop
    assert fingerprint("x", frontdesk_mode_active=True) == fingerprint(
        "x", concierge_mode_active=True
    )
