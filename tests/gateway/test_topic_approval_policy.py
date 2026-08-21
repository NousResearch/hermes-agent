"""Regression tests for exact channel/topic approval-mode resolution."""

import tools.approval as approval


def test_exact_topic_override_beats_global_without_affecting_siblings():
    config = {
        "approvals": {
            "mode": "smart",
            "channels": {"telegram": {"-1003703764467:1666": {"mode": "off"}}},
        }
    }

    assert approval.resolve_approval_mode(config, "telegram", "-1003703764467", "1666") == "off"
    assert approval.resolve_approval_mode(config, "telegram", "-1003703764467", "130") == "smart"
    assert approval.resolve_approval_mode(config, "telegram", "457500237", None) == "smart"


def test_malformed_topic_policy_falls_back_to_global_smart():
    config = {"approvals": {"mode": "smart", "channels": {"telegram": {"-100:1666": "off"}}}}

    assert approval.resolve_approval_mode(config, "telegram", "-100", "1666") == "smart"


def test_bare_yaml_off_is_accepted_as_topic_bypass():
    config = {"approvals": {"mode": "smart", "channels": {"telegram": {"-100:1666": {"mode": False}}}}}

    assert approval.resolve_approval_mode(config, "telegram", "-100", "1666") == "off"


def test_context_override_is_scoped_and_reset():
    outer_token = approval.set_current_approval_mode_override("smart")
    try:
        inner_token = approval.set_current_approval_mode_override("off")
        try:
            assert approval._get_approval_mode() == "off"
        finally:
            approval.reset_current_approval_mode_override(inner_token)
        assert approval._get_approval_mode() == "smart"
    finally:
        approval.reset_current_approval_mode_override(outer_token)
