"""A group rule's implicit policy is derived from its configured lists.

Before the fix, omitting ``policy`` in a ``group_rules`` entry defaulted the
policy to "open", so an explicitly configured allowlist/blacklist was silently
ignored during group admission. The implicit default must now follow the
configuration:

    allowlist present  -> "allowlist"
    blacklist present  -> "blacklist"
    neither list       -> "open"
    explicit policy    -> verbatim (lowercased)
"""

from __future__ import annotations

from plugins.platforms.feishu.adapter import FeishuAdapter


def _load(rules: dict) -> dict:
    settings = FeishuAdapter._load_settings({"group_rules": rules})
    return settings.group_rules


def test_allowlist_without_policy_defaults_to_allowlist():
    rules = _load({"oc_chat": {"allowlist": ["ou_a", "ou_b"]}})
    assert rules["oc_chat"].policy == "allowlist"


def test_blacklist_without_policy_defaults_to_blacklist():
    rules = _load({"oc_chat": {"blacklist": ["ou_bad"]}})
    assert rules["oc_chat"].policy == "blacklist"


def test_rule_without_lists_stays_open():
    rules = _load({"oc_chat": {}})
    assert rules["oc_chat"].policy == "open"


def test_explicit_policy_wins_over_lists():
    rules = _load({"oc_chat": {"policy": "Disabled", "allowlist": ["ou_a"]}})
    assert rules["oc_chat"].policy == "disabled"


def test_allowlist_takes_precedence_when_both_lists_present():
    rules = _load({"oc_chat": {"allowlist": ["ou_a"], "blacklist": ["ou_b"]}})
    assert rules["oc_chat"].policy == "allowlist"


def test_empty_allowlist_does_not_imply_allowlist_policy():
    # An explicitly empty list is indistinguishable from "not configured" at
    # this layer, so it keeps the historical "open" default rather than
    # flipping a chat into a silent deny-all.
    rules = _load({"oc_chat": {"allowlist": []}})
    assert rules["oc_chat"].policy == "open"
