"""Operator-facing diagnostics for Feishu inbound admission.

Admission rejections are logged at DEBUG (busy groups with ``require_mention`` drop most
traffic by design). The one rejection that is almost never intended — the code-default
``allowlist`` policy with an empty ``FEISHU_ALLOWED_USERS`` — gets one WARNING that names
the keys to set, because under multiplex a secondary profile reads only its own ``.env``
and a launch-profile ``FEISHU_GROUP_POLICY=open`` is (deliberately) not inherited.
"""

from __future__ import annotations

from typing import Any, Collection, Mapping, Optional


def empty_allowlist_drop_warning(
    *,
    chat_id: str,
    group_rules: Mapping[str, Any],
    default_group_policy: str,
    allowed_group_users: Collection[str],
) -> Optional[str]:
    """Message for a group drop caused by the empty-allowlist default, else None.

    A per-chat ``group_rules`` entry, a non-``allowlist`` policy or a populated allowlist
    means the operator configured the deny; only the untouched default is surfaced.
    """
    if chat_id in group_rules or default_group_policy != "allowlist" or allowed_group_users:
        return None
    return (
        f"[Feishu] Dropped a group message in chat {chat_id}: group policy is 'allowlist' and "
        "FEISHU_ALLOWED_USERS is empty for this profile, so every human group message is rejected. "
        "Set FEISHU_GROUP_POLICY=open or FEISHU_ALLOWED_USERS=<open_id,...> in this profile's .env "
        "(under multiplex each profile reads only its own .env; the default profile's value is not "
        "inherited), or platforms.feishu.extra.group_rules in its config.yaml. "
        "Further drops are logged at DEBUG."
    )
