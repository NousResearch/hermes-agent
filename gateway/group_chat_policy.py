"""Private receiver policy, selectively adapted from M73 group_chat_policy.py.

Source: dokterdok/hermes-agent 73fbc700664c56eabd9d7a55f178320662ef0c47.
No unstamped-primary fallback or legacy Home enrollment is retained.
"""
from gateway.session_group_messaging_identity import is_private_source, trusted_person
from gateway.slash_access import policy_from_extra

PRIVATE_ADMIN_REQUIRED = 'Group inventory requires explicit private-DM admin access.'


def private_admin_receiver(runner, source):
    """Return only a currently registered private receiver with explicit DM admin policy."""
    if not is_private_source(source):
        return None
    owner = runner._transport_owner(source)
    if owner is None:
        return None
    adapter, _ = owner
    config = getattr(adapter, 'config', None)
    extra = getattr(config, 'extra', None)
    policy = policy_from_extra(extra if isinstance(extra, dict) else {}, 'dm')
    # is_admin alone is intentionally permissive when slash gating is disabled.
    if not policy.enabled or str(source.user_id) not in policy.admin_user_ids:
        return None
    return adapter


def private_admin_event(runner, event):
    if not trusted_person(event):
        return None
    return private_admin_receiver(runner, event.source)
