"""QQ button-approval authorization must accept the chat_type the gateway
actually writes into session keys.

Regression: ``_is_authorized_interaction_for_session`` only accepted
``chat_type == "c2c"`` (QQ's API vocabulary for a 1:1 scene), but Hermes builds
QQ session keys with ``chat_type == "dm"`` (``"dm" if is_dm else "group"`` in
the inbound handler, mirrored by ``cron/scheduler.py``). Every DM approval
button click therefore fell through to ``return False`` and was logged as
"Rejected unauthorized approval click" — even though the operator openid was
byte-identical to the session's chat_id. QQ DM approvals could never be granted
by button; users had to fall back to ``/approve``.

These are behaviour contracts, not snapshots: the invariant is "the chat_type
the session key generator emits must be authorizable", plus two bounds —
widening the vocabulary must not widen *who* is allowed, and it must not extend
to spellings this adapter never produces.
"""
import types

import pytest

from gateway.platforms.qqbot.adapter import QQAdapter

OWNER = "28888E9E014BF40CB1AF90827E9AD87B"
STRANGER = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"


def _event(operator, group=None, guild=None):
    return types.SimpleNamespace(
        operator_openid=operator, group_openid=group, guild_id=guild,
    )


def _authorized(session_key, operator, group=None, guild=None):
    return QQAdapter._is_authorized_interaction_for_session(
        QQAdapter, _event(operator, group, guild), session_key,
    )


# The vocabulary that must authorize a 1:1 chat, and no wider. "dm" is what
# gateway-built session keys carry (adapter.py:1301); "c2c" is what the
# update-prompt path puts in its key via ``event.scene``. Both reach this
# authorizer in production.
ONE_TO_ONE_CHAT_TYPES = ["dm", "c2c"]


@pytest.mark.parametrize("chat_type", ONE_TO_ONE_CHAT_TYPES)
def test_owner_may_approve_own_dm_regardless_of_chat_type_spelling(chat_type):
    assert _authorized(f"agent:main:qqbot:{chat_type}:{OWNER}", OWNER) is True


@pytest.mark.parametrize("chat_type", ONE_TO_ONE_CHAT_TYPES)
def test_stranger_may_not_approve_someone_elses_dm(chat_type):
    """Widening the chat_type vocabulary must not widen authorization."""
    assert _authorized(f"agent:main:qqbot:{chat_type}:{OWNER}", STRANGER) is False


@pytest.mark.parametrize("chat_type", ["direct", "private", "im", "user"])
def test_chat_types_this_adapter_never_emits_are_not_authorized(chat_type):
    """Keep the authorization vocabulary minimal.

    ``gateway/slash_access.py`` treats "direct"/"private" as DM aliases, but the
    QQ adapter only ever builds "dm", "group" and "guild" (adapter.py:1301,
    1366, 1441). Accepting a spelling this adapter cannot produce would widen
    the authorization surface for nothing, so it must stay rejected.
    """
    assert _authorized(f"agent:main:qqbot:{chat_type}:{OWNER}", OWNER) is False


def test_dm_chat_type_written_by_session_key_builder_is_authorizable():
    """Contract between the key builder and the authorizer.

    ``cron/scheduler.py`` documents that QQ session keys mirror the inbound
    handler's ``"dm" if is_dm else "group"``. Whatever that expression yields
    for a DM must be accepted here, or button approvals silently break.
    """
    is_dm = True
    chat_type = "dm" if is_dm else "group"  # the builder's own expression
    assert _authorized(f"agent:main:qqbot:{chat_type}:{OWNER}", OWNER) is True


def test_group_requires_matching_group_and_member():
    key = f"agent:main:qqbot:group:GROUP123:{OWNER}"
    assert _authorized(key, OWNER, group="GROUP123") is True
    # Right group, wrong member
    assert _authorized(key, STRANGER, group="GROUP123") is False
    # Right member, wrong group
    assert _authorized(key, OWNER, group="OTHERGROUP") is False


def test_other_platforms_and_malformed_keys_are_rejected():
    assert _authorized(f"agent:main:telegram:dm:{OWNER}", OWNER) is False
    assert _authorized("agent:main:qqbot:dm:", OWNER) is False
    assert _authorized("garbage", OWNER) is False
    assert _authorized(f"agent:main:qqbot:dm:{OWNER}", "") is False
    assert _authorized(f"agent:main:qqbot:unknown_scene:{OWNER}", OWNER) is False
