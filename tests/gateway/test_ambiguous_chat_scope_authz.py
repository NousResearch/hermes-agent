"""A blank/unknown ``chat_type`` must satisfy BOTH scope readings of scope-split authorization.

Relay frames and restored session rows can carry ``chat_type=""``/``None``. Scope-split config —
``allow_from`` vs ``group_allow_from``, ``dm_policy`` vs ``group_policy`` — picks a key by
``is_group``, which is False for blank values. Reading the DM axis on an ambiguous source lets
DM-scoped config admit what may be a group message (and vice versa), the same defect class as
``slash_access.policy_for_source``'s group-policy wholesale pick. The fix intersects the two scope
readings for ambiguous sources.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import Platform, SessionSource


def _source(chat_type, user_id="alice"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=user_id,
        chat_id="chat-1",
        user_name="tester",
        chat_type=chat_type,
    )


@pytest.fixture
def runner():
    r = object.__new__(GatewayRunner)
    r.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    r.pairing_store = MagicMock()
    r.pairing_store.is_approved.return_value = False
    return r


# ─────────────────────────────────────────────────────────────────────
# _adapter_extra_allowlist_authorizes: allow_from vs group_allow_from
# ─────────────────────────────────────────────────────────────────────


def _extra_runner(runner, extra):
    adapter = SimpleNamespace(config=SimpleNamespace(extra=extra))
    runner._delivery_adapter_for = lambda source: adapter
    runner._authorization_adapter = lambda *a, **kw: adapter
    return runner


@pytest.mark.parametrize("chat_type", ["", None, "   "])
def test_ambiguous_source_dm_list_cannot_admit(runner, chat_type):
    """An ambiguous source carrying only a DM allowlist entry must not be admitted: the source
    may be a group message, which the operator gated with a different list."""
    _extra_runner(runner, {"allow_from": ["alice"], "group_allow_from": ["bob"]})
    assert runner._adapter_extra_allowlist_authorizes(_source(chat_type), "alice", False) is False


@pytest.mark.parametrize("chat_type", ["", None])
def test_ambiguous_source_group_list_cannot_admit(runner, chat_type):
    """Mirror image: a group_allow_from entry alone must not admit an ambiguous source that may
    be a DM."""
    _extra_runner(runner, {"group_allow_from": ["bob"]})
    assert runner._adapter_extra_allowlist_authorizes(_source(chat_type, "bob"), "bob", False) is False


@pytest.mark.parametrize("chat_type", ["", None])
def test_ambiguous_source_dual_listed_admits(runner, chat_type):
    """A principal allowed by BOTH scope lists keeps its grant on an ambiguous source."""
    _extra_runner(runner, {"allow_from": ["alice"], "group_allow_from": ["alice"]})
    assert runner._adapter_extra_allowlist_authorizes(_source(chat_type), "alice", False) is True


def test_definite_scopes_unchanged(runner):
    """A real DM still reads allow_from; a real group still reads group_allow_from."""
    _extra_runner(runner, {"allow_from": ["alice"], "group_allow_from": ["bob"]})
    assert runner._adapter_extra_allowlist_authorizes(_source("dm"), "alice", False) is True
    assert runner._adapter_extra_allowlist_authorizes(_source("dm", "bob"), "bob", False) is False
    assert runner._adapter_extra_allowlist_authorizes(_source("group", "bob"), "bob", True) is True
    assert runner._adapter_extra_allowlist_authorizes(_source("group"), "alice", True) is False


# ─────────────────────────────────────────────────────────────────────
# _own_policy_authorizes: dm_policy vs group_policy
# ─────────────────────────────────────────────────────────────────────


def _policy_runner(runner, *, dm_policy, group_policy, dm_allowed=lambda uid: True):
    adapter = SimpleNamespace(
        enforces_own_access_policy=True,
        _dm_policy=dm_policy,
        _group_policy=group_policy,
        _groups={},
        _is_dm_allowed=dm_allowed,
    )
    runner._authorization_adapter = lambda *a, **kw: adapter
    return runner


@pytest.mark.parametrize("chat_type", ["", None])
def test_ambiguous_source_dm_policy_cannot_admit_when_group_gated(runner, chat_type):
    """dm_policy=allowlist admits alice for DMs, but group_policy=disabled must not be overridden
    by an ambiguous source that may be a group message."""
    _policy_runner(runner, dm_policy="allowlist", group_policy="disabled")
    verdict = runner._own_policy_authorizes(_source(chat_type), "alice", False, None)
    assert verdict is None  # defers; a real group reading gives no verdict, DM admits alone can't


@pytest.mark.parametrize("chat_type", ["", None])
def test_ambiguous_source_denied_when_dm_check_fails(runner, chat_type):
    """An ambiguous source the DM check rejects is denied even when the group scope admits."""
    _policy_runner(runner, dm_policy="allowlist", group_policy="allowlist", dm_allowed=lambda uid: False)
    assert runner._own_policy_authorizes(_source(chat_type), "alice", False, None) is False


@pytest.mark.parametrize("chat_type", ["", None])
def test_ambiguous_source_admits_when_both_scopes_admit(runner, chat_type):
    _policy_runner(runner, dm_policy="allowlist", group_policy="allowlist")
    assert runner._own_policy_authorizes(_source(chat_type), "alice", False, None) is True


def test_definite_scope_policy_unchanged(runner):
    """Real dm/group sources keep the single-scope reading."""
    _policy_runner(runner, dm_policy="allowlist", group_policy="disabled")
    assert runner._own_policy_authorizes(_source("dm"), "alice", False, None) is True
    assert runner._own_policy_authorizes(_source("group"), "alice", True, None) is None


# ─────────────────────────────────────────────────────────────────────
# End-to-end through _is_user_authorized
# ─────────────────────────────────────────────────────────────────────


def test_is_user_authorized_ambiguous_source_not_admitted_by_dm_list(runner, monkeypatch):
    """A relayed/restored source with blank chat_type whose sender is only in the DM-scope
    ``allow_from`` is unauthorized — the DM list cannot speak for a possibly-group source."""
    for var in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_ALLOWED_CHATS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "TELEGRAM_ALLOW_BOTS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(var, raising=False)
    _extra_runner(runner, {"allow_from": ["alice"], "group_allow_from": ["bob"]})
    assert runner._is_user_authorized(_source("")) is False
    assert runner._is_user_authorized(_source("dm")) is True
