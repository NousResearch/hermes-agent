"""Signal group chat-allowlist authorization (Bug 1 / PR #53348 review).

Signal emits ``chat_id="group:<id>"`` while ``SIGNAL_GROUP_ALLOWED_USERS`` holds
the raw ``<id>`` (also exposed as ``chat_id_alt``). The group-allowlist bypass in
``_is_user_authorized`` must therefore match the raw / ``group:``-stripped id, not
only ``source.chat_id`` — otherwise an explicitly configured group stays
unauthorized and only ``*`` works.
"""

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform
from gateway.session import SessionSource


def _authz():
    obj = object.__new__(GatewayAuthorizationMixin)
    # Reach the group-allowlist branch without the trusted-upstream shortcut.
    obj._adapter_authorization_is_upstream = lambda platform: False
    return obj


def _signal_group_source(chat_id, chat_id_alt=None, user_id=None):
    return SessionSource(
        platform=Platform.SIGNAL,
        chat_id=chat_id,
        chat_type="group",
        chat_id_alt=chat_id_alt,
        user_id=user_id,
    )


def test_explicit_group_id_matches_via_chat_id_alt(monkeypatch):
    """Raw configured id authorizes the group even though chat_id is 'group:<id>'."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc123==")
    src = _signal_group_source(chat_id="group:abc123==", chat_id_alt="abc123==")
    assert _authz()._is_user_authorized(src) is True


def test_explicit_group_id_matches_by_stripping_group_prefix(monkeypatch):
    """Even without chat_id_alt, the 'group:' prefix is stripped for matching."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc123==")
    src = _signal_group_source(chat_id="group:abc123==", chat_id_alt=None)
    assert _authz()._is_user_authorized(src) is True


def test_wildcard_authorizes_any_group(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    src = _signal_group_source(chat_id="group:whatever==", chat_id_alt="whatever==")
    assert _authz()._is_user_authorized(src) is True


def test_unlisted_group_is_denied(monkeypatch):
    """A group not in the allowlist falls through; anonymous group post -> deny."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "someOtherGroup==")
    src = _signal_group_source(chat_id="group:abc123==", chat_id_alt="abc123==", user_id=None)
    assert _authz()._is_user_authorized(src) is False
