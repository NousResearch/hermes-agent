"""Pre-auth gates on the Matrix adapter (PR #52354).

Covers two gates that run BEFORE the gateway's pairing/allowlist:

* ``_is_media_download_authorized`` — whether to fetch + cache attacker-supplied
  media bytes. Open by default, tightened when an allowlist / allow-all env is
  configured.
* ``_invite_auto_join_allowed`` — whether to auto-join an invite. Fail-closed and
  inviter-based, shared by the live invite event and the pending-invite
  reconciliation sweep so both apply the same policy.
"""

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    adapter._allowed_user_ids = set()
    adapter._allowed_room_ids = set()
    return adapter


@pytest.fixture(autouse=True)
def _clear_allow_all(monkeypatch):
    # These gates read allow-all envs; ensure a clean slate per test.
    monkeypatch.delenv("MATRIX_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)


class TestMediaDownloadAuthorized:
    def test_open_by_default_without_allowlist(self):
        adapter = _make_adapter()
        assert adapter._is_media_download_authorized("@anyone:example.org") is True

    def test_allowlisted_sender_authorized(self):
        adapter = _make_adapter()
        adapter._allowed_user_ids = {"@alice:example.org"}
        assert adapter._is_media_download_authorized("@alice:example.org") is True

    def test_unauthorized_sender_blocked_when_allowlist_set(self):
        adapter = _make_adapter()
        adapter._allowed_user_ids = {"@alice:example.org"}
        assert adapter._is_media_download_authorized("@mallory:evil.org") is False

    def test_empty_sender_blocked_when_allowlist_set(self):
        adapter = _make_adapter()
        adapter._allowed_user_ids = {"@alice:example.org"}
        assert adapter._is_media_download_authorized("") is False

    @pytest.mark.parametrize("env", ["MATRIX_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS"])
    def test_allow_all_env_overrides_allowlist(self, monkeypatch, env):
        # Both the Matrix-specific and the global allow-all env must be honored,
        # matching the gateway authorization layer. Previously only
        # GATEWAY_ALLOW_ALL_USERS was checked, so an operator who set only
        # MATRIX_ALLOW_ALL_USERS had media silently blocked.
        adapter = _make_adapter()
        adapter._allowed_user_ids = {"@alice:example.org"}
        monkeypatch.setenv(env, "true")
        assert adapter._is_media_download_authorized("@mallory:evil.org") is True


class TestInviteAutoJoinAllowed:
    def test_open_by_default_without_allowlist(self):
        # The reconcile sweep preserves the adapter's existing open-by-default
        # posture when no allowlist is configured.
        adapter = _make_adapter()
        assert adapter._invite_auto_join_allowed("!room:example.org", "@alice:example.org") is True

    def test_allowlisted_inviter_allowed(self):
        adapter = _make_adapter()
        adapter._allowed_user_ids = {"@alice:example.org"}
        assert adapter._invite_auto_join_allowed("!room:example.org", "@alice:example.org") is True

    def test_unauthorized_inviter_blocked(self):
        adapter = _make_adapter()
        adapter._allowed_user_ids = {"@alice:example.org"}
        assert adapter._invite_auto_join_allowed("!room:example.org", "@mallory:evil.org") is False

    def test_room_allowlist_alone_does_not_authorize(self):
        # Regression guard: a room-allowlist match must NOT authorize an
        # auto-join from an unauthorized inviter. The inviter is unverified and
        # DMs are exempt from the room allowlist, so a room-only match would let
        # an attacker manufacture an authorized-looking context.
        adapter = _make_adapter()
        adapter._allowed_room_ids = {"!listed:example.org"}
        adapter._allowed_user_ids = {"@alice:example.org"}
        assert adapter._invite_auto_join_allowed("!listed:example.org", "@mallory:evil.org") is False

    def test_room_allowlist_only_still_requires_authorized_inviter(self):
        # With ONLY a room allowlist configured (no user allowlist), an invite
        # to an allowlisted room from an unverified inviter is still declined —
        # the room match alone does not authorize the join.
        adapter = _make_adapter()
        adapter._allowed_room_ids = {"!listed:example.org"}
        assert adapter._invite_auto_join_allowed("!listed:example.org", "@mallory:evil.org") is False

    def test_empty_inviter_blocked(self):
        adapter = _make_adapter()
        adapter._allowed_user_ids = {"@alice:example.org"}
        assert adapter._invite_auto_join_allowed("!room:example.org", "") is False

    @pytest.mark.parametrize("env", ["MATRIX_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS"])
    def test_allow_all_env_permits_join(self, monkeypatch, env):
        adapter = _make_adapter()
        monkeypatch.setenv(env, "true")
        assert adapter._invite_auto_join_allowed("!room:example.org", "@mallory:evil.org") is True
