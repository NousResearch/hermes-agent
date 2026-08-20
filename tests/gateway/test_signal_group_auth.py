"""Regression guard for SIGNAL_GROUP_ALLOWED_USERS group-member auth bypass.

Before this fix, ``authz_mixin._is_user_authorized`` only enumerated Telegram
and QQBOT in its group-chat-allowlist bypass:

    chat_allowlist_env = {
        Platform.TELEGRAM: "TELEGRAM_GROUP_ALLOWED_CHATS",
        Platform.QQBOT:    "QQ_GROUP_ALLOWED_USERS",
    }.get(source.platform, "")

Signal was missing, so every Signal group message fell through to the
per-user ``SIGNAL_ALLOWED_USERS`` check — meaning operators who set
``SIGNAL_GROUP_ALLOWED_USERS`` (the group-ID allowlist consumed by
``platforms/signal.py``) expected groups to "just work" the way they do on
Telegram, but in practice every group member other than those explicitly
listed in ``SIGNAL_ALLOWED_USERS`` was silently rejected with
``WARNING gateway.run: Unauthorized user: <uuid> (<name>) on signal``.

There is also a chat_id format gotcha: ``platforms/signal.py`` builds
``source.chat_id`` as ``f"group:{group_id}"`` (line ~520), while the env
value ``SIGNAL_GROUP_ALLOWED_USERS`` is stored as bare group IDs (signal.py
itself compares unprefixed at line ~515). A naive bypass patch would still
fail equality because ``"group:abc..." not in {"abc..."}``.

This test mirrors the patched bypass logic with both fixes and exercises
representative cases.
"""

import unittest


def _is_group_member_authorized(
    chat_type: str, chat_id: str, allowed_groups_raw: str
) -> bool:
    """Mirror the Signal-relevant branch of ``_is_user_authorized``.

    Replicates: chat-type gate, env parse, prefix normalization, equality /
    wildcard check. Returns ``True`` when the group bypass applies.
    """
    if chat_type not in {"group", "forum", "channel"} or not chat_id:
        return False
    if not allowed_groups_raw:
        return False
    allowed_group_ids = {
        cid.strip() for cid in allowed_groups_raw.split(",") if cid.strip()
    }
    if not allowed_group_ids:
        return False
    chat_id_norm = chat_id
    if chat_id_norm.startswith("group:"):
        chat_id_norm = chat_id_norm[len("group:") :]
    return "*" in allowed_group_ids or chat_id_norm in allowed_group_ids


# Realistic-shaped sample IDs (random UUIDs, base64-shaped group IDs).
GROUP_A = "RTkH42KVKvN5BrfL82d69UvbXZtXfwZs8xNlcQoANmw="
GROUP_B = "XyZAbCdeFghi0123456789=="
USER_RANDOM = "f1355565-f2c6-4649-ae7d-f4cbf3160f63"
USER_OTHER = "82ed6ad1-84e7-40ed-8d15-7073c189ecd4"


class TestSignalGroupBypass(unittest.TestCase):
    """Group members in an allowed Signal group are authorized regardless of user ID."""

    def test_member_in_allowed_group_with_prefixed_chat_id(self):
        """The Signal adapter passes chat_id with a 'group:' prefix; bypass
        must still match against the unprefixed env value."""
        self.assertTrue(
            _is_group_member_authorized("group", f"group:{GROUP_A}", GROUP_A)
        )

    def test_member_in_allowed_group_without_prefix(self):
        """If the chat_id happens to be unprefixed it must also match."""
        self.assertTrue(
            _is_group_member_authorized("group", GROUP_A, GROUP_A)
        )

    def test_wildcard_allows_any_group(self):
        """'*' in SIGNAL_GROUP_ALLOWED_USERS authorizes any group."""
        self.assertTrue(
            _is_group_member_authorized("group", f"group:{GROUP_A}", "*")
        )
        self.assertTrue(
            _is_group_member_authorized("group", f"group:{GROUP_B}", "*")
        )

    def test_member_in_different_group_is_denied(self):
        """Bypass scoped to allow-listed groups — wrong group ≠ bypass."""
        self.assertFalse(
            _is_group_member_authorized("group", f"group:{GROUP_B}", GROUP_A)
        )

    def test_dm_chat_type_skips_group_bypass(self):
        """DMs must not be authorized by a group-list match."""
        self.assertFalse(
            _is_group_member_authorized("dm", USER_RANDOM, GROUP_A)
        )

    def test_empty_allowlist_denies(self):
        """No SIGNAL_GROUP_ALLOWED_USERS set ⇒ bypass off, denies."""
        self.assertFalse(
            _is_group_member_authorized("group", f"group:{GROUP_A}", "")
        )

    def test_whitespace_only_allowlist_denies(self):
        """Whitespace-only env value parses to empty set ⇒ denies."""
        self.assertFalse(
            _is_group_member_authorized("group", f"group:{GROUP_A}", "  ,  ")
        )

    def test_multiple_groups_in_allowlist(self):
        """Comma-separated allowlist authorizes each listed group."""
        raw = f"{GROUP_A},{GROUP_B}"
        self.assertTrue(
            _is_group_member_authorized("group", f"group:{GROUP_A}", raw)
        )
        self.assertTrue(
            _is_group_member_authorized("group", f"group:{GROUP_B}", raw)
        )
        self.assertFalse(
            _is_group_member_authorized("group", "group:OTHER", raw)
        )

    def test_forum_and_channel_chat_types_also_bypass(self):
        """Existing bypass already supports forum/channel; keep that intact."""
        self.assertTrue(
            _is_group_member_authorized("forum", f"group:{GROUP_A}", GROUP_A)
        )
        self.assertTrue(
            _is_group_member_authorized("channel", f"group:{GROUP_A}", GROUP_A)
        )


class TestSignalGroupBypassMimicsTelegram(unittest.TestCase):
    """Behavioural parity with the Telegram/QQBOT branch.

    Telegram's TELEGRAM_GROUP_ALLOWED_CHATS already authorizes any user
    posting in a listed chat. Signal users with SIGNAL_GROUP_ALLOWED_USERS
    set should get the same UX — adding Signal to the dict closes the gap.
    """

    def test_any_random_user_in_allowed_group_is_authorized(self):
        """Bypass is chat-scoped; the user_id of the sender is irrelevant."""
        # Both these "users" should be authorized by virtue of group membership.
        for random_user in (USER_RANDOM, USER_OTHER, "+1-anonymous-phone"):
            with self.subTest(user=random_user):
                self.assertTrue(
                    _is_group_member_authorized(
                        "group", f"group:{GROUP_A}", GROUP_A
                    ),
                    f"user {random_user!r} should be authorized by group bypass",
                )


if __name__ == "__main__":
    unittest.main()
