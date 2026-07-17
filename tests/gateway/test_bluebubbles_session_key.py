"""BlueBubbles DM session-key stability across chatGuid variants.

BlueBubbles delivers the same 1:1 conversation under several chat_id forms:
the service-prefixed GUID (``iMessage;-;+1555…``, ``any;-;+1555…``) when the
webhook carries one, and the bare handle (``+1555…``) when it does not (the
``chat_identifier`` fallback in the adapter).  Keying sessions on the raw
chat_id splits one conversation across several SessionEntries; a message
landing on a long-untouched variant then trips the idle reset and wipes an
actively-used conversation.
"""

import pytest

from gateway.config import Platform
from gateway.session import (
    SessionSource,
    build_session_key,
    canonical_bluebubbles_identifier,
)


HANDLE = "+15551234567"


class TestCanonicalBlueBubblesIdentifier:
    @pytest.mark.parametrize(
        "raw",
        [
            f"iMessage;-;{HANDLE}",
            f"any;-;{HANDLE}",
            f"SMS;-;{HANDLE}",
            HANDLE,
        ],
    )
    def test_dm_guid_variants_collapse_to_bare_handle(self, raw):
        assert canonical_bluebubbles_identifier(raw) == HANDLE

    def test_email_handle_is_preserved(self):
        assert (
            canonical_bluebubbles_identifier("iMessage;-;user@example.com")
            == "user@example.com"
        )

    def test_group_guid_is_left_alone(self):
        """Group GUIDs use ``;+;`` and are opaque — never rewrite them."""
        guid = "iMessage;+;chat9876543210"
        assert canonical_bluebubbles_identifier(guid) == guid

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_values_pass_through(self, value):
        assert canonical_bluebubbles_identifier(value) == value


class TestBlueBubblesDMSessionKey:
    def _key(self, chat_id):
        return build_session_key(
            SessionSource(
                platform=Platform.BLUEBUBBLES,
                chat_id=chat_id,
                chat_type="dm",
                user_id=HANDLE,
            )
        )

    def test_all_dm_variants_share_one_session_key(self):
        """The regression: three forms of one conversation, one key."""
        keys = {
            self._key(f"iMessage;-;{HANDLE}"),
            self._key(f"any;-;{HANDLE}"),
            self._key(HANDLE),
        }
        assert keys == {f"agent:main:bluebubbles:dm:{HANDLE}"}

    def test_distinct_contacts_stay_isolated(self):
        assert self._key(f"any;-;{HANDLE}") != self._key("any;-;+15559999999")

    def test_group_chats_keep_their_raw_guid(self):
        """Groups must be untouched: their GUID is not a participant handle."""
        source = SessionSource(
            platform=Platform.BLUEBUBBLES,
            chat_id="iMessage;+;chat9876543210",
            chat_type="group",
            user_id=HANDLE,
        )
        key = build_session_key(source)
        assert "iMessage;+;chat9876543210" in key

    def test_threaded_dm_variants_share_one_key(self):
        def keyed(chat_id):
            return build_session_key(
                SessionSource(
                    platform=Platform.BLUEBUBBLES,
                    chat_id=chat_id,
                    chat_type="dm",
                    user_id=HANDLE,
                    thread_id="t1",
                )
            )

        assert keyed(f"any;-;{HANDLE}") == keyed(HANDLE)
        assert keyed(HANDLE) == f"agent:main:bluebubbles:dm:{HANDLE}:t1"

    def test_other_platforms_unaffected(self):
        """Only BlueBubbles is canonicalized; a lookalike id elsewhere is raw."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=f"iMessage;-;{HANDLE}",
            chat_type="dm",
            user_id="u1",
        )
        assert build_session_key(source) == (
            f"agent:main:telegram:dm:iMessage;-;{HANDLE}"
        )
