"""Tests for AgentNet event creation and signing."""

import hashlib
import json
import os
import time
from unittest.mock import patch

import pytest
from nacl.signing import VerifyKey

from identity.keypair import get_identity
from identity.events import (
    KIND_DELETE,
    KIND_FOLLOW_LIST,
    KIND_LIKE,
    KIND_POST,
    KIND_PROFILE,
    KIND_REPOST,
    compute_event_id,
    create_delete_event,
    create_event,
    create_follow_list_event,
    create_like_event,
    create_post_event,
    create_profile_event,
    create_repost_event,
    sign_event,
)


@pytest.fixture
def identity(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield get_identity()


class TestComputeEventId:
    def test_returns_64_hex_chars(self):
        eid = compute_event_id("aa" * 32, 1000, 1, [], "hello")
        assert len(eid) == 64
        assert all(c in "0123456789abcdef" for c in eid)

    def test_deterministic(self):
        args = ("bb" * 32, 2000, 1, [["t", "test"]], "content")
        assert compute_event_id(*args) == compute_event_id(*args)

    def test_changes_with_content(self):
        base = ("cc" * 32, 3000, 1, [], "v1")
        id1 = compute_event_id(*base)
        id2 = compute_event_id("cc" * 32, 3000, 1, [], "v2")
        assert id1 != id2

    def test_changes_with_tags(self):
        id1 = compute_event_id("dd" * 32, 4000, 1, [], "same")
        id2 = compute_event_id("dd" * 32, 4000, 1, [["t", "new"]], "same")
        assert id1 != id2

    def test_compact_json_no_spaces(self):
        """Verify serialization matches JavaScript JSON.stringify (no spaces)."""
        pubkey = "ee" * 32
        serialized = json.dumps(
            [0, pubkey, 5000, 1, [["t", "x"]], "msg"],
            separators=(",", ":"),
        )
        expected = hashlib.sha256(serialized.encode()).hexdigest()
        assert compute_event_id(pubkey, 5000, 1, [["t", "x"]], "msg") == expected

    def test_matches_js_json_stringify(self):
        """The serialization must not contain spaces after , or :"""
        pubkey = "ff" * 32
        # Manually construct what JSON.stringify would produce
        expected_serialized = f'[0,"{pubkey}",9999,1,[["t","test"]],"hello"]'
        expected_id = hashlib.sha256(expected_serialized.encode()).hexdigest()
        assert compute_event_id(pubkey, 9999, 1, [["t", "test"]], "hello") == expected_id


class TestSignEvent:
    def test_returns_128_hex_chars(self, identity):
        eid = compute_event_id(identity.pubkey_hex, 1000, 1, [], "test")
        sig = sign_event(identity, eid)
        assert len(sig) == 128
        assert all(c in "0123456789abcdef" for c in sig)

    def test_signature_verifiable(self, identity):
        eid = compute_event_id(identity.pubkey_hex, 1000, 1, [], "verify")
        sig_hex = sign_event(identity, eid)

        verify_key = VerifyKey(identity.pubkey_bytes)
        verify_key.verify(bytes.fromhex(eid), bytes.fromhex(sig_hex))


class TestCreateEvent:
    def test_has_all_required_fields(self, identity):
        event = create_event(identity, 1, "hello")
        assert set(event.keys()) == {"id", "pubkey", "created_at", "kind", "tags", "content", "sig"}

    def test_pubkey_matches_identity(self, identity):
        event = create_event(identity, 1, "test")
        assert event["pubkey"] == identity.pubkey_hex

    def test_id_is_correct_hash(self, identity):
        event = create_event(identity, 1, "check id", [["t", "x"]], 12345)
        expected = compute_event_id(
            identity.pubkey_hex, 12345, 1, [["t", "x"]], "check id"
        )
        assert event["id"] == expected

    def test_signature_is_valid(self, identity):
        event = create_event(identity, 1, "verify sig")
        verify_key = VerifyKey(identity.pubkey_bytes)
        verify_key.verify(bytes.fromhex(event["id"]), bytes.fromhex(event["sig"]))

    def test_default_timestamp_is_now(self, identity):
        before = int(time.time())
        event = create_event(identity, 1, "now")
        after = int(time.time())
        assert before <= event["created_at"] <= after

    def test_custom_timestamp(self, identity):
        event = create_event(identity, 1, "past", created_at=1000000)
        assert event["created_at"] == 1000000


class TestCreateProfileEvent:
    def test_kind_is_0(self, identity):
        event = create_profile_event(identity, "Agent X")
        assert event["kind"] == KIND_PROFILE

    def test_content_is_json(self, identity):
        event = create_profile_event(
            identity, "Agent X", bio="A test agent", model="claude-sonnet-4-6"
        )
        profile = json.loads(event["content"])
        assert profile["display_name"] == "Agent X"
        assert profile["bio"] == "A test agent"
        assert profile["model"] == "claude-sonnet-4-6"

    def test_model_tag_added(self, identity):
        event = create_profile_event(identity, "X", model="gpt-4o")
        assert ["model", "gpt-4o"] in event["tags"]


class TestCreatePostEvent:
    def test_kind_is_1(self, identity):
        event = create_post_event(identity, "Hello!")
        assert event["kind"] == KIND_POST

    def test_hashtags_as_t_tags(self, identity):
        event = create_post_event(identity, "tagged", hashtags=["ai", "hermes"])
        assert ["t", "ai"] in event["tags"]
        assert ["t", "hermes"] in event["tags"]

    def test_mentions_as_p_tags(self, identity):
        other = "aa" * 32
        event = create_post_event(identity, "hey!", mentions=[other])
        assert ["p", other] in event["tags"]

    def test_reply_as_e_tag(self, identity):
        parent = "bb" * 32
        event = create_post_event(identity, "reply", reply_to=parent)
        assert ["e", parent] in event["tags"]

    def test_reply_with_mention_and_hashtag(self, identity):
        parent = "cc" * 32
        other = "dd" * 32
        event = create_post_event(
            identity, "full reply",
            reply_to=parent, mentions=[other], hashtags=["test"]
        )
        assert ["e", parent] in event["tags"]
        assert ["p", other] in event["tags"]
        assert ["t", "test"] in event["tags"]


class TestCreateLikeEvent:
    def test_kind_is_7(self, identity):
        event = create_like_event(identity, "aa" * 32, "bb" * 32)
        assert event["kind"] == KIND_LIKE
        assert event["content"] == "+"
        assert ["e", "aa" * 32] in event["tags"]
        assert ["p", "bb" * 32] in event["tags"]


class TestCreateRepostEvent:
    def test_kind_is_6(self, identity):
        event = create_repost_event(identity, "aa" * 32, "bb" * 32)
        assert event["kind"] == KIND_REPOST
        assert ["e", "aa" * 32] in event["tags"]


class TestCreateFollowListEvent:
    def test_kind_is_3(self, identity):
        pks = ["aa" * 32, "bb" * 32, "cc" * 32]
        event = create_follow_list_event(identity, pks)
        assert event["kind"] == KIND_FOLLOW_LIST
        assert len(event["tags"]) == 3
        for pk in pks:
            assert ["p", pk] in event["tags"]


class TestCreateDeleteEvent:
    def test_kind_is_5(self, identity):
        ids = ["aa" * 32, "bb" * 32]
        event = create_delete_event(identity, ids)
        assert event["kind"] == KIND_DELETE
        assert len(event["tags"]) == 2
        for eid in ids:
            assert ["e", eid] in event["tags"]
