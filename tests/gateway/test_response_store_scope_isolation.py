"""Regression guard for cross-tenant RESPONSE-STORE leak (leak #3, FIX-002).

The OpenAI Responses API maps a caller-supplied ``conversation`` NAME to a
response_id, then to that response's full ``conversation_history`` (transcripts
+ tool payloads). On the shared multi-tenant Avocado deploy the name is
user-chosen and collidable: two tenants both sending ``conversation="default"``
would resolve to each other's latest response. ``ResponseStore`` has no user
scoping, so the fix namespaces the conversation NAME by tenant scope (derived
from ``gateway_session_key = "avocado:<user_id>"``) before every
get_conversation / set_conversation call.

These tests prove:
  * the scope-derivation + name-namespacing helpers behave per-tenant,
  * two tenants using the SAME conversation name no longer collide in the store,
  * no avocado scope keys the name exactly as before (single-tenant unchanged).
"""
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, ResponseStore


class TestAvocadoScopeFromKey:
    def test_avocado_prefix_extracts_scope(self):
        assert APIServerAdapter._avocado_scope_from_key("avocado:user_42") == "user_42"

    def test_non_avocado_key_is_none(self):
        assert APIServerAdapter._avocado_scope_from_key("agent:main:telegram:dm:1") is None
        assert APIServerAdapter._avocado_scope_from_key("") is None
        assert APIServerAdapter._avocado_scope_from_key(None) is None

    def test_empty_avocado_scope_is_none(self):
        assert APIServerAdapter._avocado_scope_from_key("avocado:") is None


class TestScopeConversationName:
    def test_two_tenants_same_name_get_distinct_keys(self):
        a = APIServerAdapter._scope_conversation_name("default", "avocado:alice")
        b = APIServerAdapter._scope_conversation_name("default", "avocado:bob")
        assert a != b
        assert a.endswith("\x00default")
        assert b.endswith("\x00default")
        assert "alice" in a and "bob" in b

    def test_no_scope_passes_name_through_unchanged(self):
        # Single-tenant / explicit X-Hermes-Session-Key path: identical to before.
        assert APIServerAdapter._scope_conversation_name("default", None) == "default"
        assert (
            APIServerAdapter._scope_conversation_name("default", "agent:main")
            == "default"
        )

    def test_none_conversation_stays_none(self):
        assert APIServerAdapter._scope_conversation_name(None, "avocado:alice") is None
        assert APIServerAdapter._scope_conversation_name("", "avocado:alice") == ""

    def test_traversal_in_scope_is_sanitized(self):
        scoped = APIServerAdapter._scope_conversation_name("c", "avocado:../../etc")
        assert "/" not in scoped
        assert ".." not in scoped


class TestResponseStoreNamespacedIsolation:
    """End-to-end on the store itself: with namespaced names two tenants can't
    read each other's conversation_history; with a raw shared name they could
    (the pre-fix bug), which this test documents as the contrast."""

    def test_namespaced_names_isolate_history(self, tmp_path):
        store = ResponseStore(max_size=50, db_path=str(tmp_path / "rs.db"))

        alice_name = APIServerAdapter._scope_conversation_name("default", "avocado:alice")
        bob_name = APIServerAdapter._scope_conversation_name("default", "avocado:bob")

        store.put("resp_alice", {"conversation_history": [{"role": "user", "content": "alice secret"}]})
        store.set_conversation(alice_name, "resp_alice")

        # Bob asks for HIS "default" — must not resolve to Alice's response.
        assert store.get_conversation(bob_name) is None
        # Alice still resolves her own.
        assert store.get_conversation(alice_name) == "resp_alice"

        store.put("resp_bob", {"conversation_history": [{"role": "user", "content": "bob secret"}]})
        store.set_conversation(bob_name, "resp_bob")

        assert store.get_conversation(bob_name) == "resp_bob"
        assert store.get_conversation(alice_name) == "resp_alice"

    def test_raw_shared_name_would_collide(self, tmp_path):
        """Contrast: without namespacing, the same raw name collides (the bug)."""
        store = ResponseStore(max_size=50, db_path=str(tmp_path / "rs2.db"))
        store.set_conversation("default", "resp_alice")
        store.set_conversation("default", "resp_bob")  # overwrites — collision
        assert store.get_conversation("default") == "resp_bob"
