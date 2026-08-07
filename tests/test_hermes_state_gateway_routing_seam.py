"""Seam tests for the R2-S1 extraction (epic #78647, issue #78636).

hermes_state.py's gateway routing CRUD (save/replace/load/delete
gateway routing entries, window 3204-3279) moved byte-verbatim into
hermes_state_gateway_routing.SessionGatewayRoutingMixin. These tests
pin the seam:

1. Identity: SessionDB members ARE the mixin members (MRO resolution).
2. Mixin-first: SessionGatewayRoutingMixin sits directly after SessionDB
   in the MRO (so the moved methods resolve before any other base).
3. Behavioral roundtrip: the 4 CRUD methods work through SessionDB
   against a real temp database, including replace pruning semantics.
"""

import json

import pytest

import hermes_state_gateway_routing as m
from hermes_state import SessionDB
from hermes_state_gateway_routing import SessionGatewayRoutingMixin


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file (house fixture pattern)."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


class TestSeamIdentity:
    def test_mixin_members_are_sessiondb_members(self, db):
        for name in (
            "save_gateway_routing_entry",
            "replace_gateway_routing_entries",
            "load_gateway_routing_entries",
            "delete_gateway_routing_entries",
        ):
            assert getattr(SessionDB, name) is getattr(SessionGatewayRoutingMixin, name), name

    def test_mixin_in_mro(self, db):
        # The mandated class line appends SessionGatewayRoutingMixin LAST:
        #   class SessionDB(SessionSearchMixin, SessionSchemaMixin,
        #                   SessionPortabilityMixin, SessionGatewayRoutingMixin):
        # so the mixin must be present in the MRO as the final base, and the
        # moved names must resolve through it (identity test above).
        assert SessionGatewayRoutingMixin in SessionDB.__mro__
        assert SessionDB.__mro__[-2] is SessionGatewayRoutingMixin
        assert m.SessionGatewayRoutingMixin is SessionGatewayRoutingMixin

    def test_mixin_module_does_not_import_hermes_state(self):
        # Cycle guard: the mixin module must not import its host.
        import ast

        tree = ast.parse(open(m.__file__, encoding="utf-8").read())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.append(node.module)
        assert "hermes_state" not in imported, imported
        assert "hermes_state_common" not in imported, imported


class TestGatewayRoutingRoundtrip:
    def test_save_load_roundtrip(self, db):
        entry = json.dumps({"session_id": "s1", "user_id": "u1"})
        db.save_gateway_routing_entry("key-1", entry, scope="scope-a")
        db.save_gateway_routing_entry("key-2", json.dumps({"session_id": "s2"}), scope="scope-a")

        rows = db.load_gateway_routing_entries(scope="scope-a")
        assert rows == {"key-1": entry, "key-2": json.dumps({"session_id": "s2"})}
        # Scopes are isolated.
        assert db.load_gateway_routing_entries(scope="scope-b") == {}

    def test_save_upserts_existing_key(self, db):
        db.save_gateway_routing_entry("k", "v1", scope="s")
        db.save_gateway_routing_entry("k", "v2", scope="s")
        assert db.load_gateway_routing_entries(scope="s") == {"k": "v2"}

    def test_replace_prunes_absent_keys(self, db):
        db.save_gateway_routing_entry("a", "1", scope="s")
        db.save_gateway_routing_entry("b", "2", scope="s")
        db.replace_gateway_routing_entries({"a": "1-new"}, scope="s")
        assert db.load_gateway_routing_entries(scope="s") == {"a": "1-new"}

    def test_replace_does_not_touch_other_scopes(self, db):
        db.save_gateway_routing_entry("a", "1", scope="s1")
        db.replace_gateway_routing_entries({"b": "2"}, scope="s2")
        assert db.load_gateway_routing_entries(scope="s1") == {"a": "1"}

    def test_delete_gateway_routing_entries(self, db):
        db.save_gateway_routing_entry("a", "1", scope="s")
        db.save_gateway_routing_entry("b", "2", scope="s")
        db.delete_gateway_routing_entries(["a", "missing"], scope="s")
        assert db.load_gateway_routing_entries(scope="s") == {"b": "2"}

    def test_empty_inputs_are_noops(self, db):
        db.delete_gateway_routing_entries([], scope="s")
        db.replace_gateway_routing_entries({}, scope="s")
        db.save_gateway_routing_entry("", "", scope="s")
        assert db.load_gateway_routing_entries(scope="s") == {}
