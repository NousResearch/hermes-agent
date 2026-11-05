"""A foreign import stores the origin's identifier as given, an explicit empty one included, and finds it again."""

import json

import pytest

from persistence.housekeeping.maintenance import ForeignOrigin, ForeignTurn


@pytest.mark.parametrize("identifier", [None, "", "foreign"])
def test_foreign_import_preserves_an_explicit_empty_identifier(backend, scope_a, identifier):
    maintenance = backend.open(scope_a).maintenance
    origin = ForeignOrigin(tool="tool", path="/foreign/history", foreign_session_id=identifier)
    expected = {"tool": "tool", "path": "/foreign/history", "foreign_session_id": identifier}
    imported = maintenance.import_foreign_history(origin, [ForeignTurn("user", "hello")], title="Imported", cwd=None,
                                                  profile="default")
    stored = maintenance.export_session(imported.session_id).session
    assert json.loads(stored.origin_json)["imported_from"] == expected
    assert maintenance.find_foreign_import(origin) == imported.session_id
    replay = maintenance.import_foreign_history(origin, [ForeignTurn("user", "different")], title="Replay", cwd=None,
                                                profile="default")
    assert replay.session_id == imported.session_id
    assert replay.already_imported is True
