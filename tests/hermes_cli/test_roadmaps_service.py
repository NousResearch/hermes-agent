from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import projects_db
from hermes_cli.roadmaps_service import RoadmapsService, RoadmapsUnavailable


def seed(path: Path) -> None:
    conn = projects_db.connect(path)
    conn.execute("INSERT INTO projects(id, slug, name, created_at) VALUES ('p1', 'one', 'One', 1)")
    conn.execute("INSERT INTO projects(id, slug, name, created_at) VALUES ('p2', 'two', 'Two', 1)")
    conn.execute("INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ('prof', 'p1', 'r1', 'Roadmap', 'Purpose', 'in_progress', 1, 'a', 'b', 1, 2))
    conn.execute("INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ('prof', 'p1', 'r1', 1, 'validated', 'src', 'why', 'a', 1, 'hash'))
    conn.execute("INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ('prof', 'p1', 'r1', 1, 'n1', None, 'objective', 'Node', None, 'ready', 50, 'agent', None, 1, 2))
    conn.execute("INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ('prof', 'p1', 'r1', 1, 'n2', None, 'step', 'Step', None, 'planned', 0, None, None, 1, 2))
    conn.execute("INSERT INTO roadmap_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ('prof', 'p1', 'r1', 1, 'rel', 'n1', 'n2', 'blocks', 'active', None))
    conn.execute("INSERT INTO roadmap_todos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ('prof', 'p1', 'r1', 1, 't1', 'n1', 'Todo', 'open', 0, 1, 2))
    conn.commit()
    conn.close()


def test_service_list_filters_explicit_profile_and_project(tmp_path: Path):
    path = tmp_path / 'projects.db'
    seed(path)
    service = RoadmapsService(path)
    assert [r['roadmap_id'] for r in service.list(profile_id='prof')['roadmaps']] == ['r1']
    assert service.list(profile_id='prof', project_id='p2') == {'roadmaps': [], 'scope': {'profile_id': 'prof', 'project_id': 'p2'}}
    assert service.list(profile_id='other')['roadmaps'] == []


def test_service_does_not_serialize_additional_columns(tmp_path: Path):
    path = tmp_path / 'projects.db'
    seed(path)
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE roadmaps ADD COLUMN internal_secret TEXT")
    conn.execute("UPDATE roadmaps SET internal_secret = 'must not leak'")
    conn.commit()
    conn.close()

    service = RoadmapsService(path)
    listing = service.list('prof')['roadmaps'][0]
    roadmap = service.get('prof', 'p1', 'r1')['roadmap']
    assert 'internal_secret' not in listing
    assert 'internal_secret' not in roadmap


def test_service_get_and_snapshot_are_json_safe_and_deterministic(tmp_path: Path):
    path = tmp_path / 'projects.db'
    seed(path)
    service = RoadmapsService(path)
    result = service.get('prof', 'p1', 'r1')
    assert result['found'] is True
    assert result['roadmap']['active_version'] == 1
    assert result['roadmap']['versions'][0]['nodes'][0]['node_id'] == 'n1'
    assert result['roadmap']['versions'][0]['relations'][0]['relation_id'] == 'rel'
    assert result['roadmap']['versions'][0]['todos'][0]['todo_id'] == 't1'
    assert result == service.snapshot('prof', 'p1', 'r1')
    json.dumps(result, sort_keys=True)


def test_service_unknown_scope_is_not_found_and_never_falls_back(tmp_path: Path):
    path = tmp_path / 'projects.db'
    seed(path)
    result = RoadmapsService(path).get('prof', 'missing', 'r1')
    assert result == {'found': False, 'scope': {'profile_id': 'prof', 'project_id': 'missing', 'roadmap_id': 'r1'}, 'roadmap': None}


def test_service_is_read_only_and_does_not_fake_events(tmp_path: Path):
    path = tmp_path / 'projects.db'
    seed(path)
    before = path.read_bytes()
    service = RoadmapsService(path)
    assert not hasattr(service, 'events')
    assert 'events' not in {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    service.list('prof')
    assert path.read_bytes() == before


def test_service_missing_db_is_empty_without_creating_path(tmp_path: Path):
    path = tmp_path / "missing" / "projects.db"
    result = RoadmapsService(path).list("prof")
    assert result == {"roadmaps": [], "scope": {"profile_id": "prof"}}
    assert not path.exists()
    assert not path.parent.exists()


def test_service_legacy_db_is_rejected_without_mutation(tmp_path: Path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE legacy (value TEXT)")
    conn.commit(); conn.close()
    before = path.read_bytes()
    from hermes_cli.roadmaps_service import RoadmapsUnavailable
    with pytest.raises(RoadmapsUnavailable):
        RoadmapsService(path).list("prof")
    assert path.read_bytes() == before


def test_service_closes_connection_when_metadata_inspection_fails(tmp_path: Path, monkeypatch):
    path = tmp_path / 'projects.db'
    path.touch()

    class BrokenConnection:
        row_factory = None
        closed = False

        def execute(self, *args):
            raise sqlite3.DatabaseError('metadata unavailable')

        def close(self):
            self.closed = True

    connection = BrokenConnection()
    monkeypatch.setattr(sqlite3, 'connect', lambda *args, **kwargs: connection)
    with pytest.raises(RoadmapsUnavailable):
        RoadmapsService(path).list('prof')
    assert connection.closed is True


@pytest.mark.parametrize('value', ['x' * 129])
def test_service_rejects_overlong_identifiers(tmp_path: Path, value: str):
    path = tmp_path / 'projects.db'
    seed(path)
    service = RoadmapsService(path)
    with pytest.raises(ValueError, match='at most 128'):
        service.list(value)
    with pytest.raises(ValueError, match='at most 128'):
        service.get('prof', value, 'r1')
    with pytest.raises(ValueError, match='at most 128'):
        service.get('prof', 'p1', value)


def test_list_sessions_is_scoped_allowlisted_and_read_only(tmp_path: Path):
    path = tmp_path / "projects.db"
    seed(path)
    conn = projects_db.connect(path)
    conn.execute(
        "INSERT INTO roadmap_sessions "
        "(profile_id, project_id, roadmap_id, stored_session_id, kind, state, actor, created_at, updated_at) "
        "VALUES ('prof', 'p1', 'r1', 'stored-vision', 'vision', 'active', 'pierre', 3, 3)"
    )
    conn.execute("ALTER TABLE roadmap_sessions ADD COLUMN internal_runtime_id TEXT")
    conn.execute("UPDATE roadmap_sessions SET internal_runtime_id='runtime-must-not-leak'")
    conn.commit()
    conn.close()
    before = path.read_bytes()

    result = RoadmapsService(path).list_sessions("prof", "p1", "r1")

    assert result["scope"] == {
        "profile_id": "prof", "project_id": "p1", "roadmap_id": "r1",
    }
    assert result["sessions"][0]["stored_session_id"] == "stored-vision"
    assert "internal_runtime_id" not in result["sessions"][0]
    assert "runtime_session_id" not in json.dumps(result)
    assert RoadmapsService(path).list_sessions("prof", "p1", "missing")["sessions"] == []
    assert path.read_bytes() == before


def test_list_sessions_missing_db_is_empty_without_creating_path(tmp_path: Path):
    path = tmp_path / "missing" / "projects.db"
    result = RoadmapsService(path).list_sessions("prof", "p1", "r1")
    assert result == {
        "sessions": [],
        "scope": {"profile_id": "prof", "project_id": "p1", "roadmap_id": "r1"},
    }
    assert not path.exists()
