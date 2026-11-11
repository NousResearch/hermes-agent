import pytest

from persistence import SessionArtifacts
from persistence.conversation.sessions import NewSession

from tests.contracts._scopes import MEDIA_ROOT


@pytest.mark.parametrize("sid", ["plain", "a[bc]", "a*", "a?"])
def test_delete_removes_only_the_addressed_session_artifacts(backend, scope_a, tmp_path, sid):
    store_a = backend.open(scope_a)
    directory = tmp_path / "artifacts"
    directory.mkdir()
    owned = [f"{sid}.json", f"{sid}.jsonl", f"session_{sid}.json", f"request_dump_{sid}_1.json"]
    neighbors = ["request_dump_ab_1.json", "request_dump_ac_1.json", f"{sid}.keep"]
    for name in owned + neighbors:
        (directory / name).write_text(name)
    store_a.sessions.create(NewSession(sid, "cli"))
    removed = store_a.sessions.delete(sid, media_root=MEDIA_ROOT)
    assert removed == (sid,)
    assert store_a.sessions.get(sid) is None
    SessionArtifacts(directory).remove_for(removed)
    assert sorted(p.name for p in directory.iterdir()) == sorted(neighbors)
