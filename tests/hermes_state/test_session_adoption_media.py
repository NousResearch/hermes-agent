"""Cross-profile adoption must preserve image bytes before retiring its donor."""

import base64
import hashlib
from types import SimpleNamespace

import pytest

from agent.session_persistence import _db_flush_row
from hermes_state import SessionDB

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII="
)
CONTENT = [
    {"type": "text", "text": "Read this"},
    {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")}},
]
SESSION_ID = "stranded-vision"


@pytest.fixture
def stores(tmp_path, monkeypatch):
    donor_home = tmp_path / "donor"
    target_home = tmp_path / "target"
    donor_home.mkdir()
    target_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(donor_home))
    with SessionDB(db_path=donor_home / "state.db") as donor:
        donor.create_session(SESSION_ID, "tui")
        row = _db_flush_row(SimpleNamespace(_session_db=donor),
                            {"role": "user", "content": CONTENT}, False)
        donor.append_messages_batch(SESSION_ID, [row])
        monkeypatch.setenv("HERMES_HOME", str(target_home))
        with SessionDB(db_path=target_home / "state.db") as target:
            yield donor, target


def test_adoption_preserves_image_bytes_before_retiring_donor(stores):
    donor, target = stores

    result = target.adopt_session_lineage_from(donor, SESSION_ID)

    assert result["ok"] and result["adopted"] and result["donor_retired"]
    replay = target.get_messages_as_conversation(SESSION_ID, repair_alternation=True)
    assert replay[0]["content"] == CONTENT
    digest = hashlib.sha256(PNG).hexdigest()
    assert (target.db_path.parent / "cache/transcript_media" / digest).read_bytes() == PNG
    assert donor.get_session(SESSION_ID)["archived"]
    assert donor.get_session(SESSION_ID)["end_reason"] == "adopted_by_profile"


@pytest.mark.parametrize("failure", ["missing", "corrupt", "destination-unwritable"])
def test_failed_media_adoption_leaves_donor_recoverable(stores, caplog, failure):
    donor, target = stores
    original_session = donor.get_session(SESSION_ID)
    original_messages = donor.get_messages(SESSION_ID, include_inactive=True)
    path = donor.db_path.parent / "cache/transcript_media" / hashlib.sha256(PNG).hexdigest()
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_bytes(b"corrupt image")
    else:
        # A real filesystem failure, independent of permissions/root on the test host.
        (target.db_path.parent / "cache/transcript_media" / path.name).mkdir(parents=True)

    result = target.adopt_session_lineage_from(donor, SESSION_ID)

    assert result["ok"] is False
    assert result["adopted"] is False
    assert result["donor_retired"] is False
    assert "transcript media" in result["error"]
    assert "Cannot adopt transcript media" in caplog.text
    assert target.get_session(SESSION_ID) is None
    assert donor.get_session(SESSION_ID) == original_session
    assert donor.get_messages(SESSION_ID, include_inactive=True) == original_messages
    assert not original_session["archived"]
    assert original_session["end_reason"] is None

    # Fixing the unavailable bytes/store permits retry: no flattened target id was left behind.
    if failure == "destination-unwritable":
        (target.db_path.parent / "cache/transcript_media" / path.name).rmdir()
    else:
        path.write_bytes(PNG)
    assert donor.get_messages_as_conversation(SESSION_ID, repair_alternation=True)[0]["content"] == CONTENT
    assert target.adopt_session_lineage_from(donor, SESSION_ID)["donor_retired"]
    assert target.get_messages_as_conversation(SESSION_ID, repair_alternation=True)[0]["content"] == CONTENT
