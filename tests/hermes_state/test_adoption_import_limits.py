"""Atomic adoption composition retains the public import batch bound."""

import pytest

from hermes_state import SessionDB
from tests.hermes_state.test_adoption_transfer_records import _open, _seed_compression_lineage


def test_adoption_obeys_the_existing_import_session_limit(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    donor, target = _open(home, default=True), _open(home)
    try:
        _seed_compression_lineage(donor, "bounded-root", "bounded-tip")
        monkeypatch.setattr(SessionDB, "_IMPORT_MAX_SESSIONS", 1)
        with pytest.raises(ValueError, match="at most 1"):
            target.adopt_session_lineage_from(donor, "bounded-tip")
        assert target.get_session("bounded-root") is None
        assert target.get_session("bounded-tip") is None
        assert target._read_one("SELECT 1 FROM sqlite_master WHERE name='adoption_transfers'") is None
        assert donor.get_session("bounded-root")["end_reason"] == "compression"
        assert donor.get_session("bounded-tip")["end_reason"] is None
    finally:
        donor.close()
        target.close()
