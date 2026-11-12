"""Transfer membership survives reopen and cannot commit apart from imported rows."""

import pytest

from hermes_state import SessionDB
from tests.hermes_state.test_adoption_transfer_records import _open, _retire_fails_for, _seed_compression_lineage


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def test_record_write_failure_rolls_back_target_rows_and_membership(home, monkeypatch):
    import hermes_state_adoption_records as records
    donor, target = _open(home, default=True), _open(home)
    try:
        _seed_compression_lineage(donor, "atomic-root", "atomic-tip")
        original = records.insert

        def failed_insert(conn, manifest, **options):
            original(conn, manifest, **options)
            raise ValueError("injected record failure")

        monkeypatch.setattr(records, "insert", failed_insert)
        with pytest.raises(ValueError, match="injected record failure"):
            target.adopt_session_lineage_from(donor, "atomic-tip")
        assert target.get_session("atomic-root") is None
        assert target.get_session("atomic-tip") is None
        assert target._read_one("SELECT 1 FROM sqlite_master WHERE name='adoption_transfers'") is None
        assert donor.get_session("atomic-root")["end_reason"] == "compression"
        assert donor.get_session("atomic-tip")["end_reason"] is None
        assert len(donor.get_messages("atomic-root", include_inactive=True)) == 2
    finally:
        donor.close()
        target.close()


def test_full_membership_survives_database_reopen(home, monkeypatch):
    donor, target = _open(home, default=True), _open(home)
    try:
        _seed_compression_lineage(donor, "reopen-root", "reopen-tip")
        with monkeypatch.context() as patch:
            patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for("reopen-tip"))
            first = target.adopt_session_lineage_from(donor, "reopen-tip")
        assert first["adopted"] and not first["donor_retired"]
    finally:
        donor.close()
        target.close()
    donor, target = _open(home, default=True), _open(home)
    try:
        completed = target.adopt_session_lineage_from(donor, "reopen-root")
        assert completed["donor_retired"]
        assert completed["member_ids"] == ["reopen-root", "reopen-tip"]
        assert len(target.get_messages("reopen-root", include_inactive=True)) == 2
        assert len(target.get_messages("reopen-tip", include_inactive=True)) == 2
    finally:
        donor.close()
        target.close()


def test_lost_record_row_cannot_shrink_the_completion_denominator(home):
    donor, target = _open(home, default=True), _open(home)
    try:
        _seed_compression_lineage(donor, "missing-row-root", "missing-row-tip")
        assert target.adopt_session_lineage_from(donor, "missing-row-tip", retire_donor=False)["adopted"]
        target._conn.execute("DELETE FROM adoption_transfers WHERE member_id='missing-row-tip'")
        target._conn.commit()
        refused = target.adopt_session_lineage_from(donor, "missing-row-root")
        assert refused.get("conflict") is True
        assert refused["donor_retired"] is False
        assert donor.get_session("missing-row-tip")["end_reason"] is None
    finally:
        donor.close()
        target.close()
