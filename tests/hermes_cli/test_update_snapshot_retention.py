"""Recovery retention configuration must apply equally to quick snapshots."""
import pytest

from hermes_cli import update_cmd_maint as maint


@pytest.mark.parametrize("value,expected", [(2, 2), ("2", 2), ("invalid", 1), (-1, 1)])
def test_quick_snapshot_retention(tmp_path, monkeypatch, value, expected):
    calls = []
    monkeypatch.setattr(maint, "_load_updates_cfg", lambda: {"backup_keep": value})
    monkeypatch.setattr("hermes_cli.backup.create_quick_snapshot", lambda **kw: calls.append(kw))
    monkeypatch.setattr("hermes_cli.backup.create_pre_update_snapshots_all_profiles", lambda **kw: calls.append(kw))
    maint._run_quick_snapshots()
    assert [row["keep"] for row in calls] == [expected, expected]
    assert [row["max_file_size"] for row in calls] == [1 << 30, 1 << 30]
