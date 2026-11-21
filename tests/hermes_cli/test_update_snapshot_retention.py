"""Quick snapshot retention is configurable without changing the safe default."""
import pytest

from hermes_cli import update_cmd_maint as maint


@pytest.mark.parametrize("value,expected", [(2, 2), ("2", 2), ("invalid", 1), (-1, 1)])
def test_quick_snapshot_retention(tmp_path, monkeypatch, value, expected):
    calls = []
    monkeypatch.setattr(maint, "_load_updates_cfg", lambda: {"snapshot_keep": value})
    monkeypatch.setattr("hermes_cli.backup.create_quick_snapshot", lambda **kw: calls.append(kw))
    monkeypatch.setattr("hermes_cli.backup.create_pre_update_snapshots_all_profiles", lambda **kw: calls.append(kw))
    maint._run_quick_snapshots()
    assert [row["keep"] for row in calls] == [expected, expected]
    assert [row["max_file_size"] for row in calls] == [1 << 30, 1 << 30]


@pytest.mark.parametrize("config_text,expected", [
    ("updates: {}\n", 1),
    ("updates:\n  snapshot_keep: 3\n", 3),
    ("updates:\n  backup_keep: 7\n", 1),
])
def test_real_merged_config_retention(tmp_path, monkeypatch, config_text, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(config_text)
    calls = []
    monkeypatch.setattr("hermes_cli.backup.create_quick_snapshot", lambda **kw: calls.append(kw))
    monkeypatch.setattr("hermes_cli.backup.create_pre_update_snapshots_all_profiles", lambda **kw: calls.append(kw))
    maint._run_quick_snapshots()
    assert [row["keep"] for row in calls] == [expected, expected]
