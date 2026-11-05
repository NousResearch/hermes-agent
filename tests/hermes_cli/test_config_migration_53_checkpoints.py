"""Config v53: checkpoints default on, and the migration flips only the old default."""
import hermes_yaml as yaml

from hermes_cli.config import DEFAULT_CONFIG, migrate_config


def _migrate(home, checkpoints):
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump({"_config_version": 52, "checkpoints": checkpoints}))
    migrate_config(interactive=False, quiet=True)
    return yaml.safe_load(path.read_text())


def test_saved_old_default_becomes_new_default_and_a_choice_survives(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    migrated = _migrate(tmp_path, {"enabled": False, "max_snapshots": 50})
    assert migrated["checkpoints"] == {"enabled": DEFAULT_CONFIG["checkpoints"]["enabled"], "max_snapshots": 50}
    assert migrated["_config_version"] == DEFAULT_CONFIG["_config_version"]

    # Turned back off after the migration: a later migrate run never flips it again.
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**migrated, "checkpoints": {"enabled": False}}))
    migrate_config(interactive=False, quiet=True)
    assert yaml.safe_load(path.read_text())["checkpoints"] == {"enabled": False}
