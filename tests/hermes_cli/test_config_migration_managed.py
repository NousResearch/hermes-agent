"""Config migrations persist on a package-managed (NixOS / Home Manager) install.

Contract: managed mode refuses the user's own config writes, but a migration is Hermes' own
write. Boot bootstrap runs migrate_config after every package update, and a managed install
has no other way to migrate, so the migrated file and its version stamp must reach disk while
an ordinary save_config is still refused.
"""

import os
from unittest.mock import patch

import hermes_yaml as yaml


def _read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_migration_persists_under_managed_mode_but_user_saves_do_not(tmp_path):
    from hermes_cli.config import migrate_config, save_config
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    # A managed home is created by the package's activation, with these directories.
    for name in ("cron", "sessions", "logs", "memories", "plugins"):
        (tmp_path / name).mkdir()
    config_path = tmp_path / "config.yaml"
    # 46 -> 47 drops this exact value, the briefly shipped default.
    config_path.write_text(yaml.safe_dump(
        {"_config_version": 46, "compression": {"threshold_tokens": 256000}}), encoding="utf-8")
    (tmp_path / ".env").write_text("", encoding="utf-8")

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "HERMES_MANAGED": "nixos"}):
        migrate_config(interactive=False, quiet=True)
        migrated = _read(config_path)
        assert migrated["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert "threshold_tokens" not in (migrated.get("compression") or {})

        save_config({**migrated, "model": {"default": "changed-by-user"}})
        assert _read(config_path) == migrated, "managed mode still refuses the user's saves"
