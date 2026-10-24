"""Ordering guarantees for the setup wizard's config.yaml backup (#3522).

The wizard copies ``config.yaml`` into ``backups/config/`` so a user
can recover values setup overwrote. ``hermes setup --reset`` replaces that same
file with ``DEFAULT_CONFIG``, so the copy is only useful if it is taken *before*
the reset runs — and the user has to be told where it landed even when --reset
exits the wizard early.
"""

from argparse import Namespace

from hermes_cli.config import (
    DEFAULT_CONFIG,
    get_config_path,
    load_config,
    save_config,
)

SENTINEL_MODEL = "hand-tuned-local-model-xyz"


def _make_setup_args(**overrides):
    return Namespace(
        non_interactive=overrides.get("non_interactive", True),
        section=overrides.get("section", None),
        reset=overrides.get("reset", False),
    )


def _write_user_config(tmp_path):
    """Persist a config.yaml holding a distinctive, non-default user value."""
    config_path = get_config_path()
    cfg = load_config()
    cfg["model"] = {
        "provider": "custom",
        "base_url": "http://localhost:8080/v1",
        "default": SENTINEL_MODEL,
    }
    cfg["agent"]["max_turns"] = 47
    save_config(cfg)

    assert config_path.parent == tmp_path
    text = config_path.read_text(encoding="utf-8")
    assert SENTINEL_MODEL in text, "fixture failed to persist the user value"
    return config_path


def _backups(tmp_path):
    return sorted((tmp_path / "backups" / "config").glob("config.yaml.pre-setup.*"))


class TestResetBackupOrdering:
    def test_reset_backs_up_user_config_not_the_reset_defaults(
        self, tmp_path, monkeypatch
    ):
        """The --reset backup must hold the user's config, not DEFAULT_CONFIG.

        Regression for the ordering bug: the backup block used to run after the
        --reset branch had already overwritten config.yaml, so the ``.bak`` file
        advertised as the recovery path captured the defaults that had just been
        written and the user's real config was unrecoverable.
        """
        from hermes_cli.setup import run_setup_wizard

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = _write_user_config(tmp_path)

        run_setup_wizard(_make_setup_args(non_interactive=True, reset=True))

        backups = _backups(tmp_path)
        assert len(backups) == 1, f"expected exactly one backup, got {backups}"
        backup_text = backups[0].read_text(encoding="utf-8")
        assert SENTINEL_MODEL in backup_text, (
            "backup captured the post-reset defaults instead of the user's config"
        )
        assert "47" in backup_text

        # The reset itself must still take effect on config.yaml.
        assert SENTINEL_MODEL not in config_path.read_text(encoding="utf-8")
        assert load_config()["model"] == DEFAULT_CONFIG["model"]

    def test_reset_reports_where_the_backup_landed(
        self, tmp_path, monkeypatch, capsys
    ):
        """--reset can exit early, so it must surface the backup path itself."""
        from hermes_cli.setup import run_setup_wizard

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_user_config(tmp_path)

        run_setup_wizard(_make_setup_args(non_interactive=True, reset=True))

        out = capsys.readouterr().out
        backups = _backups(tmp_path)
        assert len(backups) == 1
        assert "Configuration reset to defaults." in out
        assert "Previous config backed up to:" in out
        assert backups[0].name in out
