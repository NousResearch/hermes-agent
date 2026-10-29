"""The post-update model/MoA safety net in a Remote Config deployment (#122509)."""
from pathlib import Path


def test_restore_config_model_settings_is_a_no_op_without_file_tooling(tmp_path: Path, monkeypatch):
    """The snapshot is a LOCAL copy, so the net must never write its values into a backend
    without a config file: that would PATCH stale local values into the plane."""
    from hermes_cli import backup, config_backend
    from hermes_cli.backup import create_quick_snapshot, restore_config_model_settings_if_rewritten

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg = hermes_home / "config.yaml"
    cfg.write_text("_config_version: 39\nmodel:\n  provider: custom\n  default: zamba\n", encoding="utf-8")
    snap_id = create_quick_snapshot(label="pre-update", hermes_home=hermes_home, keep=5)
    assert snap_id
    cfg.write_text("_config_version: 39\nmodel:\n  provider: deepseek\n", encoding="utf-8")
    writes = []
    monkeypatch.setattr(backup, "supports_file_tooling", lambda: False)
    monkeypatch.setattr(config_backend, "write_config_document", lambda *a, **k: writes.append(a))
    assert restore_config_model_settings_if_rewritten(snap_id, hermes_home=hermes_home) is None
    assert writes == []
    assert "deepseek" in cfg.read_text(encoding="utf-8")
