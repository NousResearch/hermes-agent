"""Gateway (Telegram path) write-approval setter must preserve the 3-state
mode — especially 'background_only' must NOT be coerced to bool True, which
would gate foreground writes too. Regression for the gateway/CLI split bug."""

import yaml


def test_persist_preserves_background_only(tmp_path):
    from gateway.slash_commands import _persist_write_approval
    cfg = tmp_path / "config.yaml"
    stored = _persist_write_approval(cfg, "skills", "background_only")
    assert stored == "background_only"
    data = yaml.safe_load(cfg.read_text())
    assert data["skills"]["write_approval"] == "background_only"


def test_persist_memory_background_only(tmp_path):
    from gateway.slash_commands import _persist_write_approval
    cfg = tmp_path / "config.yaml"
    _persist_write_approval(cfg, "memory", "background_only")
    data = yaml.safe_load(cfg.read_text())
    assert data["memory"]["write_approval"] == "background_only"


def test_persist_bool_values(tmp_path):
    from gateway.slash_commands import _persist_write_approval
    cfg = tmp_path / "config.yaml"
    assert _persist_write_approval(cfg, "memory", True) is True
    assert _persist_write_approval(cfg, "memory", False) is False
    data = yaml.safe_load(cfg.read_text())
    assert data["memory"]["write_approval"] is False
