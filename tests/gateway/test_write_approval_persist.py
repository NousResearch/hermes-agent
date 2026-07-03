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


def test_gateway_wiring_skills_approval_background_only(tmp_path):
    """Handler-level: the gateway's parser→handle_pending_subcommand→set_mode_fn
    →_persist_write_approval path stores 'background_only' verbatim (not bool).

    Mirrors exactly how _handle_skills_command wires the closure, so a Telegram
    '/skills approval background_only' lands as the string in config.
    """
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from gateway.slash_commands import _persist_write_approval
    from tools import write_approval as wa
    import yaml
    cfg = tmp_path / "config.yaml"

    out = handle_pending_subcommand(
        wa.SKILLS, ["approval", "background_only"],
        set_mode_fn=lambda value: _persist_write_approval(cfg, "skills", value),
    )
    assert "background_only" in out
    data = yaml.safe_load(cfg.read_text())
    assert data["skills"]["write_approval"] == "background_only"


def test_gateway_wiring_memory_approval_on_stays_bool(tmp_path):
    """'on' must still persist as bool True (not the string 'on')."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from gateway.slash_commands import _persist_write_approval
    from tools import write_approval as wa
    import yaml
    cfg = tmp_path / "config.yaml"

    handle_pending_subcommand(
        wa.MEMORY, ["approval", "on"],
        memory_store=None,
        set_mode_fn=lambda value: _persist_write_approval(cfg, "memory", value),
    )
    data = yaml.safe_load(cfg.read_text())
    assert data["memory"]["write_approval"] is True
