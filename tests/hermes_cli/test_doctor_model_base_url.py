"""Doctor flags a persisted ``model.base_url`` that is another provider's canonical endpoint (#135676).

A provider switch can leave the previous provider's URL behind while ``model.provider`` moves on:
every request then 404s on the mismatched route with an outage-looking retry message, and no
diagnostic surface said anything. Only subscription-routed (OAuth) providers are judged — key-based
providers keep every override (region endpoints like minimax → minimax-cn are deliberate, #6039)."""

from hermes_cli import doctor_config


def _write_config(tmp_path, body: str):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


def test_doctor_flags_cross_provider_model_base_url(tmp_path, capsys):
    cfg = _write_config(tmp_path,
                        "model:\n"
                        "  provider: openai-codex\n"
                        "  default: gpt-5.3-codex\n"
                        "  base_url: https://api.anthropic.com\n")
    issues: list = []
    doctor_config._validate_model_config(cfg, issues)
    out = capsys.readouterr().out
    assert "model.base_url 'https://api.anthropic.com' is another provider's canonical endpoint" in out
    assert any("hermes config unset model.base_url" in i for i in issues)


def test_doctor_accepts_key_provider_cross_region_base_url(tmp_path, capsys):
    cfg = _write_config(tmp_path,
                        "model:\n"
                        "  provider: minimax\n"
                        "  default: MiniMax-M2\n"
                        "  base_url: https://api.minimaxi.com/anthropic\n")
    issues: list = []
    doctor_config._validate_model_config(cfg, issues)
    out = capsys.readouterr().out
    assert "canonical endpoint" not in out
    assert not any("canonical endpoint" in i for i in issues)
