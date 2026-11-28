"""``hermes doctor`` reports an inference provider exactly when a chat turn would find one."""
from hermes_cli import doctor as doctor_mod
from hermes_cli import doctor_config
from hermes_constants import get_hermes_home

# The shipped .env template keeps provider keys commented out, and STT_OPENAI_BASE_URL contains the
# substring OPENAI_BASE_URL: neither configures a provider.
_TEMPLATE_ENV = "# OPENROUTER_API_KEY=\n# STT_OPENAI_BASE_URL=\nTERMINAL_TIMEOUT=60\n"


def _run_env_check(monkeypatch, capsys, config_yaml: str):
    home = get_hermes_home()
    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    (home / ".env").write_text(_TEMPLATE_ENV, encoding="utf-8")
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    finding = doctor_config._check_env_file(False)
    return finding, capsys.readouterr().out


def test_commented_template_keys_are_not_a_provider(monkeypatch, capsys):
    finding, out = _run_env_check(monkeypatch, capsys, "model:\n  provider: auto\n")

    assert "No usable inference provider" in out
    assert any("hermes model" in issue for issue in finding.issues)


def test_provider_without_env_key_is_configured(monkeypatch, capsys):
    finding, out = _run_env_check(monkeypatch, capsys, "model:\n  provider: anthropic\n  default: claude-opus-5-5\n")

    assert "Inference provider configured" in out and "(anthropic)" in out
    assert not finding.issues
