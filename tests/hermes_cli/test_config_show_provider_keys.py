"""`config show` must surface the real provider key set from ``_inject_profile_env_vars()``,
not just the 10 hardcoded keys in ``_SHOW_CONFIG_API_KEYS``.

Bug: ``show_config()`` rendered only the hardcoded provider keys, so users could not
see which of the real provider credentials (registered in ``OPTIONAL_ENV_VARS`` with
``category == "provider"``) were actually configured. Values must stay masked —
only key labels are displayed.
"""
import pytest


def _show(tmp_path, monkeypatch, capsys, **env):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    (home / "config.yaml").write_text("model:\n  default: user/model\n", encoding="utf-8")
    import hermes_cli.config as cfg
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    from hermes_cli.config import show_config
    show_config()
    return capsys.readouterr().out


SECRET = "sk-gemini-test-secret-abc123xyz"


class TestConfigShowProviderKeys:
    def test_injected_provider_key_is_displayed_and_masked(self, tmp_path, monkeypatch, capsys):
        """GEMINI_API_KEY is registered via the provider-injection path (not one of the
        hardcoded 10) — a configured provider key must appear, but never raw."""
        from hermes_cli.config import OPTIONAL_ENV_VARS, _SHOW_CONFIG_API_KEYS

        meta = OPTIONAL_ENV_VARS.get("GEMINI_API_KEY")
        assert meta and meta.get("category") == "provider" and meta.get("password")
        assert "GEMINI_API_KEY" not in {k for k, _ in _SHOW_CONFIG_API_KEYS}

        out = _show(tmp_path, monkeypatch, capsys, GEMINI_API_KEY=SECRET)
        assert "Google AI Studio" in out          # friendly label, like the hardcoded rows
        assert SECRET not in out                  # raw value never printed
        assert "sk-g...3xyz" in out               # same masking convention as the hardcoded keys

    def test_provider_rows_are_deduplicated_and_credential_only(self, tmp_path, monkeypatch, capsys):
        """OPENROUTER_API_KEY is both hardcoded and provider-injected (one row); the
        special Anthropic row covers ANTHROPIC_API_KEY/TOKEN (no second row); and
        base-URL overrides are not credentials — this section is for keys."""
        out = _show(tmp_path, monkeypatch, capsys, OPENAI_BASE_URL="https://example.invalid/v1")
        assert len([l for l in out.splitlines() if l.strip().startswith("OpenRouter")]) == 1
        assert len([l for l in out.splitlines() if l.strip().startswith("Anthropic")]) == 1
        assert "https://example.invalid/v1" not in out
