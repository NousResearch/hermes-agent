"""#81952: recovery is explicit; corrupt config cannot silently choose a paid provider."""
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_corrupt_profile_exposes_recovery_and_refuses_noninteractive_admission(tmp_path, monkeypatch, capsys):
    from hermes_cli import config
    from hermes_cli.auth import AuthError, resolve_provider
    from hermes_cli.config_read_errors import FailedConfigRead, get_active_config_parse_failure

    home = tmp_path / "profiles" / "reviewer"
    home.mkdir(parents=True)
    path = home / "config.yaml"
    broken = "model: {default: [unclosed\n\tbad: :::\n"
    path.write_text(broken, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-not-a-live-key")
    token = set_hermes_home_override(home)
    try:
        recovered = config.load_config()
        assert isinstance(recovered, FailedConfigRead)
        assert recovered.read_error is not None
        assert get_active_config_parse_failure()
        assert "formatting error" in capsys.readouterr().err
        with pytest.raises(config.InvalidUserConfigError):
            config.require_parseable_user_config()
        with pytest.raises(AuthError) as refused:
            resolve_provider("auto")
        assert refused.value.code == "corrupt_config"
        assert path.read_text(encoding="utf-8") == broken
        # Repair in the same profile restores ordinary admission, not a permanent refusal.
        path.write_text("model:\n  provider: openai-codex\n  default: fixture-model\n", encoding="utf-8")
        config.require_parseable_user_config()
        assert not isinstance(config.load_config(), FailedConfigRead)
        assert get_active_config_parse_failure() is None
    finally:
        reset_hermes_home_override(token)
