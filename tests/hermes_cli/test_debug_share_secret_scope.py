"""BJS-3738: support bundles scrub opaque profile values regardless of key name."""

from unittest.mock import patch

import pytest

from agent.redact import (
    clear_vault_redaction_values,
    redact_registered_vault_values,
    register_vault_redaction_value,
)
from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
from hermes_cli.debug import collect_share_bundle
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.parametrize("bind_scope", [False, True])
@pytest.mark.parametrize("value_count", [1, 70])
def test_bundle_scrubs_neutral_profile_values(tmp_path, monkeypatch, bind_scope, value_count):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    values = [f"opaque-value-{i:03d}-abcdefgh" for i in range(value_count)]
    scope_values = {f"NEUTRAL_{i}": value for i, value in enumerate(values)}
    scope_values.update(GITHUB_PAT="eightchr", MESSAGING_CWD="sevench")
    (home / ".env").write_text("\n".join(f"{k}={v}" for k, v in scope_values.items()))
    logs = home / "logs"
    logs.mkdir()
    line = " | ".join(scope_values.values()) + " | harmless diagnostic\n"
    (logs / "gateway.log").write_text(line)
    home_token = set_hermes_home_override(home)
    scope_token = None
    try:
        if bind_scope:
            scope_token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
        # Keep the real log capture and scope loader; the dump producer supplies synthetic text.
        with patch("hermes_cli.dump.run_dump", side_effect=lambda *a, **kw: print(line)):
            raw = collect_share_bundle(redact=False)
            scrubbed = collect_share_bundle(redact=True)
        assert "gateway.log" in scrubbed
        for label, text in scrubbed.items():
            assert "harmless diagnostic" in text
            assert "sevench" in text
            for value in [*values, "eightchr"]:
                assert value in raw[label]
                assert value not in text
    finally:
        if scope_token is not None:
            reset_secret_scope(scope_token)
        clear_vault_redaction_values()
        reset_hermes_home_override(home_token)


def test_bundle_does_not_evict_or_register_vault_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    values = [f"opaque-value-{i:03d}-abcdefgh" for i in range(70)]
    (tmp_path / ".env").write_text("\n".join(f"NEUTRAL_{i}={v}" for i, v in enumerate(values)))
    vault_value = "saved-browser-credential"
    register_vault_redaction_value(vault_value)
    try:
        with patch("hermes_cli.dump.run_dump", side_effect=lambda *a, **kw: print(vault_value)):
            bundle = collect_share_bundle(redact=True)
        assert vault_value not in "\n".join(bundle.values())
        assert vault_value not in redact_registered_vault_values(vault_value)
        # Scope values are for this bundle only, not persistent browser-output redaction.
        assert redact_registered_vault_values(values[0]) == values[0]
    finally:
        clear_vault_redaction_values()
