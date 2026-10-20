"""``auth.adopt_external_logins: false`` keeps Hermes off the host gh CLI login (#132992).

``resolve_copilot_token()`` falls back to ``gh auth token`` when no Copilot env var is set.
That borrows a third host login beside the Codex CLI and Claude Code ones, so the same
opt-out must govern it: with the switch off the fallback is refused without probing gh,
the ``gh_cli`` pool row an earlier (adopting) process persisted is dropped, and Copilot
env vars — the user's own configuration, not a borrowed login — keep working.
"""

from __future__ import annotations

from unittest.mock import patch

from agent import credential_sources
from agent.credential_pool import load_pool
from hermes_cli import copilot_auth
from hermes_cli.auth import read_credential_pool

FAKE_GH_TOKEN = "gho_" + "a" * 36


def _write_config(hermes_home, adopt):
    body = "model:\n  model: gpt-5\n"
    if adopt is not None:
        body += f"auth:\n  adopt_external_logins: {'true' if adopt else 'false'}\n# arm {adopt}\n"
    (hermes_home / "config.yaml").write_text(body)


def _isolate(hermes_home, monkeypatch):
    hermes_home.mkdir()
    (hermes_home / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for var in copilot_auth.COPILOT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(credential_sources, "_notice_logged", False, raising=False)
    copilot_auth._invalidate_gh_cli_token_cache()


def test_opt_out_skips_gh_fallback_and_prunes_pool_row(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _isolate(hermes_home, monkeypatch)

    # Arm A (default): the host gh login is borrowed and persisted as a gh_cli pool row.
    _write_config(hermes_home, adopt=None)
    with patch.object(copilot_auth, "_try_gh_cli_token", return_value=FAKE_GH_TOKEN):
        assert copilot_auth.resolve_copilot_token() == (FAKE_GH_TOKEN, "gh auth token")
        assert [e.source for e in load_pool("copilot").entries()] == ["gh_cli"]
    assert [e.get("source") for e in read_credential_pool("copilot")] == ["gh_cli"]

    # Arm B (opt-out): the gh probe never runs and the row persisted above is dropped.
    _write_config(hermes_home, adopt=False)
    with patch.object(
        copilot_auth, "_try_gh_cli_token", return_value=FAKE_GH_TOKEN
    ) as gh_probe:
        assert copilot_auth.resolve_copilot_token() == ("", "")
        assert [e.source for e in load_pool("copilot").entries()] == []
        gh_probe.assert_not_called()
    assert read_credential_pool("copilot") == []

    # Arm A again: flipping adoption back on borrows the gh login exactly as before.
    _write_config(hermes_home, adopt=True)
    with patch.object(copilot_auth, "_try_gh_cli_token", return_value=FAKE_GH_TOKEN):
        assert copilot_auth.resolve_copilot_token() == (FAKE_GH_TOKEN, "gh auth token")
        assert [e.source for e in load_pool("copilot").entries()] == ["gh_cli"]


def test_opt_out_keeps_explicit_env_var_tokens(tmp_path, monkeypatch):
    """The opt-out covers only the borrowed gh login; Copilot env vars stay usable (#132992)."""
    hermes_home = tmp_path / "hermes"
    _isolate(hermes_home, monkeypatch)
    _write_config(hermes_home, adopt=False)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_envsupplied")
    with patch.object(copilot_auth, "_try_gh_cli_token") as gh_probe:
        assert copilot_auth.resolve_copilot_token() == (
            "gho_envsupplied",
            "COPILOT_GITHUB_TOKEN",
        )
        gh_probe.assert_not_called()
