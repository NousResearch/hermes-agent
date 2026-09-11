"""Tests for external-process provider auth status and Accounts-tab wiring.

Covers the copilot-acp fix class:
  * ``get_auth_status()`` dispatches on ``auth_type == "external_process"``
    (not a hardcoded slug), so future ACP-style providers inherit the
    behaviour automatically.
  * ``auth_verified``/``auth_source`` carry positive credential evidence
    (env token or on-disk GitHub Copilot credential store) while remaining
    honest — no evidence means unknown, never "signed out".
  * The Accounts-tab sign-in ``cli_command`` reflects the executable the
    user actually configured (``HERMES_COPILOT_ACP_COMMAND`` /
    ``COPILOT_CLI_PATH``), and its default is a valid Copilot CLI
    invocation (``copilot login`` — ``copilot /login`` is not a command).
"""

import os

import pytest

from hermes_cli.auth import (
    get_auth_status,
    get_external_process_provider_status,
)


@pytest.fixture()
def _clean_copilot_env(monkeypatch):
    """Neutralize host state so tests pin behaviour, not this machine."""
    for var in (
        "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
        "HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH",
        "HERMES_COPILOT_ACP_ARGS", "COPILOT_ACP_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


# --- get_auth_status dispatches on auth_type, not slug ----------------------


def test_get_auth_status_dispatches_external_process_by_auth_type(
    tmp_path, monkeypatch, _clean_copilot_env
):
    fake = tmp_path / ("copilot.exe" if os.name == "nt" else "copilot")
    fake.write_text("", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(fake))
    # Point HOME somewhere empty so on-disk credential stores don't leak in.
    monkeypatch.setenv("HOME", str(tmp_path))

    status = get_auth_status("copilot-acp")

    # The external_process status shape, not the {"logged_in": False}
    # fallthrough — proves the dispatcher reached the right branch.
    assert status.get("provider") == "copilot-acp"
    assert status.get("configured") is True
    assert status.get("resolved_command") == str(fake)
    assert "auth_verified" in status


def test_external_process_status_rejects_wrong_auth_type():
    # A provider that exists but is not external_process must be refused —
    # the generic dispatcher relies on this guard.
    assert get_external_process_provider_status("openrouter") == {"configured": False}
    assert get_external_process_provider_status("no-such-provider") == {"configured": False}


# --- auth_verified: positive evidence only ----------------------------------


def test_auth_verified_false_without_evidence(tmp_path, monkeypatch, _clean_copilot_env):
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/github-copilot
    status = get_external_process_provider_status("copilot-acp")
    assert status["auth_verified"] is False
    assert status["auth_source"] is None


def test_auth_verified_from_supported_env_token(tmp_path, monkeypatch, _clean_copilot_env):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GH_TOKEN", "gho_" + "x" * 36)  # supported OAuth prefix

    status = get_external_process_provider_status("copilot-acp")

    assert status["auth_verified"] is True
    assert status["auth_source"] == "env: GH_TOKEN"


def test_classic_pat_is_not_login_evidence(tmp_path, monkeypatch, _clean_copilot_env):
    # ghp_* classic PATs are rejected by the Copilot API — presence of one
    # must not be presented as a working login.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GH_TOKEN", "ghp_" + "x" * 36)

    status = get_external_process_provider_status("copilot-acp")

    assert status["auth_verified"] is False


def test_auth_verified_from_on_disk_credential_store(tmp_path, monkeypatch, _clean_copilot_env):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = tmp_path / ".config" / "github-copilot"
    store.mkdir(parents=True)
    (store / "hosts.json").write_text(
        '{"github.com": {"oauth_token": "gho_test"}}', encoding="utf-8"
    )

    status = get_external_process_provider_status("copilot-acp")

    assert status["auth_verified"] is True
    assert status["auth_source"] == "~/.config/github-copilot/hosts.json"


def test_empty_credential_store_is_not_evidence(tmp_path, monkeypatch, _clean_copilot_env):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = tmp_path / ".config" / "github-copilot"
    store.mkdir(parents=True)
    (store / "hosts.json").write_text("{}", encoding="utf-8")  # logged out

    status = get_external_process_provider_status("copilot-acp")

    assert status["auth_verified"] is False


def test_auth_verified_from_copilot_cli_plaintext_store(tmp_path, monkeypatch, _clean_copilot_env):
    # `copilot login` without an OS keychain writes the token into
    # ~/.copilot/config.json (JSONC, with //-comment header lines).
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".copilot"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        "// User settings belong in settings.json.\n"
        "// This file is managed automatically.\n"
        "{\n"
        '  "copilotTokens": {"https://github.com:someuser": "gho_test"},\n'
        '  "lastLoggedInUser": {"host": "https://github.com", "login": "someuser"}\n'
        "}\n",
        encoding="utf-8",
    )

    status = get_external_process_provider_status("copilot-acp")

    assert status["auth_verified"] is True
    assert status["auth_source"] == "~/.copilot/config.json"


def test_copilot_cli_store_without_tokens_is_not_evidence(tmp_path, monkeypatch, _clean_copilot_env):
    # A config.json exists after first launch even before any login —
    # its presence alone must not read as signed-in.
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".copilot"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        '// managed\n{"firstLaunchAt": "2026-01-01T00:00:00Z", "copilotTokens": {}}\n',
        encoding="utf-8",
    )

    status = get_external_process_provider_status("copilot-acp")

    assert status["auth_verified"] is False


# --- desktop picker explicit-only filter ------------------------------------


def test_explicit_filter_keeps_signed_in_external_process_row(tmp_path, monkeypatch, _clean_copilot_env):
    # A verified CLI login leaves no trace in active_provider/config/env —
    # the explicit-only desktop filter must treat it like the Anthropic OAuth
    # carve-out and keep the row.
    from hermes_cli.inventory import _filter_explicit_provider_rows

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".copilot"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        '{"copilotTokens": {"https://github.com:u": "gho_test"}}', encoding="utf-8"
    )

    class _Ctx:
        current_provider = "nous"

    rows = [{"slug": "copilot-acp", "models": ["gpt-5.4"]}]
    kept = _filter_explicit_provider_rows(rows, _Ctx())

    assert any(r["slug"] == "copilot-acp" for r in kept), \
        "signed-in copilot-acp must survive the explicit-only picker filter"


def test_explicit_filter_drops_unverified_external_process_row(tmp_path, monkeypatch, _clean_copilot_env):
    # Merely having the executable on PATH is ambient discovery, not an
    # explicit configuration — the desktop filter keeps its narrower contract.
    from hermes_cli.inventory import _filter_explicit_provider_rows

    monkeypatch.setenv("HOME", str(tmp_path))  # no credential stores

    class _Ctx:
        current_provider = "nous"

    rows = [{"slug": "copilot-acp", "models": ["gpt-5.4"]}]
    kept = _filter_explicit_provider_rows(rows, _Ctx())

    assert all(r["slug"] != "copilot-acp" for r in kept)


# --- Accounts-tab cli_command ------------------------------------------------


def test_catalog_sign_in_command_is_a_valid_copilot_invocation():
    from hermes_cli.web_server_oauth import _OAUTH_PROVIDER_CATALOG

    entry = next(e for e in _OAUTH_PROVIDER_CATALOG if e["id"] == "copilot-acp")
    # `copilot /login` is not a valid invocation — slash-commands only exist
    # inside an interactive session. The catalog must hand users a command
    # that actually starts a login flow.
    assert entry["cli_command"] == "copilot login"


def test_cli_command_reflects_configured_executable(tmp_path, monkeypatch, _clean_copilot_env):
    from hermes_cli.web_server_oauth import _external_process_cli_command

    fake = tmp_path / ("copilot.exe" if os.name == "nt" else "copilot")
    fake.write_text("", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(fake))

    rendered = _external_process_cli_command("copilot-acp", "copilot login")

    assert rendered == f"{fake} login"


def test_cli_command_untouched_for_non_external_providers(_clean_copilot_env):
    from hermes_cli.web_server_oauth import _external_process_cli_command

    assert _external_process_cli_command("nous", "hermes auth add nous") == "hermes auth add nous"


def test_cli_command_default_when_no_override(monkeypatch, _clean_copilot_env):
    from hermes_cli.web_server_oauth import _external_process_cli_command

    assert _external_process_cli_command("copilot-acp", "copilot login") == "copilot login"


# --- live catalog key from the Copilot CLI store -----------------------------


def test_catalog_key_resolves_from_copilot_cli_store(tmp_path, monkeypatch, _clean_copilot_env):
    # A user whose ONLY credential is `copilot login` must still get the live
    # model catalog — otherwise the picker silently falls back to the stale
    # curated list (visibly wrong vs. what their subscription serves).
    from unittest.mock import patch as mock_patch

    from hermes_cli import models as models_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".copilot"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        "// managed\n"
        '{"copilotTokens": {"https://github.com:u": "gho_' + "x" * 36 + '"}}\n',
        encoding="utf-8",
    )

    with mock_patch.object(
        models_mod, "_resolve_copilot_catalog_api_key", wraps=models_mod._resolve_copilot_catalog_api_key
    ), mock_patch(
        "hermes_cli.copilot_auth.exchange_copilot_token",
        return_value=("exchanged-api-token", 0.0, None),
    ), mock_patch(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        side_effect=Exception("no env creds"),
    ), mock_patch(
        "hermes_cli.auth.read_credential_pool", return_value=[]
    ):
        key = models_mod._resolve_copilot_catalog_api_key()

    assert key == "exchanged-api-token"


def test_catalog_key_empty_when_cli_store_absent(tmp_path, monkeypatch, _clean_copilot_env):
    from unittest.mock import patch as mock_patch

    from hermes_cli import models as models_mod

    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.copilot at all

    with mock_patch(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        side_effect=Exception("no env creds"),
    ), mock_patch(
        "hermes_cli.auth.read_credential_pool", return_value=[]
    ):
        key = models_mod._resolve_copilot_catalog_api_key()

    assert key == ""
