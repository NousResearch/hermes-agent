from __future__ import annotations

from types import SimpleNamespace


def _args(label: str) -> SimpleNamespace:
    return SimpleNamespace(
        provider="openai-codex",
        auth_type="oauth",
        label=label,
        no_browser=True,
        timeout=None,
        api_key=None,
    )


def _seed_pool(tmp_path, token: str = "token-a") -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        '{"version":1,"credential_pool":{"openai-codex":[{"id":"cred1","label":"primary","auth_type":"oauth","priority":0,"source":"manual:device_code","access_token":"'
        + token
        + '","refresh_token":"refresh-a"}]}}\n'
    )


def test_codex_auth_add_warns_when_new_oauth_token_duplicates_existing_pool_entry(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _seed_pool(tmp_path, token="same-access-token")

    import hermes_cli.auth_commands as auth_commands

    monkeypatch.setattr(
        auth_commands.auth_mod,
        "_codex_device_code_login",
        lambda: {
            "tokens": {
                "access_token": "same-access-token",
                "refresh_token": "refresh-b",
            },
            "base_url": "https://chatgpt.com/backend-api/codex",
            "last_refresh": "2026-01-01T00:00:00Z",
        },
    )

    auth_commands.auth_add_command(_args("secondary"))

    out = capsys.readouterr().out
    assert 'Added openai-codex OAuth credential #2: "secondary"' in out
    assert "Warning: this OAuth token matches existing openai-codex credential #1" in out
    assert '"primary"' in out
    assert "same-access-token" not in out
    assert "switch browser accounts" in out.lower()


def test_codex_auth_add_does_not_warn_for_distinct_oauth_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _seed_pool(tmp_path, token="access-token-a")

    import hermes_cli.auth_commands as auth_commands

    monkeypatch.setattr(
        auth_commands.auth_mod,
        "_codex_device_code_login",
        lambda: {
            "tokens": {
                "access_token": "access-token-b",
                "refresh_token": "refresh-b",
            },
            "base_url": "https://chatgpt.com/backend-api/codex",
            "last_refresh": "2026-01-01T00:00:00Z",
        },
    )

    auth_commands.auth_add_command(_args("secondary"))

    out = capsys.readouterr().out
    assert 'Added openai-codex OAuth credential #2: "secondary"' in out
    assert "Warning: this OAuth token matches" not in out
