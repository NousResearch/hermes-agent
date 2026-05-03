import json

import pytest

from hermes_cli.auth import AuthError, _read_codex_tokens, read_credential_pool


def _write_auth(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_inherit_root_config(profile):
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "config.yaml").write_text("auth:\n  inherit_root: true\n")


def _root_auth_payload():
    return {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "root-access",
                    "refresh_token": "root-refresh",
                },
                "last_refresh": "2026-05-03T00:00:00Z",
            }
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "root1",
                    "label": "device_code",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "device_code",
                    "access_token": "root-access",
                    "refresh_token": "root-refresh",
                }
            ]
        },
    }


def test_named_profile_does_not_inherit_root_codex_auth_by_default(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "planchet"

    _write_auth(root / "auth.json", _root_auth_payload())
    _write_auth(
        profile / "auth.json",
        {"version": 1, "providers": {}, "credential_pool": {"openrouter": []}},
    )

    monkeypatch.setenv("HERMES_HOME", str(profile))

    with pytest.raises(AuthError, match="No Codex credentials"):
        _read_codex_tokens(_lock=False)
    assert read_credential_pool("openai-codex") == []


def test_named_profile_inherits_root_codex_auth_when_enabled(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "planchet"

    _write_auth(root / "auth.json", _root_auth_payload())
    _write_auth(
        profile / "auth.json",
        {"version": 1, "providers": {}, "credential_pool": {"openrouter": []}},
    )
    _write_inherit_root_config(profile)

    monkeypatch.setenv("HERMES_HOME", str(profile))

    tokens = _read_codex_tokens(_lock=False)
    assert tokens["tokens"]["access_token"] == "root-access"

    pool = read_credential_pool("openai-codex")
    assert len(pool) == 1
    assert pool[0]["access_token"] == "root-access"


def test_profile_local_auth_overrides_root_fallback_when_enabled(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "planchet"

    _write_auth(root / "auth.json", _root_auth_payload())
    _write_auth(
        profile / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "profile-access",
                        "refresh_token": "profile-refresh",
                    }
                }
            },
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "local1",
                        "label": "device_code",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "device_code",
                        "access_token": "profile-access",
                        "refresh_token": "profile-refresh",
                    }
                ]
            },
        },
    )
    _write_inherit_root_config(profile)

    monkeypatch.setenv("HERMES_HOME", str(profile))

    tokens = _read_codex_tokens(_lock=False)
    assert tokens["tokens"]["access_token"] == "profile-access"

    pool = read_credential_pool("openai-codex")
    assert len(pool) == 1
    assert pool[0]["access_token"] == "profile-access"
