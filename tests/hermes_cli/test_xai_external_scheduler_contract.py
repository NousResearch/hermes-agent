"""Fleet-owned xAI OAuth refresh writer contract."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL


def _jwt_with_exp(exp_epoch: int) -> str:
    import base64

    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": exp_epoch}).encode("utf-8")
    ).rstrip(b"=")
    return f"h.{payload.decode('utf-8')}.s"


def _xai_entry(*, access_token: str, refresh_token: str, label: str = "Neo Grok Sub") -> dict:
    return {
        "id": "neo-grok-sub",
        "label": label,
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "base_url": DEFAULT_XAI_OAUTH_BASE_URL,
    }


def _write_external_store(hermes_home: Path) -> None:
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "oauth:\n  refresh_owner: external\n",
        encoding="utf-8",
    )
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "xai-oauth": [
                        _xai_entry(
                            access_token="old-access",
                            refresh_token="old-refresh",
                        )
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_external_scheduler_authority_updates_xai_oauth_tokens(tmp_path, monkeypatch):
    from hermes_cli.auth import read_credential_pool, write_credential_pool

    hermes_home = tmp_path / "hermes"
    _write_external_store(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    write_credential_pool(
        "xai-oauth",
        [_xai_entry(access_token="new-access", refresh_token="new-refresh")],
        oauth_token_write_authority="external-scheduler",
    )

    [persisted] = read_credential_pool("xai-oauth")
    assert persisted["access_token"] == "new-access"
    assert persisted["refresh_token"] == "new-refresh"


@pytest.mark.parametrize("authority", [None, "scheduler", "external_scheduler"])
def test_external_owner_rejects_unsanctioned_xai_token_writers(
    tmp_path,
    monkeypatch,
    authority,
):
    from hermes_cli.auth import read_credential_pool, write_credential_pool

    hermes_home = tmp_path / "hermes"
    _write_external_store(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    kwargs = {}
    if authority is not None:
        kwargs["oauth_token_write_authority"] = authority
    write_credential_pool(
        "xai-oauth",
        [_xai_entry(access_token="stolen-access", refresh_token="stolen-refresh")],
        **kwargs,
    )

    [persisted] = read_credential_pool("xai-oauth")
    assert persisted["access_token"] == "old-access"
    assert persisted["refresh_token"] == "old-refresh"


def test_standalone_xai_pool_write_remains_compatible(tmp_path, monkeypatch):
    from hermes_cli.auth import read_credential_pool, write_credential_pool

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    write_credential_pool(
        "xai-oauth",
        [_xai_entry(access_token="standalone-access", refresh_token="standalone-refresh")],
    )

    [persisted] = read_credential_pool("xai-oauth")
    assert persisted["access_token"] == "standalone-access"
    assert persisted["refresh_token"] == "standalone-refresh"


@pytest.mark.parametrize(
    "config_text",
    [
        "oauth:\n  refresh_owner: external\n",
        "oauth:\n  refresh_owner: fleet\n",
        "oauth: []\n",
        "{]\n",
    ],
)
def test_legacy_singleton_refresh_fails_closed_without_spending_external_token(
    tmp_path,
    monkeypatch,
    config_text,
):
    from hermes_cli.auth import AuthError, _refresh_xai_oauth_tokens

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(config_text, encoding="utf-8")
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    refresh_calls = []
    monkeypatch.setattr(
        "hermes_cli.auth.refresh_xai_oauth_pure",
        lambda *_args, **_kwargs: refresh_calls.append(True),
    )

    with pytest.raises(AuthError) as exc_info:
        _refresh_xai_oauth_tokens(
            {"access_token": "stale-access", "refresh_token": "fleet-refresh"},
            token_endpoint="https://auth.x.ai/oauth2/token",
            timeout_seconds=5.0,
        )

    assert exc_info.value.code == "xai_external_refresh_forbidden"
    assert exc_info.value.relogin_required is False
    assert refresh_calls == []


def test_runtime_resolver_adopts_scheduler_pool_without_refreshing_singleton(
    tmp_path,
    monkeypatch,
):
    from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

    hermes_home = tmp_path / "hermes"
    _write_external_store(hermes_home)
    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    payload["providers"]["xai-oauth"] = {
        "tokens": {
            "access_token": _jwt_with_exp(int(time.time()) - 60),
            "refresh_token": "stale-singleton-refresh",
        },
        "discovery": {"token_endpoint": "https://auth.x.ai/oauth2/token"},
    }
    scheduler_access = _jwt_with_exp(int(time.time()) + 7200)
    payload["credential_pool"]["xai-oauth"][0].update(
        access_token=scheduler_access,
        refresh_token="scheduler-refresh",
    )
    (hermes_home / "auth.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    refresh_calls = []
    monkeypatch.setattr(
        "hermes_cli.auth.refresh_xai_oauth_pure",
        lambda *_args, **_kwargs: refresh_calls.append(True),
    )

    credentials = resolve_xai_oauth_runtime_credentials(force_refresh=True)

    assert credentials["api_key"] == scheduler_access
    assert refresh_calls == []


def test_external_pool_proactive_and_reactive_refresh_only_adopt_disk_tokens(
    tmp_path,
    monkeypatch,
):
    from agent.credential_pool import CredentialPool, PooledCredential

    hermes_home = tmp_path / "hermes"
    _write_external_store(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    expiring_access = _jwt_with_exp(int(time.time()) + 30)
    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    payload["credential_pool"]["xai-oauth"][0]["access_token"] = expiring_access
    (hermes_home / "auth.json").write_text(json.dumps(payload), encoding="utf-8")
    entry = PooledCredential.from_dict(
        "xai-oauth", payload["credential_pool"]["xai-oauth"][0]
    )
    pool = CredentialPool("xai-oauth", [entry])
    refresh_calls = []
    monkeypatch.setattr(
        "hermes_cli.auth.refresh_xai_oauth_pure",
        lambda *_args, **_kwargs: refresh_calls.append(True),
    )

    selected = pool.select()
    assert selected is not None
    assert selected.access_token == expiring_access
    assert refresh_calls == []

    assert pool.try_refresh_current() is None
    assert refresh_calls == []

    scheduler_access = _jwt_with_exp(int(time.time()) + 7200)
    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    payload["credential_pool"]["xai-oauth"][0].update(
        access_token=scheduler_access,
        refresh_token="scheduler-rotated-refresh",
    )
    (hermes_home / "auth.json").write_text(json.dumps(payload), encoding="utf-8")

    adopted = pool.try_refresh_current()
    assert adopted is not None
    assert adopted.access_token == scheduler_access
    assert adopted.refresh_token == "scheduler-rotated-refresh"
    assert refresh_calls == []


@pytest.mark.parametrize("requested_label", [None, "Neo Grok Sub", "generic lane"])
def test_interactive_xai_add_uses_named_profile_dedicated_lane_without_secret_output(
    tmp_path,
    monkeypatch,
    capsys,
    requested_label,
):
    from hermes_cli.auth_commands import auth_add_command

    profile_home = tmp_path / ".hermes" / "profiles" / "neo"
    _write_external_store(profile_home)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(
        "hermes_cli.auth._xai_oauth_device_code_login",
        lambda **_kwargs: {
            "tokens": {
                "access_token": "interactive-access-secret",
                "refresh_token": "interactive-refresh-secret",
            },
            "base_url": DEFAULT_XAI_OAUTH_BASE_URL,
            "last_refresh": "2026-08-15T10:00:00Z",
        },
    )

    auth_add_command(
        SimpleNamespace(
            provider="xai-oauth",
            auth_type="oauth",
            api_key=None,
            label=requested_label,
            timeout=3,
            no_browser=True,
        )
    )

    payload = json.loads((profile_home / "auth.json").read_text(encoding="utf-8"))
    added = next(
        entry
        for entry in payload["credential_pool"]["xai-oauth"]
        if entry["access_token"] == "interactive-access-secret"
    )
    assert added["label"] == "Neo Grok Sub"
    output = capsys.readouterr().out
    assert "interactive-access-secret" not in output
    assert "interactive-refresh-secret" not in output
