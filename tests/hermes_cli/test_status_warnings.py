from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import status_warnings


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_config(path: Path, *, provider: str = "openai-codex", version: int = 28) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"_config_version: {version}\nmodel:\n  provider: {provider}\n",
        encoding="utf-8",
    )


@pytest.fixture()
def profile_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "liz"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    _write_config(profile / "config.yaml")
    return {"root": root, "profile": profile}


def test_provider_auth_diagnostic_reports_profile_local_pool_and_unset_active_provider(profile_layout):
    _write_json(profile_layout["profile"] / "auth.json", {
        "version": 1,
        "credential_pool": {
            "openai-codex": [{
                "id": "local-1",
                "access_token": "should-not-leak",
                "refresh_token": "should-not-leak",
                "last_status": "ok",
            }],
        },
    })

    diag = status_warnings.collect_provider_auth_diagnostic(
        config={"model": {"provider": "openai-codex"}},
        hermes_home=profile_layout["profile"],
        default_root=profile_layout["root"],
    )

    assert diag["credential_pool"]["source"] == "local"
    assert diag["credential_pool"]["entry_statuses"] == [{"last_status": "ok"}]
    assert [w["code"] for w in diag["warnings"]] == ["auth_active_provider_drift"]
    rendered = json.dumps(diag)
    assert "should-not-leak" not in rendered
    status_warnings.assert_no_secretish_keys(diag)


def test_provider_auth_diagnostic_reports_global_fallback_for_named_profile(profile_layout):
    _write_json(profile_layout["root"] / "auth.json", {
        "version": 1,
        "credential_pool": {
            "openai-codex": [{
                "id": "global-1",
                "api_key": "should-not-leak",
                "last_error_code": "expired_token",
                "last_error_reason": "refresh_failed",
            }],
        },
    })
    _write_json(profile_layout["profile"] / "auth.json", {
        "version": 1,
        "active_provider": "openai-codex",
        "credential_pool": {},
    })

    diag = status_warnings.collect_provider_auth_diagnostic(
        config={"model": {"provider": "openai-codex"}},
        hermes_home=profile_layout["profile"],
        default_root=profile_layout["root"],
    )

    assert diag["credential_pool"]["source"] == "global-fallback"
    assert diag["credential_pool"]["entry_statuses"] == [{
        "last_error_code": "expired_token",
        "last_error_reason": "refresh_failed",
    }]
    assert [w["code"] for w in diag["warnings"]] == ["profile_global_auth_fallback"]
    assert "should-not-leak" not in json.dumps(diag)
    status_warnings.assert_no_secretish_keys(diag)


def test_provider_auth_diagnostic_reports_none_when_no_pool(profile_layout):
    _write_json(profile_layout["profile"] / "auth.json", {
        "version": 1,
        "active_provider": "anthropic",
    })

    diag = status_warnings.collect_provider_auth_diagnostic(
        config={"model": {"provider": "openai-codex"}},
        hermes_home=profile_layout["profile"],
        default_root=profile_layout["root"],
    )

    assert diag["credential_pool"]["source"] == "none"
    assert diag["credential_pool"]["entries"] == 0
    assert [w["code"] for w in diag["warnings"]] == ["auth_active_provider_drift"]


def test_config_version_drift_lists_profiles_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    _write_config(root / "config.yaml", version=27)
    _write_config(root / "profiles" / "liz" / "config.yaml", version=24)
    _write_config(root / "profiles" / "ellie" / "config.yaml", version=28)
    _write_json(root / "profiles" / "liz" / "auth.json", {"access_token": "should-not-leak"})

    drift = status_warnings.collect_config_version_drift(
        default_root=root,
        latest_config_version=28,
    )

    assert drift["behind"] == [
        {"profile": "default", "config_version": 27, "latest_config_version": 28},
        {"profile": "liz", "config_version": 24, "latest_config_version": 28},
    ]
    assert drift["warning"]["code"] == "profile_config_version_drift"
    assert "should-not-leak" not in json.dumps(drift)


def test_launchd_warning_when_pid_running_but_last_exit_status_nonzero(monkeypatch):
    monkeypatch.setattr(status_warnings.sys, "platform", "darwin")
    monkeypatch.setattr("hermes_cli.gateway.get_launchd_label", lambda: "ai.hermes.gateway-liz")

    def fake_run(*args, **kwargs):
        assert args[0] == ["launchctl", "list", "ai.hermes.gateway-liz"]
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout='''{
    "Label" = "ai.hermes.gateway-liz";
    "LastExitStatus" = 256;
    "PID" = 97157;
}''',
            stderr="",
        )

    warning = status_warnings.collect_launchd_gateway_warning(
        gateway_pid=97157,
        run=fake_run,
    )

    assert warning == {
        "code": "launchd_gateway_state_mismatch",
        "severity": "warning",
        "message": "gateway PID is running but launchd reports a nonzero status/LastExitStatus",
        "label": "ai.hermes.gateway-liz",
        "gateway_pid": 97157,
        "launchctl_status": 1,
        "launchd_pid": 97157,
        "last_exit_status": 256,
    }


def test_launchd_no_warning_when_launchd_status_clean(monkeypatch):
    monkeypatch.setattr(status_warnings.sys, "platform", "darwin")
    monkeypatch.setattr("hermes_cli.gateway.get_launchd_label", lambda: "ai.hermes.gateway-liz")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='''{
    "LastExitStatus" = 0;
    "PID" = 97157;
}''',
            stderr="",
        )

    assert status_warnings.collect_launchd_gateway_warning(
        gateway_pid=97157,
        run=fake_run,
    ) is None
