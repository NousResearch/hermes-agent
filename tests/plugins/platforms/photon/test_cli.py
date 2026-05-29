"""Tests for Photon CLI helpers."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon import cli as photon_cli
from plugins.platforms.photon import auth as photon_auth
from plugins.platforms.photon import tunnel as photon_tunnel


class _Proc:
    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture(autouse=True)
def _clear_photon_gateway_auth_env(monkeypatch: Any) -> None:
    monkeypatch.delenv("PHOTON_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("PHOTON_ALLOW_ALL_USERS", raising=False)


def _setup_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "project_name": None,
        "phone": "",
        "first_name": None,
        "last_name": None,
        "email": None,
        "no_browser": True,
        "new_project": False,
        "skip_sidecar_install": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _patch_setup_basics(monkeypatch: Any) -> None:
    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(photon_auth, "setup_lock", lambda: nullcontext())


def _with_node_modules(tmp_path: Path, monkeypatch: Any) -> Path:
    sidecar = tmp_path / "sidecar"
    (sidecar / "node_modules").mkdir(parents=True)
    monkeypatch.setattr(photon_cli, "_SIDECAR_DIR", sidecar)
    monkeypatch.setattr(photon_cli.shutil, "which", lambda _name: "/usr/bin/npm")
    return sidecar


def test_register_cli_parses_quick_setup_and_tunnel_commands() -> None:
    parser = argparse.ArgumentParser()
    photon_cli.register_cli(parser)

    quick = parser.parse_args(["quick-setup", "--phone", "+1234567"])
    assert quick.photon_command == "quick-setup"
    assert quick.phone == "+1234567"

    allow = parser.parse_args(["allow-phone", "+1234567"])
    assert allow.photon_command == "allow-phone"
    assert allow.phone == "+1234567"

    login = parser.parse_args(["login", "--debug-auth"])
    assert login.photon_command == "login"
    assert login.debug_auth is True

    tunnel = parser.parse_args(["webhook", "tunnel", "start"])
    assert tunnel.photon_command == "webhook"
    assert tunnel.photon_webhook_command == "tunnel"
    assert tunnel.photon_tunnel_command == "start"


def test_setup_reuses_local_project_without_remote_create(monkeypatch: Any) -> None:
    _patch_setup_basics(monkeypatch)
    monkeypatch.setattr(
        photon_auth,
        "load_project_credentials",
        lambda: ("spectrum-local", "secret-local"),
    )
    monkeypatch.setattr(
        photon_auth,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("created")),
    )

    rc = photon_cli._cmd_setup(_setup_args())

    assert rc == 0


def test_setup_adopts_single_named_remote_project(monkeypatch: Any) -> None:
    _patch_setup_basics(monkeypatch)
    stored: dict[str, Any] = {}
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [{
        "id": "dash-1",
        "name": "Hermes Agent",
        "spectrum": True,
        "platforms": ["imessage"],
        "spectrumProjectId": "spectrum-1",
        "projectSecret": "secret-1",
    }])
    monkeypatch.setattr(
        photon_auth,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("created")),
    )

    def fake_store(project_id: str, project_secret: str, **extra: Any) -> None:
        stored["project_id"] = project_id
        stored["project_secret"] = project_secret
        stored["extra"] = extra

    monkeypatch.setattr(photon_auth, "store_project_credentials", fake_store)

    rc = photon_cli._cmd_setup(_setup_args())

    assert rc == 0
    assert stored["project_id"] == "spectrum-1"
    assert stored["project_secret"] == "secret-1"
    assert stored["extra"]["dashboard_project_id"] == "dash-1"
    assert stored["extra"]["source"] == "remote-adopted"


def test_setup_fetches_details_before_adopting_remote_project(
    monkeypatch: Any,
) -> None:
    _patch_setup_basics(monkeypatch)
    stored: dict[str, Any] = {}
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [{
        "id": "dash-1",
        "name": "Hermes Agent",
        "spectrum": True,
        "platforms": ["imessage"],
        "spectrumProjectId": "spectrum-1",
    }])
    monkeypatch.setattr(photon_auth, "get_project", lambda _token, _project_id: {
        "id": "dash-1",
        "name": "Hermes Agent",
        "spectrum": True,
        "platforms": ["imessage"],
        "spectrumProjectId": "spectrum-1",
        "projectSecret": "secret-1",
    })
    monkeypatch.setattr(
        photon_auth,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("created")),
    )

    def fake_store(project_id: str, project_secret: str, **extra: Any) -> None:
        stored["project_id"] = project_id
        stored["project_secret"] = project_secret
        stored["extra"] = extra

    monkeypatch.setattr(photon_auth, "store_project_credentials", fake_store)

    rc = photon_cli._cmd_setup(_setup_args())

    assert rc == 0
    assert stored["project_id"] == "spectrum-1"
    assert stored["project_secret"] == "secret-1"
    assert stored["extra"]["source"] == "remote-adopted"


def test_setup_stops_on_multiple_named_remote_projects(monkeypatch: Any) -> None:
    _patch_setup_basics(monkeypatch)
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [
        {
            "id": "dash-1",
            "name": "Hermes Agent",
            "spectrum": True,
            "platforms": ["imessage"],
            "spectrumProjectId": "spectrum-1",
            "projectSecret": "secret-1",
        },
        {
            "id": "dash-2",
            "name": "Hermes Agent",
            "spectrum": True,
            "platforms": ["imessage"],
            "spectrumProjectId": "spectrum-2",
            "projectSecret": "secret-2",
        },
    ])
    monkeypatch.setattr(
        photon_auth,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("created")),
    )

    rc = photon_cli._cmd_setup(_setup_args())

    assert rc == 1


def test_setup_noninteractive_does_not_create_without_flag(monkeypatch: Any) -> None:
    _patch_setup_basics(monkeypatch)
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [])
    monkeypatch.setattr(
        photon_auth,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("created")),
    )
    monkeypatch.setattr(photon_cli, "_confirm_new_project", lambda _name: False)

    rc = photon_cli._cmd_setup(_setup_args())

    assert rc == 1


def test_setup_clears_bad_dashboard_token_on_project_list_auth_error(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    _patch_setup_basics(monkeypatch)
    cleared: list[bool] = []
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(
        photon_auth,
        "list_projects",
        lambda _token: (_ for _ in ()).throw(
            photon_auth.PhotonDashboardAuthError("Photon rejected token")
        ),
    )
    monkeypatch.setattr(
        photon_auth,
        "clear_photon_token",
        lambda: cleared.append(True) or True,
    )

    rc = photon_cli._cmd_setup(_setup_args())

    captured = capsys.readouterr()
    assert rc == 1
    assert cleared == [True]
    assert "hermes photon login" in captured.err


def test_quick_setup_auto_creates_when_no_remote_project(monkeypatch: Any) -> None:
    _patch_setup_basics(monkeypatch)
    stored: dict[str, Any] = {}
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [])
    monkeypatch.setattr(photon_auth, "create_project", lambda _token, *, name: {
        "id": "dash-quick",
        "name": name,
        "spectrumProjectId": "spectrum-quick",
        "projectSecret": "secret-quick",
        "platforms": ["imessage"],
    })
    monkeypatch.setattr(photon_auth, "create_user", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(photon_cli, "_start_managed_tunnel_and_register", lambda: 0)

    def fake_store(project_id: str, project_secret: str, **extra: Any) -> None:
        stored["project_id"] = project_id
        stored["project_secret"] = project_secret
        stored["extra"] = extra

    monkeypatch.setattr(photon_auth, "store_project_credentials", fake_store)

    rc = photon_cli._cmd_quick_setup(_setup_args(phone="+1234567"))

    assert rc == 0
    assert stored["project_id"] == "spectrum-quick"
    assert stored["extra"]["source"] == "auto-new"
    assert photon_auth.load_allowed_phone_numbers() == ["+1234567"]


def test_quick_setup_fetches_created_project_credentials(monkeypatch: Any) -> None:
    _patch_setup_basics(monkeypatch)
    stored: dict[str, Any] = {}
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [])
    monkeypatch.setattr(photon_auth, "create_project", lambda _token, *, name: {
        "id": "dash-new",
        "name": name,
        "spectrum": True,
        "platforms": ["imessage"],
    })
    monkeypatch.setattr(photon_auth, "get_project", lambda _token, _project_id: {
        "id": "dash-new",
        "name": "Hermes Agent",
        "spectrum": True,
        "platforms": ["imessage"],
        "spectrumProjectId": "spectrum-new",
        "projectSecret": "secret-new",
    })
    monkeypatch.setattr(
        photon_auth,
        "regenerate_project_secret",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("regenerated")
        ),
    )
    monkeypatch.setattr(photon_auth, "create_user", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(photon_cli, "_start_managed_tunnel_and_register", lambda: 0)

    def fake_store(project_id: str, project_secret: str, **extra: Any) -> None:
        stored["project_id"] = project_id
        stored["project_secret"] = project_secret
        stored["extra"] = extra

    monkeypatch.setattr(photon_auth, "store_project_credentials", fake_store)

    rc = photon_cli._cmd_quick_setup(_setup_args(phone="+1234567"))

    assert rc == 0
    assert stored["project_id"] == "spectrum-new"
    assert stored["project_secret"] == "secret-new"
    assert stored["extra"]["dashboard_project_id"] == "dash-new"


def test_quick_setup_regenerates_secret_for_new_project(
    monkeypatch: Any,
) -> None:
    _patch_setup_basics(monkeypatch)
    stored: dict[str, Any] = {}
    regenerated: list[str] = []
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [])
    monkeypatch.setattr(photon_auth, "create_project", lambda _token, *, name: {
        "id": "dash-new",
        "name": name,
        "spectrum": True,
        "platforms": ["imessage"],
    })
    monkeypatch.setattr(photon_auth, "get_project", lambda _token, _project_id: {
        "id": "dash-new",
        "name": "Hermes Agent",
        "spectrum": True,
        "platforms": ["imessage"],
        "spectrumProjectId": "spectrum-new",
    })

    def fake_regenerate(_token: str, project_id: str) -> dict[str, str]:
        regenerated.append(project_id)
        return {"projectSecret": "secret-new"}

    monkeypatch.setattr(photon_auth, "regenerate_project_secret", fake_regenerate)
    monkeypatch.setattr(photon_auth, "create_user", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(photon_cli, "_start_managed_tunnel_and_register", lambda: 0)

    def fake_store(project_id: str, project_secret: str, **extra: Any) -> None:
        stored["project_id"] = project_id
        stored["project_secret"] = project_secret
        stored["extra"] = extra

    monkeypatch.setattr(photon_auth, "store_project_credentials", fake_store)

    rc = photon_cli._cmd_quick_setup(_setup_args(phone="+1234567"))

    assert rc == 0
    assert regenerated == ["dash-new"]
    assert stored["project_id"] == "spectrum-new"
    assert stored["project_secret"] == "secret-new"
    assert stored["extra"]["dashboard_project_id"] == "dash-new"


def test_quick_setup_requires_login_first(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: None)
    monkeypatch.setattr(
        photon_auth,
        "load_project_credentials",
        lambda: (None, None),
    )
    monkeypatch.setattr(
        photon_cli,
        "_run_base_setup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("base setup")),
    )

    rc = photon_cli._cmd_quick_setup(_setup_args(phone="+1234567"))

    out = capsys.readouterr().out
    assert rc == 1
    assert "hermes photon login" in out
    assert "hermes photon quick-setup --phone '+<country-code><number>'" in out


def test_quick_setup_reuses_project_credentials_without_login(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: None)
    monkeypatch.setattr(
        photon_auth,
        "load_project_credentials",
        lambda: ("spectrum-local", "secret-local"),
    )
    monkeypatch.setattr(photon_auth, "setup_lock", lambda: nullcontext())
    monkeypatch.setattr(
        photon_auth,
        "list_projects",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("listed")),
    )
    monkeypatch.setattr(
        photon_auth,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("created")),
    )
    monkeypatch.setattr(photon_auth, "create_user", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(photon_cli, "_start_managed_tunnel_and_register", lambda: 0)

    rc = photon_cli._cmd_quick_setup(_setup_args(phone="+1234567"))

    assert rc == 0
    assert photon_auth.load_allowed_phone_numbers() == ["+1234567"]


def test_allow_phone_command_writes_photon_allowlist(capsys: Any) -> None:
    rc = photon_cli._cmd_allow_phone(argparse.Namespace(phone="+15551234567"))

    captured = capsys.readouterr()
    assert rc == 0
    assert photon_auth.load_allowed_phone_numbers() == ["+15551234567"]
    assert "phone authorized" in captured.out
    assert "hermes gateway restart" in captured.out


def test_setup_new_project_flag_creates_and_stores(monkeypatch: Any) -> None:
    _patch_setup_basics(monkeypatch)
    stored: dict[str, Any] = {}
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [])
    monkeypatch.setattr(photon_auth, "create_project", lambda _token, *, name: {
        "id": "dash-1",
        "name": name,
        "spectrumProjectId": "spectrum-1",
        "projectSecret": "secret-1",
        "platforms": ["imessage"],
    })

    def fake_store(project_id: str, project_secret: str, **extra: Any) -> None:
        stored["project_id"] = project_id
        stored["project_secret"] = project_secret
        stored["extra"] = extra

    monkeypatch.setattr(photon_auth, "store_project_credentials", fake_store)

    rc = photon_cli._cmd_setup(_setup_args(new_project=True))

    assert rc == 0
    assert stored["project_id"] == "spectrum-1"
    assert stored["project_secret"] == "secret-1"
    assert stored["extra"]["source"] == "explicit-new"


def test_setup_new_project_clears_bad_dashboard_token_on_create_auth_error(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    _patch_setup_basics(monkeypatch)
    cleared: list[bool] = []
    monkeypatch.setattr(
        photon_auth,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            photon_auth.PhotonDashboardAuthError("Photon rejected token")
        ),
    )
    monkeypatch.setattr(
        photon_auth,
        "clear_photon_token",
        lambda: cleared.append(True) or True,
    )

    rc = photon_cli._cmd_setup(_setup_args(new_project=True))

    captured = capsys.readouterr()
    assert rc == 1
    assert cleared == [True]
    assert "hermes photon login" in captured.err


def test_setup_new_project_writes_project_env(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("PHOTON_PROJECT_ID", raising=False)
    monkeypatch.delenv("PHOTON_PROJECT_SECRET", raising=False)
    _patch_setup_basics(monkeypatch)
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [])
    monkeypatch.setattr(photon_auth, "create_project", lambda _token, *, name: {
        "id": "dash-1",
        "name": name,
        "spectrumProjectId": "spectrum-1",
        "projectSecret": "secret-1",
        "platforms": ["imessage"],
    })

    rc = photon_cli._cmd_setup(_setup_args(new_project=True))

    assert rc == 0
    env_text = (home / ".env").read_text(encoding="utf-8")
    assert "PHOTON_PROJECT_ID=spectrum-1" in env_text
    assert "PHOTON_PROJECT_SECRET=secret-1" in env_text

    from hermes_cli.config import get_missing_env_vars

    missing = {entry["name"] for entry in get_missing_env_vars(required_only=False)}
    assert "PHOTON_PROJECT_ID" not in missing
    assert "PHOTON_PROJECT_SECRET" not in missing


def test_projects_select_stores_existing_project(monkeypatch: Any) -> None:
    stored: dict[str, Any] = {}
    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [{
        "id": "dash-1",
        "name": "Hermes Agent",
        "spectrum": True,
        "platforms": ["imessage"],
        "spectrumProjectId": "spectrum-1",
        "projectSecret": "secret-1",
    }])

    def fake_store(project_id: str, project_secret: str, **extra: Any) -> None:
        stored["project_id"] = project_id
        stored["project_secret"] = project_secret
        stored["extra"] = extra

    monkeypatch.setattr(photon_auth, "store_project_credentials", fake_store)
    args = argparse.Namespace(
        photon_projects_command="select",
        project_id="dash-1",
    )

    rc = photon_cli._cmd_projects(args)

    assert rc == 0
    assert stored["project_id"] == "spectrum-1"
    assert stored["extra"]["source"] == "manual-select"


def test_projects_select_fetches_missing_project_secret(monkeypatch: Any) -> None:
    stored: dict[str, Any] = {}
    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(photon_auth, "list_projects", lambda _token: [{
        "id": "dash-1",
        "name": "Hermes Agent",
        "spectrum": True,
        "platforms": ["imessage"],
        "spectrumProjectId": "spectrum-1",
    }])
    monkeypatch.setattr(photon_auth, "get_project", lambda _token, _project_id: {
        "id": "dash-1",
        "name": "Hermes Agent",
        "spectrum": True,
        "platforms": ["imessage"],
        "spectrumProjectId": "spectrum-1",
        "projectSecret": "secret-1",
    })

    def fake_store(project_id: str, project_secret: str, **extra: Any) -> None:
        stored["project_id"] = project_id
        stored["project_secret"] = project_secret
        stored["extra"] = extra

    monkeypatch.setattr(photon_auth, "store_project_credentials", fake_store)
    args = argparse.Namespace(
        photon_projects_command="select",
        project_id="dash-1",
    )

    rc = photon_cli._cmd_projects(args)

    assert rc == 0
    assert stored["project_id"] == "spectrum-1"
    assert stored["project_secret"] == "secret-1"
    assert stored["extra"]["source"] == "manual-select"


def test_webhook_register_noops_when_url_exists_and_secret_set(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("PHOTON_WEBHOOK_SECRET", "secret")
    monkeypatch.setattr(
        photon_auth,
        "load_project_credentials",
        lambda: ("spectrum-1", "project-secret"),
    )
    monkeypatch.setattr(photon_auth, "list_webhooks", lambda *_args: [{
        "id": "hook-1",
        "webhookUrl": "https://example.com/photon/webhook",
    }])
    monkeypatch.setattr(
        photon_auth,
        "register_webhook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("registered")),
    )
    monkeypatch.setattr(photon_cli, "_save_public_webhook_url", lambda _url: True)
    args = argparse.Namespace(
        photon_webhook_command="register",
        url="https://example.com/photon/webhook",
    )

    rc = photon_cli._cmd_webhook(args)

    assert rc == 0


def test_webhook_register_stops_when_url_exists_and_secret_missing(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("PHOTON_WEBHOOK_SECRET", raising=False)
    monkeypatch.setattr(
        photon_auth,
        "load_project_credentials",
        lambda: ("spectrum-1", "project-secret"),
    )
    monkeypatch.setattr(photon_auth, "list_webhooks", lambda *_args: [{
        "id": "hook-1",
        "webhookUrl": "https://example.com/photon/webhook",
    }])
    monkeypatch.setattr(
        photon_auth,
        "register_webhook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("registered")),
    )
    monkeypatch.setattr(
        photon_auth,
        "delete_webhook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("deleted")),
    )
    args = argparse.Namespace(
        photon_webhook_command="register",
        url="https://example.com/photon/webhook",
    )

    rc = photon_cli._cmd_webhook(args)

    assert rc == 1


def test_managed_tunnel_start_registers_webhook_and_saves_state(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "photon").mkdir()
    (home / "photon" / "tunnel.json").write_text(json.dumps({
        "managed": True,
        "pid": 999999,
        "public_url": "https://old.trycloudflare.com",
        "webhook_url": "https://old.trycloudflare.com/photon/webhook",
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(
        "PHOTON_WEBHOOK_PUBLIC_URL",
        "https://old.trycloudflare.com/photon/webhook",
    )
    monkeypatch.delenv("PHOTON_WEBHOOK_SECRET", raising=False)
    commands: list[list[str]] = []

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: Any) -> None:
            commands.append(cmd)
            self.pid = 4242
            self.returncode = None
            stdout = kwargs["stdout"]
            stdout.write("Your quick Tunnel has been created! https://fresh.trycloudflare.com\n")
            stdout.flush()

        def poll(self) -> None:
            return None

    deleted: list[str] = []
    registered: list[str] = []
    hooks = [
        {
            "id": "old-hook",
            "webhookUrl": "https://old.trycloudflare.com/photon/webhook",
        },
        {
            "id": "manual-hook",
            "webhookUrl": "https://example.com/photon/webhook",
        },
    ]
    monkeypatch.setattr(photon_tunnel.shutil, "which", lambda _name: "/usr/local/bin/cloudflared")
    monkeypatch.setattr(photon_tunnel.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: ("proj", "secret"))
    monkeypatch.setattr(photon_auth, "list_webhooks", lambda *_args: list(hooks))
    monkeypatch.setattr(
        photon_auth,
        "delete_webhook",
        lambda *_args, webhook_id: deleted.append(webhook_id),
    )

    def fake_register(_project_id: str, _project_secret: str, *, webhook_url: str) -> dict[str, str]:
        registered.append(webhook_url)
        return {"webhookUrl": webhook_url, "signingSecret": "SECRET123"}

    monkeypatch.setattr(photon_auth, "register_webhook", fake_register)

    rc = photon_cli._start_managed_tunnel_and_register()

    assert rc == 0
    assert commands == [[
        "/usr/local/bin/cloudflared",
        "tunnel",
        "--config",
        os.devnull,
        "--url",
        "http://127.0.0.1:8788",
        "--no-autoupdate",
    ]]
    assert deleted == ["old-hook"]
    assert registered == ["https://fresh.trycloudflare.com/photon/webhook"]
    state = json.loads((home / "photon" / "tunnel.json").read_text(encoding="utf-8"))
    assert state["public_url"] == "https://fresh.trycloudflare.com"
    assert state["webhook_url"] == "https://fresh.trycloudflare.com/photon/webhook"
    env_text = (home / ".env").read_text(encoding="utf-8")
    assert "PHOTON_WEBHOOK_SECRET=SECRET123" in env_text
    assert "PHOTON_WEBHOOK_PUBLIC_URL=https://fresh.trycloudflare.com/photon/webhook" in env_text


def test_managed_tunnel_does_not_delete_manual_trycloudflare_url(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(
        "PHOTON_WEBHOOK_PUBLIC_URL",
        "https://manual.trycloudflare.com/photon/webhook",
    )
    monkeypatch.delenv("PHOTON_WEBHOOK_SECRET", raising=False)
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: ("proj", "secret"))
    monkeypatch.setattr(photon_auth, "list_webhooks", lambda *_args: [{
        "id": "manual-hook",
        "webhookUrl": "https://manual.trycloudflare.com/photon/webhook",
    }])
    monkeypatch.setattr(
        photon_auth,
        "delete_webhook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("deleted")),
    )
    registered: list[str] = []
    monkeypatch.setattr(
        photon_tunnel,
        "start",
        lambda *_args, **_kwargs: photon_tunnel.TunnelStartResult(
            success=True,
            public_url="https://fresh.trycloudflare.com",
            webhook_url="https://fresh.trycloudflare.com/photon/webhook",
        ),
    )
    monkeypatch.setattr(
        photon_auth,
        "register_webhook",
        lambda *_args, webhook_url: registered.append(webhook_url) or {"signingSecret": "SECRET"},
    )

    rc = photon_cli._start_managed_tunnel_and_register()

    assert rc == 0
    assert registered == ["https://fresh.trycloudflare.com/photon/webhook"]


def test_managed_tunnel_auto_installs_cloudflared(
    tmp_path: Path, monkeypatch: Any, capsys: Any,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    commands: list[list[str]] = []
    registered: list[str] = []
    managed_binary = home / "bin" / "cloudflared"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("PHOTON_WEBHOOK_SECRET", raising=False)
    monkeypatch.setattr(photon_tunnel.shutil, "which", lambda _name: None)

    def fake_install(*, emit: Any = None) -> str:
        if emit:
            emit("  cloudflared not found — installing managed copy")
        return str(managed_binary)

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: Any) -> None:
            commands.append(cmd)
            self.pid = 4242
            self.returncode = None
            stdout = kwargs["stdout"]
            stdout.write("Your quick Tunnel has been created! https://fresh.trycloudflare.com\n")
            stdout.flush()

        def poll(self) -> None:
            return None

    monkeypatch.setattr(photon_tunnel, "install_managed_cloudflared", fake_install)
    monkeypatch.setattr(photon_tunnel.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: ("proj", "secret"))
    monkeypatch.setattr(photon_auth, "list_webhooks", lambda *_args: [])
    monkeypatch.setattr(
        photon_auth,
        "register_webhook",
        lambda *_args, webhook_url: registered.append(webhook_url) or {"signingSecret": "SECRET"},
    )

    rc = photon_cli._start_managed_tunnel_and_register()

    captured = capsys.readouterr()
    assert rc == 0
    assert "installing managed copy" in captured.out
    assert commands[0][0] == str(managed_binary)
    assert registered == ["https://fresh.trycloudflare.com/photon/webhook"]


def test_install_managed_cloudflared_verifies_release_digest(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    payload = b"cloudflared-binary"
    asset = photon_tunnel.CloudflaredAsset(
        name="cloudflared-linux-amd64",
        download_url="https://example.test/cloudflared",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        version="2026.5.2",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel, "_platform_asset_name", lambda: asset.name)
    monkeypatch.setattr(photon_tunnel, "_cloudflared_release_asset", lambda _name: asset)

    def fake_download(_url: str, destination: Path) -> None:
        destination.write_bytes(payload)

    monkeypatch.setattr(photon_tunnel, "_download_url", fake_download)

    installed = photon_tunnel.install_managed_cloudflared()

    assert installed == str(home / "bin" / "cloudflared")
    assert (home / "bin" / "cloudflared").read_bytes() == payload
    manifest = json.loads(
        (home / "bin" / "cloudflared.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["asset_sha256"] == asset.sha256
    assert manifest["binary_sha256"] == hashlib.sha256(payload).hexdigest()


def test_install_managed_cloudflared_reuses_matching_manifest(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    payload = b"installed-cloudflared"
    target = bin_dir / "cloudflared"
    target.write_bytes(payload)
    asset = photon_tunnel.CloudflaredAsset(
        name="cloudflared-darwin-arm64.tgz",
        download_url="https://example.test/cloudflared.tgz",
        sha256=hashlib.sha256(b"archive").hexdigest(),
        size=len(b"archive"),
        version="2026.5.2",
    )
    (bin_dir / "cloudflared.manifest.json").write_text(
        json.dumps({
            "asset": asset.name,
            "asset_sha256": asset.sha256,
            "binary_sha256": hashlib.sha256(payload).hexdigest(),
            "version": asset.version,
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel, "_platform_asset_name", lambda: asset.name)
    monkeypatch.setattr(photon_tunnel, "_cloudflared_release_asset", lambda _name: asset)
    monkeypatch.setattr(
        photon_tunnel,
        "_download_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("downloaded")),
    )

    installed = photon_tunnel.install_managed_cloudflared()

    assert installed == str(target)
    assert target.read_bytes() == payload


def test_install_managed_cloudflared_updates_stale_manifest(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    target = bin_dir / "cloudflared"
    target.write_bytes(b"old-cloudflared")
    new_payload = b"new-cloudflared"
    asset = photon_tunnel.CloudflaredAsset(
        name="cloudflared-linux-amd64",
        download_url="https://example.test/cloudflared",
        sha256=hashlib.sha256(new_payload).hexdigest(),
        size=len(new_payload),
        version="2026.5.2",
    )
    (bin_dir / "cloudflared.manifest.json").write_text(
        json.dumps({
            "asset": asset.name,
            "asset_sha256": hashlib.sha256(b"old-cloudflared").hexdigest(),
            "binary_sha256": hashlib.sha256(b"old-cloudflared").hexdigest(),
            "version": "2026.4.0",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel, "_platform_asset_name", lambda: asset.name)
    monkeypatch.setattr(photon_tunnel, "_cloudflared_release_asset", lambda _name: asset)

    def fake_download(_url: str, destination: Path) -> None:
        destination.write_bytes(new_payload)

    monkeypatch.setattr(photon_tunnel, "_download_url", fake_download)

    installed = photon_tunnel.install_managed_cloudflared()

    assert installed == str(target)
    assert target.read_bytes() == new_payload
    manifest = json.loads(
        (bin_dir / "cloudflared.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["asset_sha256"] == asset.sha256
    assert manifest["binary_sha256"] == hashlib.sha256(new_payload).hexdigest()


def test_install_managed_cloudflared_uses_existing_when_update_check_fails(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    target = bin_dir / "cloudflared"
    target.write_bytes(b"existing-cloudflared")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel, "_platform_asset_name", lambda: "cloudflared-linux-amd64")
    monkeypatch.setattr(
        photon_tunnel,
        "_cloudflared_release_asset",
        lambda _name: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    monkeypatch.setattr(
        photon_tunnel,
        "_download_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("downloaded")),
    )

    installed = photon_tunnel.install_managed_cloudflared()

    assert installed == str(target)
    assert target.read_bytes() == b"existing-cloudflared"


def test_install_managed_cloudflared_uses_existing_when_update_download_fails(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    target = bin_dir / "cloudflared"
    target.write_bytes(b"existing-cloudflared")
    asset = photon_tunnel.CloudflaredAsset(
        name="cloudflared-linux-amd64",
        download_url="https://example.test/cloudflared",
        sha256=hashlib.sha256(b"new-cloudflared").hexdigest(),
        size=len(b"new-cloudflared"),
        version="2026.5.2",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel, "_platform_asset_name", lambda: asset.name)
    monkeypatch.setattr(photon_tunnel, "_cloudflared_release_asset", lambda _name: asset)
    monkeypatch.setattr(
        photon_tunnel,
        "_download_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("download failed")),
    )

    installed = photon_tunnel.install_managed_cloudflared()

    assert installed == str(target)
    assert target.read_bytes() == b"existing-cloudflared"


def test_install_managed_cloudflared_fails_without_existing_when_update_check_fails(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel, "_platform_asset_name", lambda: "cloudflared-linux-amd64")
    monkeypatch.setattr(
        photon_tunnel,
        "_cloudflared_release_asset",
        lambda _name: (_ for _ in ()).throw(RuntimeError("network down")),
    )

    with pytest.raises(RuntimeError, match="network down"):
        photon_tunnel.install_managed_cloudflared()


def test_resolve_cloudflared_binary_updates_existing_managed_copy(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    managed_binary = bin_dir / "cloudflared"
    managed_binary.write_bytes(b"existing-cloudflared")
    installed: list[str] = []
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        photon_tunnel,
        "install_managed_cloudflared",
        lambda **_kwargs: installed.append("updated") or str(managed_binary),
    )

    resolved = photon_tunnel.resolve_cloudflared_binary()

    assert resolved == str(managed_binary)
    assert installed == ["updated"]
    assert (
        photon_tunnel.resolve_cloudflared_binary(auto_install=False)
        == str(managed_binary)
    )


def test_install_managed_cloudflared_rejects_bad_digest(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    asset = photon_tunnel.CloudflaredAsset(
        name="cloudflared-linux-amd64",
        download_url="https://example.test/cloudflared",
        sha256=hashlib.sha256(b"expected").hexdigest(),
        size=len(b"actual"),
        version="2026.5.2",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel, "_platform_asset_name", lambda: asset.name)
    monkeypatch.setattr(photon_tunnel, "_cloudflared_release_asset", lambda _name: asset)

    def fake_download(_url: str, destination: Path) -> None:
        destination.write_bytes(b"actual")

    monkeypatch.setattr(photon_tunnel, "_download_url", fake_download)

    try:
        photon_tunnel.install_managed_cloudflared()
    except RuntimeError as e:
        assert "checksum mismatch" in str(e)
    else:
        raise AssertionError("expected checksum failure")
    assert not (home / "bin" / "cloudflared").exists()


def test_cloudflared_downloads_use_ssl_context(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    sentinel_context = object()
    calls: list[Any] = []
    monkeypatch.setattr(photon_tunnel, "_ssl_context", lambda: sentinel_context)

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self._read = False

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self, *_args: Any) -> bytes:
            if self._read:
                return b""
            self._read = True
            return self.payload

    def fake_urlopen_json(request: Any, **kwargs: Any) -> FakeResponse:
        calls.append((request, kwargs))
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(photon_tunnel.urllib.request, "urlopen", fake_urlopen_json)

    assert photon_tunnel._urlopen_json("https://example.test/releases/latest") == {"ok": True}
    assert calls[-1][1]["context"] is sentinel_context

    def fake_urlopen_download(request: Any, **kwargs: Any) -> FakeResponse:
        calls.append((request, kwargs))
        return FakeResponse(b"cloudflared")

    monkeypatch.setattr(photon_tunnel.urllib.request, "urlopen", fake_urlopen_download)
    destination = tmp_path / "cloudflared"

    photon_tunnel._download_url("https://example.test/cloudflared", destination)

    assert destination.read_bytes() == b"cloudflared"
    assert calls[-1][1]["context"] is sentinel_context


def test_managed_tunnel_missing_cloudflared_is_actionable(
    tmp_path: Path, monkeypatch: Any, capsys: Any,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        photon_tunnel,
        "install_managed_cloudflared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("download failed")
        ),
    )
    monkeypatch.setattr(
        photon_auth,
        "register_webhook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("registered")),
    )

    rc = photon_cli._start_managed_tunnel_and_register()

    captured = capsys.readouterr()
    assert rc == 1
    assert "cloudflared install failed" in captured.err
    assert "brew install cloudflared" in captured.err
    assert "hermes photon webhook register" in captured.err


def test_status_next_step_selection(monkeypatch: Any) -> None:
    env: dict[str, str] = {}
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda key: env.get(key))
    allowed: list[str] = []
    monkeypatch.setattr(photon_auth, "load_allowed_phone_numbers", lambda: allowed)
    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: None)
    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: (None, None))
    assert photon_cli._next_status_step("✓ installed", {}) == "hermes photon login"

    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: "token")
    assert (
        photon_cli._next_status_step("✓ installed", {})
        == "hermes photon quick-setup --phone '+<country-code><number>'"
    )

    monkeypatch.setattr(photon_auth, "load_project_credentials", lambda: ("proj", "secret"))
    assert (
        photon_cli._next_status_step("✗ run `hermes photon install-sidecar`", {})
        == "hermes photon install-sidecar"
    )

    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: None)
    assert (
        photon_cli._next_status_step("✗ run `hermes photon install-sidecar`", {})
        == "hermes photon install-sidecar"
    )
    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: "token")

    assert photon_cli._next_status_step("✓ installed", {}) == "hermes photon webhook tunnel start"

    env["PHOTON_WEBHOOK_SECRET"] = "secret"
    env["PHOTON_WEBHOOK_PUBLIC_URL"] = "https://managed.trycloudflare.com/photon/webhook"
    assert (
        photon_cli._next_status_step("✓ installed", {"running": False})
        == "hermes photon webhook tunnel start"
    )

    env["PHOTON_WEBHOOK_PUBLIC_URL"] = "https://example.com/photon/webhook"
    assert (
        photon_cli._next_status_step("✓ installed", {})
        == "hermes photon allow-phone '+<country-code><number>'"
    )

    allowed.append("+15551234567")
    assert "hermes gateway run -v" in photon_cli._next_status_step("✓ installed", {})


def test_adapter_registers_gateway_setup_fn() -> None:
    captured: dict[str, Any] = {}

    class Ctx:
        def register_platform(self, **kwargs: Any) -> None:
            captured["platform"] = kwargs

        def register_cli_command(self, **kwargs: Any) -> None:
            captured["cli"] = kwargs

    photon_adapter.register(Ctx())

    assert captured["platform"]["setup_fn"] is photon_cli.interactive_setup
    assert captured["cli"]["setup_fn"] is photon_cli.register_cli


def test_interactive_setup_prints_incomplete_guidance(
    monkeypatch: Any, capsys: Any,
) -> None:
    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(
        photon_auth,
        "load_project_credentials",
        lambda: (None, None),
    )
    monkeypatch.setattr(photon_cli, "_cmd_quick_setup", lambda _args: 1)

    photon_cli.interactive_setup()

    out = capsys.readouterr().out
    assert "Photon iMessage setup is not complete yet" in out
    assert "hermes photon quick-setup --phone '+<country-code><number>'" in out
    assert "hermes photon status" in out


def test_sidecar_dependency_status_missing_node_modules(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    monkeypatch.setattr(photon_cli, "_SIDECAR_DIR", tmp_path / "sidecar")

    status = photon_cli._sidecar_dependency_status()

    assert "hermes photon install-sidecar" in status


def test_sidecar_dependency_status_rejects_old_spectrum_ts(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    _with_node_modules(tmp_path, monkeypatch)

    def fake_run(*_args: Any, **_kwargs: Any) -> _Proc:
        return _Proc(
            returncode=0,
            stdout=json.dumps({
                "dependencies": {
                    "spectrum-ts": {"version": "0.1.2"},
                },
            }),
        )

    monkeypatch.setattr(photon_cli.subprocess, "run", fake_run)

    status = photon_cli._sidecar_dependency_status()

    assert "spectrum-ts 0.1.2 is too old" in status


def test_sidecar_dependency_status_surfaces_npm_problems(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    _with_node_modules(tmp_path, monkeypatch)

    def fake_run(*_args: Any, **_kwargs: Any) -> _Proc:
        return _Proc(
            returncode=1,
            stdout=json.dumps({
                "dependencies": {
                    "spectrum-ts": {"version": "1.7.2"},
                },
                "problems": ["invalid: spectrum-ts@1.7.2 from the root project"],
            }),
        )

    monkeypatch.setattr(photon_cli.subprocess, "run", fake_run)

    status = photon_cli._sidecar_dependency_status()

    assert "npm reports invalid: spectrum-ts@1.7.2" in status


def test_sidecar_dependency_status_accepts_current_spectrum_ts(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    _with_node_modules(tmp_path, monkeypatch)

    def fake_run(*_args: Any, **_kwargs: Any) -> _Proc:
        return _Proc(
            returncode=0,
            stdout=json.dumps({
                "dependencies": {
                    "spectrum-ts": {"version": "1.7.2"},
                },
            }),
        )

    monkeypatch.setattr(photon_cli.subprocess, "run", fake_run)

    status = photon_cli._sidecar_dependency_status()

    assert status == "✓ installed (spectrum-ts 1.7.2)"
