"""Tests for Photon CLI helpers."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
from typing import Any

from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon import cli as photon_cli
from plugins.platforms.photon import auth as photon_auth
from plugins.platforms.photon import tunnel as photon_tunnel


class _Proc:
    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


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


def test_quick_setup_requires_login_first(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(photon_auth, "load_photon_token", lambda: None)
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
        lambda: photon_tunnel.TunnelStartResult(
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


def test_managed_tunnel_missing_cloudflared_is_actionable(
    tmp_path: Path, monkeypatch: Any, capsys: Any,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(photon_tunnel.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        photon_auth,
        "register_webhook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("registered")),
    )

    rc = photon_cli._start_managed_tunnel_and_register()

    captured = capsys.readouterr()
    assert rc == 1
    assert "brew install cloudflared" in captured.err
    assert "hermes photon webhook register" in captured.err


def test_status_next_step_selection(monkeypatch: Any) -> None:
    env: dict[str, str] = {}
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda key: env.get(key))
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

    assert photon_cli._next_status_step("✓ installed", {}) == "hermes photon webhook tunnel start"

    env["PHOTON_WEBHOOK_SECRET"] = "secret"
    env["PHOTON_WEBHOOK_PUBLIC_URL"] = "https://managed.trycloudflare.com/photon/webhook"
    assert (
        photon_cli._next_status_step("✓ installed", {"running": False})
        == "hermes photon webhook tunnel start"
    )

    env["PHOTON_WEBHOOK_PUBLIC_URL"] = "https://example.com/photon/webhook"
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
