"""Tests for Photon CLI helpers."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

from plugins.platforms.photon import cli as photon_cli
from plugins.platforms.photon import auth as photon_auth


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
