"""Regression tests for #57413 — desktop attachment paths must use HERMES_HOME."""

from __future__ import annotations

import pytest

from hermes_constants import (
    format_desktop_attachment_ref,
    get_desktop_attachments_dir,
    resolve_desktop_attachment_ref,
)

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def client(monkeypatch):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous_auth_required


def test_get_desktop_attachments_dir_uses_hermes_home(monkeypatch, tmp_path):
    home = tmp_path / "data"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert get_desktop_attachments_dir() == home / "desktop-attachments"


def test_resolve_desktop_attachment_ref_maps_under_hermes_home(monkeypatch, tmp_path):
    home = tmp_path / "data"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    resolved = resolve_desktop_attachment_ref(".hermes/desktop-attachments/report.txt")
    assert resolved == home / "desktop-attachments" / "report.txt"
    assert resolve_desktop_attachment_ref("../secrets") is None


def test_format_desktop_attachment_ref_round_trip(monkeypatch, tmp_path):
    home = tmp_path / "data"
    target = home / "desktop-attachments" / "report.txt"
    target.parent.mkdir(parents=True)
    target.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref = format_desktop_attachment_ref(target)
    assert ref == ".hermes/desktop-attachments/report.txt"
    assert resolve_desktop_attachment_ref(ref) == target.resolve()


def test_fs_read_text_resolves_desktop_attachment_alias(client, tmp_path, monkeypatch):
    """GET /api/fs/read-text must not join .hermes/desktop-attachments against cwd."""
    data_home = tmp_path / "data"
    data_home.mkdir()
    install_cwd = tmp_path / "opt" / "hermes"
    install_cwd.mkdir(parents=True)
    attachment = data_home / "desktop-attachments" / "report.txt"
    attachment.parent.mkdir(parents=True)
    attachment.write_text("hello", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(data_home))
    monkeypatch.chdir(install_cwd)

    resp = client.get(
        "/api/fs/read-text",
        params={"path": ".hermes/desktop-attachments/report.txt"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "hello"
