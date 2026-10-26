"""Mission-control (computer view) router tests.

Standalone mount: the surface router includes no auth of its own — web_server's
auth middleware is app-level and inherited by the mount (web/ AGENTS.md), so
these tests exercise surface logic against a temp HERMES_HOME.
"""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def home(tmp_path, monkeypatch):
    home = tmp_path / "h"
    home.mkdir()
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    yield home
    reset_hermes_home_override(token)


@pytest.fixture()
def client():
    from hermes_cli.web_routers.computer import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _enable(monkeypatch, enabled=True):
    from hermes_cli import config

    monkeypatch.setattr(
        config, "load_config", lambda: {"dashboard": {"computer_view": enabled}}
    )


def test_disabled_by_default(client, monkeypatch, home):
    _enable(monkeypatch, False)
    assert client.get("/api/computer/status").json() == {
        "enabled": False, "last_capture_at": None, "count": 0,
    }
    r = client.get("/api/computer/screenshot")
    assert r.status_code == 409
    assert "dashboard.computer_view" in r.json()["detail"]


def test_capture_and_rate_cap(client, monkeypatch, home):
    _enable(monkeypatch)
    from hermes_cli.web_routers import computer

    calls = []

    def fake_run(cmd, **kw):
        Path(cmd[1]).write_bytes(b"\x89PNGFAKE")
        calls.append(cmd)
        import subprocess as sp

        return sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(computer, "subprocess_run", fake_run)
    r1 = client.get("/api/computer/screenshot")
    assert r1.status_code == 200
    assert r1.headers["content-type"].startswith("image/png")
    r2 = client.get("/api/computer/screenshot")  # within rate window: no second shell-out
    assert r2.status_code == 200
    assert len(calls) == 1

    st = client.get("/api/computer/status").json()
    assert st["enabled"] is True and st["count"] == 1


def test_capture_failure_503(client, monkeypatch, home):
    _enable(monkeypatch)
    from hermes_cli.web_routers import computer
    import subprocess as sp

    def fake_run(cmd, **kw):
        return sp.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(computer, "subprocess_run", fake_run)
    r = client.get("/api/computer/screenshot")
    assert r.status_code == 503
    assert "boom" in r.json()["detail"]
    # the failed capture file must not linger in the inventory
    assert client.get("/api/computer/status").json()["count"] == 0


def test_retention_prune(client, monkeypatch, home):
    _enable(monkeypatch)
    from hermes_cli.web_routers import computer
    import subprocess as sp
    import time as _time

    cap = home / "screenshots"
    cap.mkdir(parents=True)
    # Backdate the seeds: freshly written mtimes would hit the 3s rate cap and
    # the ticker would (correctly) re-serve the newest one without capturing.
    for i in range(computer._MAX_KEEP + 5):
        f = cap / f"screen-old{i}.png"
        f.write_bytes(b"\x89PNGOLD" + bytes([i]))
        old = _time.time() - 3600
        os.utime(f, (old, old))

    def fake_run(cmd, **kw):
        Path(cmd[1]).write_bytes(b"\x89PNGNEW")
        return sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(computer, "subprocess_run", fake_run)
    assert client.get("/api/computer/screenshot").status_code == 200
    assert client.get("/api/computer/status").json()["count"] <= computer._MAX_KEEP


def test_missing_home_fails_closed(client, monkeypatch, tmp_path):
    _enable(monkeypatch)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(tmp_path / "missing-home"))
    try:
        # A genuinely absent home: the capture dir must refuse, never create one.
        r = client.get("/api/computer/screenshot")
        assert r.status_code == 503
        assert "refusing" in r.json()["detail"]
        assert not (tmp_path / "missing-home").exists()
    finally:
        reset_hermes_home_override(token)


def test_config_default_is_off():
    """Invariant for the new DEFAULT_CONFIG key: the merged default is False."""
    import tempfile

    from hermes_cli.config import load_config

    with tempfile.TemporaryDirectory() as td:
        old = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = td
        try:
            assert load_config()["dashboard"]["computer_view"] is False
        finally:
            if old is not None:
                os.environ["HERMES_HOME"] = old
            else:
                os.environ.pop("HERMES_HOME", None)