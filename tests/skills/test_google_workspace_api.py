"""Tests for Google Workspace gws bridge and CLI wrapper."""

import importlib.util
import json
import os
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


BRIDGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/gws_bridge.py"
)
API_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)


@pytest.fixture
def bridge_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("gws_bridge_test", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("gws_api_test", API_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # Ensure the gws CLI code path is taken even when the binary isn't
    # installed (CI).  Without this, calendar_list() falls through to the
    # Python SDK path which imports ``googleapiclient`` — not in deps.
    module._gws_binary = lambda: "/usr/bin/gws"
    # Bypass authentication check — no real token file in CI.
    module._ensure_authenticated = lambda: None
    return module


def _write_token(path: Path, *, token="ya29.test", expiry=None, **extra):
    data = {
        "token": token,
        "refresh_token": "1//refresh",
        "client_id": "123.apps.googleusercontent.com",
        "client_secret": "secret",
        "token_uri": "https://oauth2.googleapis.com/token",
        **extra,
    }
    if expiry is not None:
        data["expiry"] = expiry
    path.write_text(json.dumps(data))


def test_bridge_returns_valid_token(bridge_module, tmp_path):
    """Non-expired token is returned without refresh."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.valid", expiry=future)

    result = bridge_module.get_valid_token()
    assert result == "ya29.valid"


def test_bridge_refreshes_expired_token(bridge_module, tmp_path):
    """Expired token triggers a refresh via token_uri."""
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.old", expiry=past)

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "access_token": "ya29.refreshed",
        "expires_in": 3600,
    }).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = bridge_module.get_valid_token()

    assert result == "ya29.refreshed"
    # Verify persisted
    saved = json.loads(token_path.read_text())
    assert saved["token"] == "ya29.refreshed"
    assert saved["type"] == "authorized_user"


def test_bridge_exits_on_missing_token(bridge_module):
    """Missing token file causes exit with code 1."""
    with pytest.raises(SystemExit):
        bridge_module.get_valid_token()


def test_bridge_main_injects_token_env(bridge_module, tmp_path):
    """main() sets GOOGLE_WORKSPACE_CLI_TOKEN in subprocess env."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.injected", expiry=future)

    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return MagicMock(returncode=0)

    with patch.object(sys, "argv", ["gws_bridge.py", "gmail", "+triage"]):
        with patch.object(subprocess, "run", side_effect=capture_run):
            with pytest.raises(SystemExit):
                bridge_module.main()

    assert captured["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "ya29.injected"
    assert captured["cmd"] == ["gws", "gmail", "+triage"]


def test_api_calendar_list_uses_events_list(api_module):
    """calendar_list calls _run_gws with events list + params."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="{}", stderr="")

    args = api_module.argparse.Namespace(
        start="", end="", max=25, calendar="primary", func=api_module.calendar_list,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.calendar_list(args)

    cmd = captured["cmd"]
    # _gws_binary() returns "/usr/bin/gws", so cmd[0] is that binary
    assert cmd[0] == "/usr/bin/gws"
    assert "calendar" in cmd
    assert "events" in cmd
    assert "list" in cmd
    assert "--params" in cmd
    params = json.loads(cmd[cmd.index("--params") + 1])
    assert "timeMin" in params
    assert "timeMax" in params
    assert params["calendarId"] == "primary"


def test_api_calendar_list_respects_date_range(api_module):
    """calendar list with --start/--end passes correct time bounds."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="{}", stderr="")

    args = api_module.argparse.Namespace(
        start="2026-04-01T00:00:00Z",
        end="2026-04-07T23:59:59Z",
        max=25,
        calendar="primary",
        func=api_module.calendar_list,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.calendar_list(args)

    cmd = captured["cmd"]
    params_idx = cmd.index("--params")
    params = json.loads(cmd[params_idx + 1])
    assert params["timeMin"] == "2026-04-01T00:00:00Z"
    assert params["timeMax"] == "2026-04-07T23:59:59Z"


def test_api_get_credentials_refresh_persists_authorized_user_type(api_module, monkeypatch):
    token_path = api_module.TOKEN_PATH
    _write_token(token_path, token="ya29.old")

    class FakeCredentials:
        def __init__(self):
            self.expired = True
            self.refresh_token = "1//refresh"
            self.valid = True

        def refresh(self, request):
            self.expired = False

        def to_json(self):
            return json.dumps({
                "token": "ya29.refreshed",
                "refresh_token": "1//refresh",
                "client_id": "123.apps.googleusercontent.com",
                "client_secret": "secret",
                "token_uri": "https://oauth2.googleapis.com/token",
            })

    class FakeCredentialsModule:
        @staticmethod
        def from_authorized_user_file(filename, scopes):
            assert filename == str(token_path)
            assert scopes == api_module.SCOPES
            return FakeCredentials()

    google_module = types.ModuleType("google")
    oauth2_module = types.ModuleType("google.oauth2")
    credentials_module = types.ModuleType("google.oauth2.credentials")
    credentials_module.Credentials = FakeCredentialsModule
    transport_module = types.ModuleType("google.auth.transport")
    requests_module = types.ModuleType("google.auth.transport.requests")
    requests_module.Request = lambda: object()

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_module)

    creds = api_module.get_credentials()

    saved = json.loads(token_path.read_text())
    assert isinstance(creds, FakeCredentials)
    assert saved["token"] == "ya29.refreshed"
    assert saved["type"] == "authorized_user"


def test_drive_download_path_rejects_unsafe_output(api_module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))

    with pytest.raises(ValueError):
        api_module._safe_drive_download_path("../outside.txt", "ignored.txt")
    with pytest.raises(ValueError):
        api_module._safe_drive_download_path(str(tmp_path / "outside.txt"), "ignored.txt")


def test_drive_download_path_rejects_directory_output(api_module, tmp_path, monkeypatch):
    """_safe_drive_download_path rejects output that resolves to a directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))

    # '.' has an empty basename (Path('.').name == '')
    with pytest.raises(ValueError):
        api_module._safe_drive_download_path(".", "filename.txt")

    # existing directory as the target
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    with pytest.raises(ValueError):
        api_module._safe_drive_download_path("existing", "filename.txt")


def test_drive_download_path_sanitizes_remote_name_and_enforces_safe_root(api_module, tmp_path, monkeypatch):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(safe_root)
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

    out_path = api_module._safe_drive_download_path("", "../outside/payload")
    assert out_path == safe_root / "payload"

    escape_link = safe_root / "escape"
    escape_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        api_module._safe_drive_download_path("escape/file.txt", "ignored.txt")


def test_drive_download_writes_sanitized_remote_name_inside_cwd(api_module, tmp_path, monkeypatch):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    outside_target = tmp_path / "outside.txt"
    monkeypatch.chdir(safe_root)
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

    class FakeRequest:
        pass

    class FakeFiles:
        def get(self, **kwargs):
            assert kwargs["fileId"] == "file-1"
            return self

        def execute(self):
            return {"id": "file-1", "name": "../outside.txt", "mimeType": "text/plain"}

        def get_media(self, **kwargs):
            assert kwargs["fileId"] == "file-1"
            return FakeRequest()

    class FakeService:
        def files(self):
            return FakeFiles()

    class FakeDownloader:
        def __init__(self, fh, request):
            self.fh = fh
            self.done = False

        def next_chunk(self):
            if not self.done:
                self.fh.write(b"drive bytes")
                self.done = True
            return None, True

    googleapiclient_module = types.ModuleType("googleapiclient")
    http_module = types.ModuleType("googleapiclient.http")
    http_module.MediaIoBaseDownload = FakeDownloader
    monkeypatch.setitem(sys.modules, "googleapiclient", googleapiclient_module)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http_module)
    monkeypatch.setattr(api_module, "build_service", lambda *_args: FakeService())

    args = api_module.argparse.Namespace(file_id="file-1", output="", export_mime="")
    api_module.drive_download(args)

    assert (safe_root / "outside.txt").read_bytes() == b"drive bytes"
    assert not outside_target.exists()


def test_open_drive_download_destination_rejects_symlink_parent(api_module, tmp_path):
    """_open_drive_download_destination rejects symlinks in parent components."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    target = tmp_path / "target"
    target.mkdir()

    link = cwd / "link"
    link.symlink_to(target)

    path = cwd / "link" / "file.txt"
    with pytest.raises(ValueError, match="symlink|non-directory"):
        api_module._open_drive_download_destination(path, cwd)


def test_open_drive_download_destination_creates_real_parents(api_module, tmp_path):
    """_open_drive_download_destination creates and writes through real parent dirs."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    path = cwd / "real" / "nested" / "file.txt"
    with api_module._open_drive_download_destination(path, cwd) as fh:
        fh.write(b"payload")

    assert path.read_bytes() == b"payload"
    assert path.parent.is_dir()


def test_open_drive_download_destination_rejects_final_symlink(api_module, tmp_path):
    """_open_drive_download_destination rejects a symlink as the final file."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")

    link = cwd / "file.txt"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|non-directory"):
        api_module._open_drive_download_destination(link, cwd)

    assert outside.read_text() == "outside"


def test_open_drive_download_destination_windows_fallback_writes_file(
    api_module, tmp_path, monkeypatch
):
    """On Windows, fallback open path still creates parent dirs and writes bytes."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    out_path = cwd / "nested" / "file.bin"

    monkeypatch.setattr(api_module.sys, "platform", "win32")

    with api_module._open_drive_download_destination(out_path, cwd) as fh:
        fh.write(b"payload")

    assert out_path.read_bytes() == b"payload"


def test_drive_download_rejects_symlink_parent_after_mkdir(api_module, tmp_path, monkeypatch):
    """drive_download refuses to write when a parent directory is a symlink (TOCTOU guard).

    Simulates the race: _safe_drive_download_path validated the path before
    the symlink existed; by the time drive_download calls mkdir the parent
    has been swapped for a symlink.  The post-mkdir guard must detect this.
    """
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    monkeypatch.chdir(safe_root)
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

    # The symlink is the attacker-inserted swap: safe_root/linkdir → target_dir.
    link_dir = safe_root / "linkdir"
    link_dir.symlink_to(target_dir)

    # Bypass _safe_drive_download_path to supply the pre-swap path (as it would
    # have been returned before the symlink was created).
    spoofed_out_path = safe_root / "linkdir" / "payload.bin"
    monkeypatch.setattr(
        api_module, "_safe_drive_download_path", lambda *args, **kwargs: spoofed_out_path
    )

    class FakeFiles:
        def get(self, **kwargs):
            return self

        def execute(self):
            return {"id": "file-1", "name": "payload.bin", "mimeType": "application/octet-stream"}

        def get_media(self, **kwargs):
            return object()

    class FakeService:
        def files(self):
            return FakeFiles()

    googleapiclient_module = types.ModuleType("googleapiclient")
    http_module = types.ModuleType("googleapiclient.http")
    http_module.MediaIoBaseDownload = MagicMock()
    monkeypatch.setitem(sys.modules, "googleapiclient", googleapiclient_module)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http_module)
    monkeypatch.setattr(api_module, "build_service", lambda *_args: FakeService())

    args = api_module.argparse.Namespace(file_id="file-1", output="linkdir/payload.bin", export_mime="")
    with pytest.raises(SystemExit):
        api_module.drive_download(args)
