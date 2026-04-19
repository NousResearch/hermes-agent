from __future__ import annotations

from pathlib import Path

import pytest

from integrations import zectrix_note_mcp as zectrix


def test_join_url_and_unwrap_response():
    assert zectrix._join_url("/open/v1/todos") == "https://cloud.zectrix.com/open/v1/todos"
    assert zectrix._unwrap_response({"code": 0, "data": {"id": 1}}) == {"id": 1}


def test_unwrap_response_raises_on_api_error():
    with pytest.raises(RuntimeError, match="Zectrix API error 401"):
        zectrix._unwrap_response({"code": 401, "msg": "unauthorized"})


def test_resolve_device_id_prefers_explicit_or_default(monkeypatch):
    monkeypatch.setattr(zectrix, "DEFAULT_DEVICE_ID", "AA:BB:CC:DD:EE:FF")
    monkeypatch.setattr(zectrix, "_list_devices_raw", lambda: [])
    assert zectrix._resolve_device_id(None) == "AA:BB:CC:DD:EE:FF"
    assert zectrix._resolve_device_id(" 11:22:33 ") == "11:22:33"


def test_resolve_device_id_uses_single_discovered_device(monkeypatch):
    monkeypatch.setattr(zectrix, "DEFAULT_DEVICE_ID", None)
    monkeypatch.setattr(
        zectrix,
        "_list_devices_raw",
        lambda: [{"deviceId": "AA:BB:CC:DD:EE:FF", "alias": "one"}],
    )
    assert zectrix._resolve_device_id(None) == "AA:BB:CC:DD:EE:FF"


def test_resolve_device_id_requires_choice_for_multiple_devices(monkeypatch):
    monkeypatch.setattr(zectrix, "DEFAULT_DEVICE_ID", None)
    monkeypatch.setattr(
        zectrix,
        "_list_devices_raw",
        lambda: [
            {"deviceId": "AA:BB:CC:DD:EE:FF"},
            {"deviceId": "11:22:33:44:55:66"},
        ],
    )
    with pytest.raises(RuntimeError, match="multiple devices exist"):
        zectrix._resolve_device_id(None)


def test_multipart_images_builds_files(tmp_path: Path):
    img = tmp_path / "note.png"
    img.write_bytes(b"fakepng")
    files = zectrix._multipart_images([str(img)])
    assert files[0][0] == "images"
    assert files[0][1][0] == "note.png"
    assert files[0][1][1] == b"fakepng"


def test_api_request_rejects_without_key(monkeypatch):
    monkeypatch.setattr(zectrix, "API_KEY", "")
    with pytest.raises(RuntimeError, match="ZECTRIX_API_KEY is not set"):
        zectrix._headers()
