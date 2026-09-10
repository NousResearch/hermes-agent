from __future__ import annotations

import hashlib
import io
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "scripts" / "desktop-update" / "appimage_update.py"
SPEC = importlib.util.spec_from_file_location("appimage_update", MODULE_PATH)
assert SPEC and SPEC.loader
appimage_update = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(appimage_update)


class Response:
    def __init__(self, body: bytes):
        self.body = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1):
        return self.body.read(size)


def release_payload(binary: bytes, *, digest: str | None = None) -> tuple[dict, dict]:
    asset = {
        "name": "Hermes-0.20.6-linux-x64.AppImage",
        "browser_download_url": "https://downloads.example/hermes.AppImage",
    }
    if digest is not None:
        asset["digest"] = digest
    release = {"tag_name": "v2026.8.24", "draft": False, "prerelease": False, "assets": [asset]}
    return release, asset


def test_appimage_arch_maps_supported_machines():
    assert appimage_update.appimage_arch("x86_64") == "x64"
    assert appimage_update.appimage_arch("aarch64") == "arm64"
    assert appimage_update.appimage_arch("armv7l") == "armv7l"


def test_appimage_arch_rejects_unknown_machine():
    with pytest.raises(appimage_update.AppImageUpdateError, match="Unsupported"):
        appimage_update.appimage_arch("mips64")


def test_update_appimage_verifies_digest_and_replaces_atomically(tmp_path: Path):
    old = b"old appimage"
    new = b"new appimage"
    target = tmp_path / "Hermes.AppImage"
    target.write_bytes(old)
    target.chmod(0o755)
    digest = hashlib.sha256(new).hexdigest()
    release, _ = release_payload(new, digest=f"sha256:{digest}")
    responses = {
        "https://api.example/releases/latest": Response(json.dumps(release).encode()),
        "https://downloads.example/hermes.AppImage": Response(new),
    }

    def opener(request, timeout):
        assert timeout in (30, 120)
        return responses[request.full_url]

    assert (
        appimage_update.update_appimage(
            target, arch="x64", release_api="https://api.example/releases/latest", opener=opener
        )
        == "v2026.8.24"
    )
    assert target.read_bytes() == new
    assert target.stat().st_mode & 0o111
    assert list(tmp_path.glob("*.download")) == []


def test_update_appimage_does_not_need_sidecar_when_release_has_digest(tmp_path: Path):
    new = b"new appimage"
    digest = hashlib.sha256(new).hexdigest()
    release, _ = release_payload(new, digest=f"sha256:{digest}")
    release["assets"].append(
        {
            "name": "SHA256SUMS.txt",
            "browser_download_url": "https://downloads.example/SHA256SUMS.txt",
        }
    )
    target = tmp_path / "Hermes.AppImage"
    target.write_bytes(b"old")

    def opener(request, timeout):
        if request.full_url.endswith("latest"):
            return Response(json.dumps(release).encode())
        if request.full_url.endswith("hermes.AppImage"):
            return Response(new)
        raise AssertionError("release digest should avoid the checksum sidecar")

    appimage_update.update_appimage(
        target, arch="x64", release_api="https://api.example/releases/latest", opener=opener
    )
    assert target.read_bytes() == new


def test_update_appimage_checksum_mismatch_keeps_existing_file(tmp_path: Path):
    target = tmp_path / "Hermes.AppImage"
    target.write_bytes(b"known-good")
    release, _ = release_payload(b"bad", digest=f"sha256:{hashlib.sha256(b'expected').hexdigest()}")
    responses = {
        "https://api.example/releases/latest": Response(json.dumps(release).encode()),
        "https://downloads.example/hermes.AppImage": Response(b"bad"),
    }

    def opener(request, timeout):
        return responses[request.full_url]

    with pytest.raises(appimage_update.AppImageUpdateError, match="checksum mismatch"):
        appimage_update.update_appimage(
            target, arch="x64", release_api="https://api.example/releases/latest", opener=opener
        )
    assert target.read_bytes() == b"known-good"
    assert list(tmp_path.glob("*.download")) == []


def test_update_appimage_accepts_checksum_sidecar_for_older_release(tmp_path: Path):
    new = b"new appimage"
    digest = hashlib.sha256(new).hexdigest()
    release, asset = release_payload(new)
    release["assets"].append(
        {
            "name": "SHA256SUMS.txt",
            "browser_download_url": "https://downloads.example/SHA256SUMS.txt",
        }
    )
    target = tmp_path / "Hermes.AppImage"
    target.write_bytes(b"old")
    responses = {
        "https://api.example/releases/latest": Response(json.dumps(release).encode()),
        "https://downloads.example/SHA256SUMS.txt": Response(f"{digest}  {asset['name']}\n".encode()),
        "https://downloads.example/hermes.AppImage": Response(new),
    }

    def opener(request, timeout):
        return responses[request.full_url]

    appimage_update.update_appimage(
        target, arch="x64", release_api="https://api.example/releases/latest", opener=opener
    )
    assert target.read_bytes() == new


def test_update_appimage_rejects_ambiguous_checksum_sidecar(tmp_path: Path):
    new = b"new appimage"
    digest = hashlib.sha256(new).hexdigest()
    release, asset = release_payload(new)
    release["assets"].append(
        {
            "name": "SHA256SUMS.txt",
            "browser_download_url": "https://downloads.example/SHA256SUMS.txt",
        }
    )
    target = tmp_path / "Hermes.AppImage"
    target.write_bytes(b"old")
    responses = {
        "https://api.example/releases/latest": Response(json.dumps(release).encode()),
        "https://downloads.example/SHA256SUMS.txt": Response(
            f"{digest}  {asset['name']}\n{digest}  {asset['name']}\n".encode()
        ),
    }

    def opener(request, timeout):
        return responses[request.full_url]

    with pytest.raises(appimage_update.AppImageUpdateError, match="multiple entries"):
        appimage_update.update_appimage(
            target, arch="x64", release_api="https://api.example/releases/latest", opener=opener
        )
    assert target.read_bytes() == b"old"


def test_update_appimage_updates_symlink_target_without_replacing_link(tmp_path: Path):
    real = tmp_path / "real.AppImage"
    real.write_bytes(b"old")
    link = tmp_path / "Hermes.AppImage"
    link.symlink_to(real)
    release, _ = release_payload(b"new", digest=f"sha256:{hashlib.sha256(b'new').hexdigest()}")

    def opener(request, timeout):
        if request.full_url.endswith("latest"):
            return Response(json.dumps(release).encode())
        return Response(b"new")

    appimage_update.update_appimage(
        link, arch="x64", release_api="https://api.example/releases/latest", opener=opener
    )
    assert link.is_symlink()
    assert real.read_bytes() == b"new"
