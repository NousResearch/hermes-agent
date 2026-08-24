#!/usr/bin/env python3
"""Download and atomically replace the latest Hermes AppImage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_RELEASE_API = "https://api.github.com/repos/NousResearch/hermes-agent/releases/latest"
USER_AGENT = "Hermes-AppImage-Updater"
CHUNK_SIZE = 1024 * 1024
SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")


class AppImageUpdateError(RuntimeError):
    """Raised when an AppImage update cannot be completed safely."""


def appimage_arch(machine: str | None = None) -> str:
    """Map the host architecture to electron-builder's artifact suffix."""
    value = (machine or platform.machine()).lower()
    if value in {"x86_64", "amd64"}:
        return "x64"
    if value in {"aarch64", "arm64"}:
        return "arm64"
    if value in {"armv7l", "armv7", "armhf"}:
        return "armv7l"
    raise AppImageUpdateError(f"Unsupported Linux architecture: {value or 'unknown'}")


def _request(url: str) -> Request:
    return Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
        },
    )


def _read_json(url: str, opener: Callable[..., object] = urlopen) -> dict:
    try:
        with opener(_request(url), timeout=30) as response:  # type: ignore[union-attr]
            payload = json.load(response)  # type: ignore[arg-type]
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise AppImageUpdateError(f"Could not check for the latest AppImage: {exc}") from exc
    if not isinstance(payload, dict):
        raise AppImageUpdateError("The release service returned an invalid response")
    return payload


def _asset_for_release(release: dict, arch: str) -> dict:
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise AppImageUpdateError("The latest release has no downloadable assets")

    candidates = []
    arch_token = f"linux-{arch}".lower()
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        if name.lower().endswith(".appimage") and arch_token in name.lower():
            candidates.append(asset)
    if len(candidates) != 1:
        names = ", ".join(str(a.get("name") or "") for a in candidates)
        detail = f" ({names})" if names else ""
        raise AppImageUpdateError(f"Could not uniquely identify the {arch} AppImage{detail}")
    asset = candidates[0]
    if not asset.get("browser_download_url"):
        raise AppImageUpdateError("The latest AppImage has no download URL")
    return asset


def _checksum_from_release(release: dict, asset: dict, opener: Callable[..., object]) -> str:
    digest = str(asset.get("digest") or "")
    if digest.lower().startswith("sha256:"):
        expected = digest.split(":", 1)[1].strip().lower()
        if SHA256_RE.fullmatch(expected):
            return expected

    # Older releases may omit `digest`; only accept a checksum from this release.
    assets = release.get("assets") or []
    asset_name = str(asset.get("name") or "")
    sidecars = [
        candidate
        for candidate in assets
        if isinstance(candidate, dict)
        and str(candidate.get("name") or "").lower()
        in {f"{asset_name}.sha256".lower(), "sha256sums.txt", "checksums.txt"}
    ]
    for sidecar in sidecars:
        url = sidecar.get("browser_download_url")
        if not url:
            continue
        try:
            with opener(_request(str(url)), timeout=30) as response:  # type: ignore[union-attr]
                text = response.read().decode("utf-8", "replace")  # type: ignore[union-attr]
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise AppImageUpdateError(f"Could not download the AppImage checksum: {exc}") from exc
        if str(sidecar.get("name") or "").lower() == f"{asset_name}.sha256".lower():
            matches = SHA256_RE.findall(text)
            if len(matches) == 1:
                return matches[0].lower()
            continue

        matches = []
        for line in text.splitlines():
            match = re.fullmatch(r"\s*([a-fA-F0-9]{64})\s+\*?(.+?)\s*", line)
            if match and Path(match.group(2)).name == asset_name:
                matches.append(match.group(1).lower())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AppImageUpdateError(f"The checksum file has multiple entries for {asset_name}")
    raise AppImageUpdateError("The latest AppImage has no verifiable SHA-256 digest")


def _download_and_install(
    target: Path,
    asset: dict,
    expected_sha256: str,
    opener: Callable[..., object] = urlopen,
) -> None:
    target = target.expanduser()
    if target.is_symlink():
        try:
            target = target.resolve(strict=True)
        except OSError as exc:
            raise AppImageUpdateError(f"AppImage symlink target is unavailable: {target}") from exc
    if not target.is_file():
        raise AppImageUpdateError(f"AppImage path is not a regular file: {target}")
    parent = target.parent
    mode = stat.S_IMODE(target.stat().st_mode) | stat.S_IXUSR
    temp_path: Path | None = None
    actual = hashlib.sha256()
    try:
        with opener(_request(str(asset["browser_download_url"])), timeout=120) as response:  # type: ignore[union-attr]
            with tempfile.NamedTemporaryFile(
                dir=parent, prefix=f".{target.name}.", suffix=".download", delete=False
            ) as temp:
                temp_path = Path(temp.name)
                source = response  # type: ignore[assignment]
                while True:
                    chunk = source.read(CHUNK_SIZE)  # type: ignore[union-attr]
                    if not chunk:
                        break
                    actual.update(chunk)
                    temp.write(chunk)
                temp.flush()
                os.fsync(temp.fileno())
        actual_hex = actual.hexdigest()
        if actual_hex != expected_sha256:
            raise AppImageUpdateError(
                f"AppImage checksum mismatch (expected {expected_sha256}, got {actual_hex})"
            )
        os.chmod(temp_path, mode)
        os.replace(temp_path, target)
        temp_path = None
        try:
            dir_fd = os.open(parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise AppImageUpdateError(f"Could not download the latest AppImage: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def update_appimage(
    target: str | Path,
    *,
    arch: str,
    release_api: str = DEFAULT_RELEASE_API,
    opener: Callable[..., object] = urlopen,
) -> str:
    """Install the verified latest AppImage and return its release tag."""
    release = _read_json(release_api, opener)
    if release.get("draft") or release.get("prerelease"):
        raise AppImageUpdateError("The latest release is not stable")
    asset = _asset_for_release(release, arch)
    expected = _checksum_from_release(release, asset, opener)
    _download_and_install(Path(target), asset, expected, opener)
    return str(release.get("tag_name") or "latest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--release-api", default=DEFAULT_RELEASE_API)
    args = parser.parse_args(argv)
    try:
        tag = update_appimage(args.target, arch=args.arch, release_api=args.release_api)
    except AppImageUpdateError as exc:
        print(f"AppImage update failed: {exc}", file=sys.stderr)
        return 1
    print(f"AppImage updated to {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
