"""Download and safely install a verified Tor Expert Bundle."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

import httpx

from darkloom.constants import (
    CURRENT_ARCH,
    CURRENT_PLATFORM,
    TOR_BINARY_DIR,
    TOR_RELEASE_SIGNING_FINGERPRINTS,
    TOR_RELEASE_SIGNING_KEY,
    TOR_VERSION,
    get_download_url,
    get_signature_url,
    get_tor_binary_path,
    is_tor_installed,
)

# A compressed bundle should never need this much space.  Enforcing the limit
# from tar headers, before writing anything, also prevents sparse/zip bombs.
MAX_EXPANDED_SIZE = 512 * 1024 * 1024

from darkloom.privacy import get_logger

logger = get_logger(__name__)


class DownloadError(Exception):
    """Raised when a Tor binary cannot be securely installed."""


@dataclass(frozen=True)
class BundleManifest:
    """The complete file list and SHA-256 of every executable in a bundle."""

    files: frozenset[str]
    executable_digests: Mapping[str, str]


# Manifests are deliberately version/platform specific.  Maintainers populate
# this table only after inspecting a Tor release.  An unknown bundle
# must fail closed rather than silently acquiring trust from its own contents.
BUNDLE_MANIFESTS: dict[tuple[str, str], BundleManifest] = {}


def _normalise_member_name(name: str) -> PurePosixPath:
    """Return a safe relative POSIX archive path."""
    if not name or "\x00" in name:
        raise DownloadError("archive contains an empty or NUL-containing path")
    path = PurePosixPath(name)
    if path.is_absolute() or name.startswith(("/", "\\")):
        raise DownloadError(f"archive contains an absolute path: {name!r}")
    # Backslashes are separators on Windows; reject them everywhere so a file
    # validated on one platform cannot mean something different on another.
    if "\\" in name or any(part in ("", ".", "..") for part in path.parts):
        raise DownloadError(f"archive contains path traversal: {name!r}")
    return path


def _inspect_archive(
    tar: tarfile.TarFile,
    manifest: BundleManifest,
    max_expanded_size: int,
) -> list[tuple[tarfile.TarInfo, PurePosixPath]]:
    """Validate every member and return a safe extraction plan."""
    plan: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
    seen: set[str] = set()
    files: set[str] = set()
    expanded = 0

    for member in tar.getmembers():
        path = _normalise_member_name(member.name)
        name = path.as_posix()
        if name in seen:
            raise DownloadError(f"archive contains duplicate member: {name}")
        seen.add(name)

        if member.isdev() or member.isfifo():
            raise DownloadError(f"archive contains a device or FIFO: {name}")
        if member.islnk():
            raise DownloadError(f"archive contains a hard link: {name}")
        if not (member.isfile() or member.isdir() or member.issym()):
            raise DownloadError(f"archive contains unsupported member type: {name}")

        if member.issym():
            target = PurePosixPath(member.linkname)
            if target.is_absolute() or member.linkname.startswith(("/", "\\")):
                raise DownloadError(f"symlink target leaves staging directory: {name}")
            resolved: list[str] = list(path.parent.parts)
            for part in target.parts:
                if part in ("", "."):
                    continue
                if part == "..":
                    if not resolved:
                        raise DownloadError(
                            f"symlink target leaves staging directory: {name}"
                        )
                    resolved.pop()
                else:
                    if "\\" in part:
                        raise DownloadError(f"unsafe symlink target: {member.linkname}")
                    resolved.append(part)
            files.add(name)
        elif member.isfile():
            if member.size < 0:
                raise DownloadError(f"archive has a negative file size: {name}")
            if member.mode & 0o111 and name not in manifest.executable_digests:
                raise DownloadError(f"archive contains an unexpected executable: {name}")
            expanded += member.size
            if expanded > max_expanded_size:
                raise DownloadError("archive exceeds maximum expanded size")
            files.add(name)
        plan.append((member, path))

    if files != set(manifest.files):
        unexpected = sorted(files - set(manifest.files))
        missing = sorted(set(manifest.files) - files)
        raise DownloadError(
            f"archive file allowlist mismatch; unexpected={unexpected}, missing={missing}"
        )
    if not set(manifest.executable_digests).issubset(files):
        raise DownloadError("executable digest manifest refers to a missing file")
    return plan


def _extract_verified_archive(
    archive: Path,
    staging: Path,
    manifest: BundleManifest,
    *,
    max_expanded_size: int = MAX_EXPANDED_SIZE,
) -> None:
    """Inspect, extract without following links, and verify a bundle."""
    with tarfile.open(archive, "r:gz") as tar:
        plan = _inspect_archive(tar, manifest, max_expanded_size)
        # Directories first.  Files are created with restrictive modes and no
        # archive ownership/permission metadata is trusted.
        for member, relative in plan:
            destination = staging.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True, mode=0o700)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = tar.extractfile(member)
                if source is None:
                    raise DownloadError(f"cannot read archive member: {member.name}")
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(destination, flags, 0o600)
                with source, os.fdopen(fd, "wb") as output:
                    shutil.copyfileobj(source, output)

        # Links are made last, so they can never influence file extraction.
        for member, relative in plan:
            if member.issym():
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                destination.symlink_to(member.linkname)

    for name, expected in manifest.executable_digests.items():
        path = staging.joinpath(*PurePosixPath(name).parts)
        if path.is_symlink() or not path.is_file():
            raise DownloadError(f"executable is not a regular file: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != expected.lower():
            raise DownloadError(f"SHA-256 mismatch for executable: {name}")
        if os.name != "nt":
            path.chmod(0o755)

    # No allowlisted regular file may arrive executable unless its digest was
    # explicitly pinned (archive modes themselves are otherwise ignored).
    executable_names = set(manifest.executable_digests)
    for member_name in manifest.files:
        path = staging.joinpath(*PurePosixPath(member_name).parts)
        if path.is_file() and not path.is_symlink() and member_name not in executable_names:
            path.chmod(0o600)


def _atomic_install(staging: Path, destination: Path) -> None:
    """Publish a verified staging tree, retaining/rolling back the old tree."""
    backup = destination.with_name(f".{destination.name}.previous-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    old_moved = False
    try:
        if destination.exists():
            destination.rename(backup)
            old_moved = True
        staging.rename(destination)
    except BaseException:
        if old_moved and not destination.exists():
            backup.rename(destination)
        raise
    if old_moved:
        shutil.rmtree(backup)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_signature(artifact: Path, signature: Path) -> str:
    """Verify a detached signature in an isolated keyring and return its signer."""
    if not TOR_RELEASE_SIGNING_KEY.is_file():
        raise DownloadError("Bundled Tor release signing key is missing")
    with tempfile.TemporaryDirectory(prefix="darkloom-gpg-") as home:
        home_path = Path(home)
        home_path.chmod(0o700)
        base = ["gpg", "--batch", "--no-tty", "--homedir", home]
        try:
            imported = subprocess.run(
                base + ["--status-fd", "1", "--import", str(TOR_RELEASE_SIGNING_KEY)],
                capture_output=True, text=True, check=False,
            )
            if imported.returncode:
                raise DownloadError(f"Could not import bundled Tor signing key: {imported.stderr.strip()}")
            checked = subprocess.run(
                base + ["--status-fd", "1", "--verify", str(signature), str(artifact)],
                capture_output=True, text=True, check=False,
            )
        except FileNotFoundError as exc:
            raise DownloadError("GnuPG is required to authenticate Tor downloads") from exc

    statuses = [line.removeprefix("[GNUPG:] ").split() for line in checked.stdout.splitlines()
                if line.startswith("[GNUPG:] ")]
    fatal = {"BADSIG", "ERRSIG", "EXPSIG", "EXPKEYSIG", "REVKEYSIG", "NO_PUBKEY"}
    failures = [parts[0] for parts in statuses if parts and parts[0] in fatal]
    valid = next((parts for parts in statuses if parts and parts[0] == "VALIDSIG"), None)
    if checked.returncode or failures or valid is None:
        reason = ", ".join(failures) or checked.stderr.strip() or "no valid signature"
        raise DownloadError(f"Tor bundle signature verification failed: {reason}")
    signer = valid[1].upper()
    primary = valid[-1].upper()
    if signer not in TOR_RELEASE_SIGNING_FINGERPRINTS and primary not in TOR_RELEASE_SIGNING_FINGERPRINTS:
        raise DownloadError(f"Tor bundle was signed by unknown key {signer}")
    return primary if primary in TOR_RELEASE_SIGNING_FINGERPRINTS else signer


def _metadata_path() -> Path:
    return TOR_BINARY_DIR / "install-metadata.json"


def validate_installed_binary(*, strict: bool = True) -> bool:
    """Re-hash the executable against authenticated installation metadata."""
    binary = get_tor_binary_path()
    metadata_path = _metadata_path()
    if not binary.is_file():
        return False
    if not metadata_path.is_file():
        if strict:
            raise DownloadError("Tor installation has no signature-verification metadata")
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        signer = metadata["signer_fingerprint"].upper()
        expected = metadata["executable_sha256"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DownloadError("Tor installation verification metadata is invalid") from exc
    if signer not in TOR_RELEASE_SIGNING_FINGERPRINTS:
        raise DownloadError(f"Tor installation records unknown signer {signer}")
    if not isinstance(expected, str) or _sha256(binary) != expected:
        raise DownloadError("Installed Tor executable does not match its verified digest")
    return True


def _download(url: str, destination: Path, progress_callback=None) -> None:
    """Download a file with optional progress tracking."""
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with destination.open("wb") as stream:
            for chunk in response.iter_bytes(chunk_size=65536):
                stream.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total)


def download_tor_binary(progress_callback=None, force: bool = False) -> Path:
    if is_tor_installed() and not force:
        return get_tor_binary_path()

    manifest = BUNDLE_MANIFESTS.get((CURRENT_PLATFORM, CURRENT_ARCH))
    if manifest is None:
        raise DownloadError(
            f"no verified bundle manifest for {CURRENT_PLATFORM}/{CURRENT_ARCH}"
        )

    url = get_download_url()
    TOR_BINARY_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        archive = Path(tmp.name)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{TOR_BINARY_DIR.name}.staging-", dir=TOR_BINARY_DIR.parent
        )
    )

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
            if response.status_code >= 400:
                raise DownloadError(f"HTTP {response.status_code} downloading {url}")
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with archive.open("wb") as output:
                for chunk in response.iter_bytes(chunk_size=65536):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)

        _extract_verified_archive(archive, staging, manifest)
        _atomic_install(staging, TOR_BINARY_DIR)
        return get_tor_binary_path()
    except httpx.HTTPError as exc:
        raise DownloadError(f"Failed to download Tor bundle: {exc}") from exc
    except (tarfile.TarError, OSError) as exc:
        raise DownloadError(f"Failed to install Tor bundle: {exc}") from exc
    finally:
        archive.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
