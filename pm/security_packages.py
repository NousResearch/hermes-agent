"""Optional security executables. Acquisition and publication belong to PM."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

from pm import paths
from pm.lock import Lockfile
from pm.package import InstallError
from pm.packages import BinaryPackage, _RUST_TRIPLE
from pm.registry import register
from pm.store import flatten_single_dir


@register
class Bws(BinaryPackage):
    name = "bws"
    optional = True
    binary_rel = {"posix": "bws", "win32": "bws.exe"}
    gaps = {"linux-arm64-bionic": "upstream does not ship an Android binary"}

    def fetch_url(self, version: str, target: str) -> str:
        # The static musl build runs on glibc too. One portable Linux artifact
        # removes the consumer's ldd probe without losing musl support.
        triple = _RUST_TRIPLE[target].replace("linux-gnu", "linux-musl")
        platform = "macos-universal" if target.startswith("darwin") else triple
        return f"https://github.com/bitwarden/sdk-sm/releases/download/bws-v{version}/bws-{platform}-{version}.zip"

    def stage(self, store, staged: Path, version: str, target: str) -> None:
        super().stage(store, staged, version, target)
        binary = self.binary(staged, target)
        if binary is not None and binary.is_file():
            binary.chmod(0o755)


class _SignedBinary(BinaryPackage):
    """The same PM artifact transaction also acquires pinned provenance files.

    The consumer owns signature semantics, not a second downloader. Checksum
    signatures must cover the *archive* pinned by PM, not an unrelated blob.
    Every locked provenance artifact is required even without a verifier on
    PATH. This intentionally strengthens cold-install availability: a missing
    or corrupt signature/key/certificate cannot silently reduce the pinned
    closure. Verifier execution remains optional; explicit rejection is fatal.
    """
    optional = True
    gaps = {
        "win32-x64": "this integration requires a Linux/macOS host (or WSL)",
        "win32-arm64": "this integration requires a Linux/macOS host (or WSL)",
        "linux-arm64-bionic": "upstream does not ship an Android binary",
    }

    def unpack(self, archive: Path, staged: Path, target: str) -> None:
        if archive.name.endswith((".txt", ".sig", ".pem", ".asc")):
            directory = staged / ".provenance"
            directory.mkdir(parents=True)
            shutil.copyfile(archive, directory / archive.name)
        else:
            super().unpack(archive, staged, target)
            flatten_single_dir(staged)

    def stage(self, store, staged: Path, version: str, target: str) -> None:
        binary = self.binary(staged, target)
        if binary is None or not binary.is_file() or binary.is_symlink():
            raise InstallError(self.name, "archive binary is not a regular file")
        binary.chmod(0o755)
        provenance = staged / ".provenance"
        artifacts = Lockfile(paths.lockfile_path()).artifacts(self.name, target)
        archive = artifacts[0]
        filename = Path(urlparse(archive["url"]).path).name
        rows = (line.split() for line in (provenance / "checksums.txt").read_text(encoding="utf-8-sig").splitlines())
        expected = next((row[0] for row in rows if len(row) == 2 and row[1].lstrip("*") == filename), None)
        if expected != archive["sha256"]:
            raise InstallError(self.name, "release checksums do not cover the pinned archive")
        self.verify_provenance(provenance)

    def verify_provenance(self, directory: Path) -> None:
        raise NotImplementedError


@register
class Tirith(_SignedBinary):
    name = "tirith"
    binary_rel = {"posix": "tirith"}

    def fetch_url(self, version: str, target: str) -> str:
        return f"https://github.com/sheeki03/tirith/releases/download/v{version}/tirith-{_RUST_TRIPLE[target]}.tar.gz"

    def fetch_urls(self, version: str, target: str) -> list[str]:
        archive = self.fetch_url(version, target)
        base = archive.rsplit("/", 1)[0]
        return [archive, *[f"{base}/{name}" for name in ("checksums.txt", "checksums.txt.sig", "checksums.txt.pem")]]

    def verify_provenance(self, directory: Path) -> None:
        from tools.tirith_security import _verify_release_provenance

        _, reason = _verify_release_provenance(directory, logging.getLogger(__name__).warning)
        if reason:
            raise InstallError(self.name, reason)


@register
class IronProxy(_SignedBinary):
    name = "iron-proxy"
    binary_rel = {"posix": "iron-proxy"}
    # The release omits its public key asset; pin the key from its source commit.
    signing_key_url = "https://raw.githubusercontent.com/paradigmxyz/iron-proxy/be5f255d0d9d10d8573bd65f480dd48a07772bf1/public-key.asc"

    def _probe_env(self) -> dict:
        from agent.proxy_sources.iron_proxy import _allowlisted_env

        return _allowlisted_env()

    def fetch_url(self, version: str, target: str) -> str:
        platform, arch = target.split("-")
        arch = "amd64" if arch == "x64" else arch
        return f"https://github.com/paradigmxyz/iron-proxy/releases/download/v{version}/iron-proxy_{version}_{platform}_{arch}.tar.gz"

    def fetch_urls(self, version: str, target: str) -> list[str]:
        archive = self.fetch_url(version, target)
        base = archive.rsplit("/", 1)[0]
        return [archive, f"{base}/checksums.txt", f"{base}/checksums.txt.asc", self.signing_key_url]

    def verify_provenance(self, directory: Path) -> None:
        from agent.proxy_sources.iron_proxy import _verify_checksums_signature

        _verify_checksums_signature(directory, directory / "checksums.txt")