"""Reviewed native release assets for the catalog installer."""

from __future__ import annotations

import hashlib
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

_TARGETS = {
    "linux-amd64": (".tar.gz", "min_glibc"),
    "linux-arm64": (".tar.gz", "min_glibc"),
    "macos-amd64": (".zip", "min_macos"),
    "macos-arm64": (".zip", "min_macos"),
}
# Magic/word-size/endian prefix, then CPU field offset and little-endian value.
_EXECUTABLE_HEADERS = {
    "linux-amd64": (b"\x7fELF\x02\x01", 18, b"\x3e\x00"),
    "linux-arm64": (b"\x7fELF\x02\x01", 18, b"\xb7\x00"),
    "macos-amd64": (b"\xcf\xfa\xed\xfe", 4, b"\x07\x00\x00\x01"),
    "macos-arm64": (b"\xcf\xfa\xed\xfe", 4, b"\x0c\x00\x00\x01"),
}
_OS_NAMES = {"darwin": "macos", "linux": "linux"}
_RELEASE_HOSTS = frozenset({"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"})
_MAX_ASSET_BYTES = 64 * 1024 * 1024
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"[0-9]{1,4}\.[0-9]{1,4}(?:\.[0-9]{1,4})?\Z")


@dataclass(frozen=True)
class NativeAsset:
    filename: str
    sha256: str
    min_glibc: str | None = None
    min_macos: str | None = None

    def to_dict(self) -> dict[str, str]:
        value = {"filename": self.filename, "sha256": self.sha256}
        if self.min_glibc is not None:
            value["min_glibc"] = self.min_glibc
        if self.min_macos is not None:
            value["min_macos"] = self.min_macos
        return value


@dataclass(frozen=True)
class NativeAssets:
    tag: str
    member: str
    destination: str
    targets: dict[str, NativeAsset]

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "member": self.member,
            "destination": self.destination,
            "targets": {key: value.to_dict() for key, value in self.targets.items()},
        }


def github_repository(repo: str) -> str:
    """Return the owner/repo for a catalog GitHub release asset."""
    match = re.fullmatch(r"https://github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9_.-]+)/?", repo)
    if match:
        name = match[2].removesuffix(".git")
        if name not in {"", ".", ".."}:
            return f"{match[1]}/{name}"
    raise ValueError("native_assets requires an HTTPS GitHub repository")


def _component(value: object) -> bool:
    return isinstance(value, str) and value not in {".", ".."} and bool(_SAFE_COMPONENT.fullmatch(value))


def _relative_path(value: object) -> bool:
    return isinstance(value, str) and all(
        _component(part) and part.lower() != ".git" for part in value.split("/")
    )


def _parse_target(key: str, raw: object) -> NativeAsset:
    suffix, baseline = _TARGETS[key]
    if not isinstance(raw, dict) or set(raw) != {"filename", "sha256", baseline}:
        raise ValueError(f"native_assets target {key} requires filename, sha256 and {baseline}")
    filename, sha256, version = raw["filename"], raw["sha256"], raw[baseline]
    if not _component(filename) or not filename.endswith(suffix):
        raise ValueError(f"native_assets target {key} requires a {suffix} filename")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ValueError(f"native_assets target {key} requires a lowercase SHA-256 digest")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError(f"native_assets target {key} requires a dotted version for {baseline}")
    return NativeAsset(filename, sha256, **{baseline: version})


def parse_native_assets(value: object, *, repo: str, platforms: list[str]) -> NativeAssets | None:
    """Validate the declaration shared by the live catalog and admission gate."""
    if value is None:
        return None
    github_repository(repo)
    if not isinstance(value, dict) or set(value) != {"tag", "member", "destination", "targets"}:
        raise ValueError("native_assets requires tag, member, destination and targets")
    if not _component(value["tag"]):
        raise ValueError("native_assets.tag must be a single release tag component")
    for key in ("member", "destination"):
        if not _relative_path(value[key]) or value[key].split("/")[0].startswith("."):
            raise ValueError(f"native_assets.{key} must name a relative file in the package")
    targets = value["targets"]
    if not isinstance(targets, dict) or not targets or targets.keys() - _TARGETS.keys():
        raise ValueError("native_assets.targets must name supported Linux or macOS CPU targets")
    if platforms and any(key.split("-")[0] not in platforms for key in targets):
        raise ValueError("native_assets targets must match the catalog platforms")
    return NativeAssets(value["tag"], value["member"], value["destination"],
                        {key: _parse_target(key, raw) for key, raw in targets.items()})


def _version_parts(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")) if _VERSION.fullmatch(version) else ()


def select_native_asset(
    assets: NativeAssets, os_family: str, arch: str, version: str,
) -> tuple[str, NativeAsset] | None:
    """Select from the backend's OS and native CPU facts."""
    platform = _OS_NAMES.get(os_family)
    if platform is None or not any(key.startswith(f"{platform}-") for key in assets.targets):
        return None
    key = f"{platform}-{arch}"
    asset = assets.targets.get(key)
    if asset is None:
        raise ValueError(f"No reviewed native host is available for {key}")
    minimum = asset.min_glibc or asset.min_macos or ""
    baseline = "glibc" if platform == "linux" else "macOS"
    actual, required = _version_parts(version), _version_parts(minimum)
    if not actual or (actual + (0,) * 3)[:3] < (required + (0,) * 3)[:3]:
        raise ValueError(f"The native host for {key} requires {baseline} {minimum} or later")
    return key, asset


class _ReleaseRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urlsplit(newurl)
        if url.scheme != "https" or url.netloc not in _RELEASE_HOSTS:
            raise ValueError("GitHub redirected the native host download outside release storage")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_native_archive(repo: str, tag: str, asset: NativeAsset, archive_path: Path) -> None:
    """Fetch a pinned GitHub release file and verify it before inspecting its members."""
    url = f"https://github.com/{github_repository(repo)}/releases/download/{tag}/{asset.filename}"
    digest = hashlib.sha256()
    size = 0
    try:
        with build_opener(_ReleaseRedirects()).open(url, timeout=30) as response, archive_path.open("xb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > _MAX_ASSET_BYTES:
                    raise ValueError(f"Native archive {asset.filename} is too large")
                digest.update(block)
                output.write(block)
    except HTTPError as exc:
        raise ValueError(f"Could not download {asset.filename} (HTTP {exc.code})") from exc
    except OSError as exc:
        raise ValueError(f"Could not download {asset.filename}: {getattr(exc, 'reason', exc)}") from exc
    if digest.hexdigest() != asset.sha256:
        raise ValueError(f"SHA-256 mismatch for {asset.filename}")


def _one_member(entries: list, member: str):
    if len(entries) != 1:
        raise ValueError(f"Native archive must contain exactly one {member}")
    return entries[0]


def _open_zip_member(archive: zipfile.ZipFile, member: str):
    entry = _one_member([item for item in archive.infolist() if item.filename == member], member)
    kind = stat.S_IFMT(entry.external_attr >> 16)
    if entry.is_dir() or kind not in {0, stat.S_IFREG} or not 0 < entry.file_size <= _MAX_ASSET_BYTES:
        raise ValueError(f"Native archive member {member} must be a bounded regular file")
    return archive.open(entry)


def _open_tar_member(archive: tarfile.TarFile, member: str):
    entry = _one_member([item for item in archive if item.name == member], member)
    if not entry.isreg() or not 0 < entry.size <= _MAX_ASSET_BYTES:
        raise ValueError(f"Native archive member {member} must be a bounded regular file")
    return archive.extractfile(entry)


def extract_native_executable(archive_path: Path, member: str, destination: Path, *, target: str) -> None:
    """Copy the regular executable for the declared target out of a verified archive."""
    is_zip = archive_path.suffix == ".zip"
    opener, get_member = (zipfile.ZipFile, _open_zip_member) if is_zip else (tarfile.open, _open_tar_member)
    magic, offset, machine = _EXECUTABLE_HEADERS[target]
    try:
        with opener(archive_path, "r" if is_zip else "r:gz") as archive, get_member(archive, member) as source:
            header = source.read(20)
            if not header.startswith(magic) or header[offset:offset + len(machine)] != machine:
                raise ValueError(f"Native archive member {member} does not match {target}")
            with destination.open("xb") as output:
                output.write(header)
                shutil.copyfileobj(source, output)
        destination.chmod(0o755)
    except (OSError, tarfile.TarError, zipfile.BadZipFile, EOFError, RuntimeError) as exc:
        raise ValueError(f"Could not stage native executable {member}: {exc}") from exc


def managed_native_paths(assets: NativeAssets | None, record: dict | None) -> frozenset[str]:
    """Paths owned by the installer on either side of a plugin replacement."""
    catalog = (record or {}).get("catalog")
    receipt = catalog.get("native_asset") if isinstance(catalog, dict) else None
    old_path = receipt.get("path") if isinstance(receipt, dict) else None
    current_path = assets.destination if assets is not None else None
    return frozenset(path for path in (old_path, current_path) if _relative_path(path))


def stage_catalog_native_asset(catalog: dict | None, assets: NativeAssets | None, staged: Path) -> None:
    """Provision the reviewed executable and record its origin before publication."""
    if assets is None or catalog is None:
        return
    from hermes_cli.plugins_cmd import PluginOperationError
    from hermes_platform.host import facts

    if catalog["pin"] != catalog["sha"]:
        raise PluginOperationError("Native release assets require the exact reviewed source commit.")
    os_family = facts.os_family()
    version = facts.glibc_version() if os_family == "linux" else facts.macos_version()
    try:
        selection = select_native_asset(assets, os_family, facts.native_arch(), version)
    except ValueError as exc:
        raise PluginOperationError(str(exc), failure_class="incompatible") from exc
    if selection is None:
        return
    key, asset = selection
    destination = staged.resolve() / assets.destination
    try:
        if destination.resolve() != destination or destination.exists():
            raise ValueError(f"Native host destination is already occupied or unsafe: {assets.destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native-asset-", dir=staged.parent) as scratch:
            archive = Path(scratch) / asset.filename
            download_native_archive(catalog["repo"], assets.tag, asset, archive)
            extract_native_executable(archive, assets.member, destination, target=key)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PluginOperationError(str(exc)) from exc
    catalog["native_asset"] = {"target": key, "sha256": asset.sha256, "path": assets.destination}
