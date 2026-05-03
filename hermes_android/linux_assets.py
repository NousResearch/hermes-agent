from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TERMUX_MAIN_BASE_URL = "https://packages.termux.dev/apt/termux-main"
TERMUX_PACKAGES_INDEX_TEMPLATE = TERMUX_MAIN_BASE_URL + "/dists/stable/main/binary-{termux_arch}/Packages"
TERMUX_PREFIX = "/data/data/com.termux/files/usr"
ANDROID_LINUX_ASSET_ROOT = "hermes-linux"

ANDROID_TO_TERMUX_ARCH = {
    "arm64-v8a": "aarch64",
    "x86_64": "x86_64",
}
TERMUX_TO_ANDROID_ARCH = {value: key for key, value in ANDROID_TO_TERMUX_ARCH.items()}

ROOT_PACKAGES = [
    "bash",
    "busybox",
    "bzip2",
    "coreutils",
    "curl",
    "diffutils",
    "findutils",
    "gawk",
    "git",
    "grep",
    "gzip",
    "less",
    "llama-cpp",
    "procps",
    "sed",
    "tar",
    "util-linux",
    "xz-utils",
]

IGNORED_DEPENDENCIES = {
    "termux-am",
    "termux-am-socket",
    "termux-auth",
    "termux-core",
    "termux-exec",
    "termux-keyring",
    "termux-licenses",
    "termux-tools",
}

TEXT_SHEBANG_PREFIX_RE = re.compile(r"^#!" + re.escape(TERMUX_PREFIX) + r"/bin/([^\n\r/]+)")


@dataclass(frozen=True)
class TermuxPackageRecord:
    name: str
    version: str
    filename: str
    sha256: str
    depends: tuple[str, ...] = ()

    @property
    def download_url(self) -> str:
        return f"{TERMUX_MAIN_BASE_URL}/{self.filename}"


def parse_depends(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    resolved: list[str] = []
    for chunk in raw.split(","):
        options = []
        for option in chunk.split("|"):
            name = re.sub(r"\s*\(.*?\)", "", option).strip()
            if not name:
                continue
            options.append(name)
        if not options:
            continue
        selected = next((name for name in options if name not in IGNORED_DEPENDENCIES), options[0])
        if selected in IGNORED_DEPENDENCIES:
            continue
        resolved.append(selected)
    return tuple(dict.fromkeys(resolved))


def parse_packages_index(text: str) -> dict[str, TermuxPackageRecord]:
    records: dict[str, TermuxPackageRecord] = {}
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if current.get("Package") and current.get("Version") and current.get("Filename") and current.get("SHA256"):
                record = TermuxPackageRecord(
                    name=current["Package"],
                    version=current["Version"],
                    filename=current["Filename"],
                    sha256=current["SHA256"],
                    depends=parse_depends(current.get("Depends")),
                )
                records[record.name] = record
            current = {}
            continue
        if line.startswith((" ", "\t")):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key] = value.strip()
    if current.get("Package") and current.get("Version") and current.get("Filename") and current.get("SHA256"):
        record = TermuxPackageRecord(
            name=current["Package"],
            version=current["Version"],
            filename=current["Filename"],
            sha256=current["SHA256"],
            depends=parse_depends(current.get("Depends")),
        )
        records[record.name] = record
    return records


def resolve_dependency_closure(
    records: dict[str, TermuxPackageRecord],
    root_packages: Iterable[str] = ROOT_PACKAGES,
) -> list[TermuxPackageRecord]:
    pending = list(root_packages)
    seen: set[str] = set()
    ordered: list[TermuxPackageRecord] = []
    while pending:
        name = pending.pop(0)
        if name in seen or name in IGNORED_DEPENDENCIES:
            continue
        record = records.get(name)
        if record is None:
            raise KeyError(f"Termux package '{name}' was not found in the package index")
        seen.add(name)
        ordered.append(record)
        for dependency in record.depends:
            if dependency not in seen and dependency not in IGNORED_DEPENDENCIES:
                pending.append(dependency)
    return ordered


def asset_manifest_path(output_dir: str | Path, android_abi: str) -> Path:
    return Path(output_dir) / ANDROID_LINUX_ASSET_ROOT / android_abi / "manifest.json"


def asset_prefix_dir(output_dir: str | Path, android_abi: str) -> Path:
    return Path(output_dir) / ANDROID_LINUX_ASSET_ROOT / android_abi / "prefix"


def verify_sha256(payload: bytes, expected: str) -> None:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: expected {expected}, got {actual}")


def load_data_tar_bytes_from_deb(payload: bytes) -> tuple[bytes, str]:
    blob = io.BytesIO(payload)
    magic = blob.read(8)
    if magic != b"!<arch>\n":
        raise ValueError("Not a valid ar archive")
    while True:
        header = blob.read(60)
        if not header:
            break
        if len(header) != 60:
            raise ValueError("Truncated ar archive header")
        name = header[:16].decode("utf-8", "replace").strip().rstrip("/")
        size = int(header[48:58].decode("utf-8", "replace").strip())
        file_payload = blob.read(size)
        if size % 2 == 1:
            blob.read(1)
        if name.startswith("data.tar"):
            return file_payload, name
    raise FileNotFoundError("data.tar member not found in deb archive")


def open_data_tar(data_bytes: bytes, member_name: str) -> tarfile.TarFile:
    if member_name.endswith(".xz"):
        return tarfile.open(fileobj=io.BytesIO(data_bytes), mode="r:xz")
    if member_name.endswith(".gz"):
        return tarfile.open(fileobj=io.BytesIO(data_bytes), mode="r:gz")
    if member_name.endswith(".bz2"):
        return tarfile.open(fileobj=io.BytesIO(data_bytes), mode="r:bz2")
    return tarfile.open(fileobj=io.BytesIO(data_bytes), mode="r:")


def normalize_text_shebang(content: bytes) -> bytes:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    lines = text.splitlines(keepends=True)
    if not lines:
        return content
    first = lines[0]
    match = TEXT_SHEBANG_PREFIX_RE.match(first)
    if not match:
        return content
    interpreter = match.group(1)
    suffix = first[first.find(match.group(0)) + len(match.group(0)) :]
    lines[0] = f"#!/usr/bin/env {interpreter}{suffix}"
    return "".join(lines).encode("utf-8")


def strip_termux_prefix(path: str) -> str | None:
    normalized = path.lstrip("./")
    prefix = TERMUX_PREFIX.lstrip("/") + "/"
    if not normalized.startswith(prefix):
        return None
    return normalized[len(prefix) :]


def _normalize_asset_link_path(path: str) -> str:
    return str(path).replace("\\", "/").strip().lstrip("/")


def serializable_manifest(android_abi: str, packages: Iterable[TermuxPackageRecord], links: Iterable[dict] = ()) -> dict:
    package_list = list(packages)
    normalized_links = [
        {
            **link,
            "path": _normalize_asset_link_path(link.get("path", "")),
            "target": _normalize_asset_link_path(link.get("target", "")),
        }
        for link in links
    ]
    return {
        "asset_root": ANDROID_LINUX_ASSET_ROOT,
        "android_abi": android_abi,
        "termux_arch": ANDROID_TO_TERMUX_ARCH[android_abi],
        "root_packages": list(ROOT_PACKAGES),
        "ignored_dependencies": sorted(IGNORED_DEPENDENCIES),
        "links": normalized_links,
        "packages": [
            {
                "name": package.name,
                "version": package.version,
                "filename": package.filename,
                "sha256": package.sha256,
                "depends": list(package.depends),
            }
            for package in package_list
        ],
    }


def write_manifest(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
