"""Wisdom package contract: what the Gateway hashes, verifies and refuses.

Every byte here is checked server-side (``canonical-hash-vectors.v1.json`` in the test tree):
content manifest lines, the ``sha256:`` address, author-description sanitizing and the
canonical ``skill.manifest.json`` dump. Change nothing without a vector proving it still matches.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tools.skills_sync_client_wire import ObjectSet, build_commit, build_tree

MAX_FILES = 32
MAX_FILE_BYTES = 256 * 1024
MAX_TREE_BYTES = 512 * 1024
MAX_TREE_DEPTH = 3
ROOT_FILES = frozenset({"SKILL.md", "skill.manifest.json"})
SUPPORT_DIRS = frozenset({"refs", "assets"})
TEXT_SUFFIXES = frozenset({".txt", ".md", ".rst", ".adoc", ".asciidoc"})
BLOCKED_SEGMENTS = frozenset({
    ".circleci", ".git", ".github", ".gitlab", ".hooks", ".tox", ".venv", "__pycache__",
    "hooks", "node_modules", "scripts", "templates", "venv",
})
PACKAGE_MANAGER_FILES = frozenset({
    "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "requirements.txt",
    "pyproject.toml", "poetry.lock", "Pipfile", "Cargo.toml", "Cargo.lock", "go.mod", "go.sum",
    "Gemfile", "Gemfile.lock",
})
WINDOWS_RESERVED = frozenset({"con", "prn", "aux", "nul", "clock$", "conin$", "conout$"}
                             | {f"com{n}" for n in range(1, 10)} | {f"lpt{n}" for n in range(1, 10)})
# Instruction-only packages: a SKILL.md pointing at scripts/ or templates/ would silently lose them.
ACTIVE_REFERENCE_RE = re.compile(r"(?i)(?:^|[\s\[(`'\"])(?:scripts?|templates?)/[^\s)`'\"]+")
SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PackageError(ValueError):
    """The package violates the instruction-only contract; never retried automatically."""


# --- manifest (server schema v1, additionalProperties: false everywhere) -----------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolRequirement(_Strict):
    name: str
    minimum_version: str | None = None
    auto_install: Literal[False] = False
    requires_admin: bool = False


class PluginRequirement(_Strict):
    id: str
    minimum_version: str | None = None
    required: bool = True


class _Model(_Strict):
    capabilities: list[str] = Field(default_factory=list)
    minimum_context_window: int | None = None


class _Filesystem(_Strict):
    read: list[str] = Field(default_factory=list)
    write: list[str] = Field(default_factory=list)


class _Network(_Strict):
    destinations: list[str] = Field(default_factory=list)


class _Runtime(_Strict):
    shell: bool = False
    browser: bool = False
    code: bool = False
    sandbox: bool = True


class _Hermes(_Strict):
    minimum_version: str


class SystemSpec(_Strict):
    hermes: _Hermes
    platforms: list[str] = Field(default_factory=list)
    architectures: list[str] = Field(default_factory=list)
    model: _Model = Field(default_factory=_Model)
    tools: list[ToolRequirement] = Field(default_factory=list)
    plugins: list[PluginRequirement] = Field(default_factory=list)
    credentials: list[str] = Field(default_factory=list)
    connections: list[str] = Field(default_factory=list)
    filesystem: _Filesystem = Field(default_factory=_Filesystem)
    network: _Network = Field(default_factory=_Network)
    runtime: _Runtime = Field(default_factory=_Runtime)
    hardware: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)


class PackageManifest(_Strict):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=512)
    requirements: SystemSpec


def sha256_address(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def content_hash(files: list[tuple[str, str]]) -> str:
    """``files`` = ``[(path, sha256_address), ...]``; lines sorted as whole strings."""
    return sha256_address("".join(sorted(f"{p} file {h}\n" for p, h in files)).encode("utf-8"))


def sanitize_description(raw: str) -> str:
    """Mirror of the Gateway's descriptionSanitizer: strip tags, drop control chars, NFC, trim."""
    value = re.sub(r"<[^>]*>", "", raw)
    value = "".join(c for c in value if ord(c) in (9, 10, 13) or (ord(c) > 0x1F and ord(c) != 0x7F))
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = "\n".join(line.rstrip() for line in value.split("\n")).strip()
    if not value or len(value.encode("utf-8")) > 4096:
        raise PackageError("author description must be 1..4096 canonical UTF-8 bytes")
    return value


def manifest_bytes(manifest: PackageManifest) -> bytes:
    """Server-canonical dump: schema field order (matches the Gateway's vectors), no whitespace."""
    return json.dumps(manifest.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def parse_manifest(raw: bytes) -> PackageManifest:
    def no_dupes(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate JSON key: {k}")
            out[k] = v
        return out
    return PackageManifest.model_validate(json.loads(raw.decode("utf-8"), object_pairs_hook=no_dupes))


def slug_for(name: str) -> str:
    candidate = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:64]
    if not SLUG_RE.fullmatch(candidate):
        raise PackageError(f"skill name {name!r} cannot become a Wisdom slug")
    return candidate


# --- path + content policy (applies to what we upload AND what we download) --------------------
def _check_path(raw: str) -> tuple[PurePosixPath, str]:
    """Validate one package-relative POSIX path; return it plus its casefolded collision key."""
    if not raw or "\\" in raw or raw.startswith("/") or unicodedata.normalize("NFC", raw) != raw \
            or re.search(r"%[0-9a-fA-F]{2}", raw):
        raise PackageError(f"non-canonical package path: {raw!r}")
    parts = raw.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise PackageError(f"unsafe package path: {raw!r}")
    keys = []
    for seg in parts:
        if seg.endswith((".", " ")) or any(ord(c) < 0x20 or ord(c) == 0x7F or c in '<>:"|?*' for c in seg) \
                or len(seg.encode("utf-8")) > 255:
            raise PackageError(f"unsafe install-target path: {raw!r}")
        key = seg.casefold()
        if key.split(".", 1)[0] in WINDOWS_RESERVED or key in BLOCKED_SEGMENTS:
            raise PackageError(f"reserved or active-content path: {raw!r}")
        keys.append(key)
    if len(parts) - 1 > MAX_TREE_DEPTH:
        raise PackageError(f"package path deeper than {MAX_TREE_DEPTH}: {raw!r}")
    pure = PurePosixPath(raw)
    if len(parts) == 1:
        if pure.name not in ROOT_FILES:
            raise PackageError(f"unsupported root file {raw}; Wisdom packages are instruction-only")
    elif parts[0] not in SUPPORT_DIRS or pure.name in ROOT_FILES or pure.suffix.casefold() not in TEXT_SUFFIXES:
        raise PackageError(f"unsupported package path {raw}; only refs/ and assets/ text files are allowed")
    if pure.name in PACKAGE_MANAGER_FILES:
        raise PackageError(f"package-manager manifest is not allowed: {raw}")
    return pure, "/".join(keys)


def _check_text(body: bytes, path: str) -> None:
    if b"\x00" in body or body.startswith(b"#!"):
        raise PackageError(f"binary or executable content is not allowed: {path}")
    try:
        body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackageError(f"binary content is not allowed: {path}") from exc


def verify_files(files: list[tuple[str, str, bytes]], *, require_manifest: bool = True) -> str:
    """Validate ``(path, mode, bytes)`` records (upload or download) and return their content hash."""
    if len(files) > MAX_FILES:
        raise PackageError(f"package exceeds {MAX_FILES} files")
    keys: set[str] = set()
    total = 0
    hashes = []
    for path, mode, body in files:
        _, key = _check_path(path)
        if key in keys:
            raise PackageError(f"install-target collision: {path}")
        keys.add(key)
        if mode != "file":
            raise PackageError(f"executable/unknown mode rejected: {path}")
        if len(body) > MAX_FILE_BYTES:
            raise PackageError(f"file exceeds {MAX_FILE_BYTES} bytes: {path}")
        total += len(body)
        if total > MAX_TREE_BYTES:
            raise PackageError(f"package exceeds {MAX_TREE_BYTES} total bytes")
        _check_text(body, path)
        hashes.append((path, sha256_address(body)))
    names = {p for p, _ in hashes}
    if "SKILL.md" not in names or (require_manifest and "skill.manifest.json" not in names):
        raise PackageError("not a complete Wisdom package (SKILL.md + skill.manifest.json)")
    skill_md = next(b for p, _, b in files if p == "SKILL.md").decode("utf-8")
    if ACTIVE_REFERENCE_RE.search(skill_md):
        raise PackageError("SKILL.md references scripts/ or templates/; Wisdom packages are instruction-only")
    return content_hash(hashes)


def read_source(source: Path) -> list[tuple[str, str, bytes]]:
    """Read a local skill directory as package records, refusing anything the contract forbids."""
    if not source.is_dir() or source.is_symlink():
        raise PackageError("skill root must be a regular directory")
    records = []
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source).as_posix()
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PackageError(f"symlinks and special files are not supported: {rel}")
        if path.is_dir():
            if path.parent == source and path.name not in SUPPORT_DIRS:
                raise PackageError(f"unsupported directory {rel}; scripts/templates cannot be shared")
            continue
        st = path.stat()
        if st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) or st.st_nlink != 1:
            raise PackageError(f"executable or hard-linked content is not allowed: {rel}")
        records.append((rel, "file", path.read_bytes()))
    return records


# --- authoring -----------------------------------------------------------------------------------
def _strings(value) -> list[str]:
    items = [value] if isinstance(value, str) else [v for v in value if isinstance(v, str)] if isinstance(value, list) else []
    return list(dict.fromkeys(s.strip() for s in items if s.strip()))[:64]


def infer_spec(skill_md: str, *, hermes_version: str) -> SystemSpec:
    """Conservative requirements from frontmatter + the authoring host; no inventory is exported."""
    from agent.skill_utils import parse_frontmatter
    fm, _ = parse_frontmatter(skill_md)
    fm = fm if isinstance(fm, dict) else {}
    meta = (fm.get("metadata") or {}).get("hermes") if isinstance(fm.get("metadata"), dict) else None
    meta = meta if isinstance(meta, dict) else {}
    os_names = {"darwin": "macOS", "mac": "macOS", "macos": "macOS", "osx": "macOS"}
    def norm_os(v):
        k = v.strip().casefold()
        return os_names.get(k) or ("Linux" if k.startswith("linux") else "Windows" if k.startswith("win") else v.strip())
    def norm_arch(v):
        k = v.strip().casefold().replace("-", "_")
        return "arm64" if k in {"aarch64", "arm64"} else "x86_64" if k in {"amd64", "x64", "x86_64"} else v.strip()
    platforms = [norm_os(p) for p in _strings(fm["platforms"])] if "platforms" in fm else [norm_os(platform.system())]
    arch_src = fm.get("architectures", meta.get("architectures"))
    archs = [norm_arch(a) for a in _strings(arch_src)] if arch_src is not None else [norm_arch(platform.machine())]
    tool_names = _strings(meta.get("requires_toolsets")) + _strings(meta.get("requires_tools"))
    tools = [ToolRequirement(name=n) for n in dict.fromkeys(tool_names)][:64]
    plugins = []
    for item in (meta.get("requires_plugins") or []) if isinstance(meta.get("requires_plugins"), list) else []:
        if isinstance(item, str) and item.strip():
            plugins.append(PluginRequirement(id=item.strip()))
        elif isinstance(item, dict) and str(item.get("id") or "").strip():
            plugins.append(PluginRequirement(id=str(item["id"]).strip(),
                                             minimum_version=str(item["minimum_version"]).strip() if item.get("minimum_version") else None,
                                             required=item.get("required", True) is not False))
    names = {t.name for t in tools}
    return SystemSpec(
        hermes=_Hermes(minimum_version=hermes_version), platforms=[p for p in platforms if p],
        architectures=[a for a in archs if a], tools=tools, plugins=plugins[:64],
        runtime=_Runtime(shell=bool(names & {"shell", "terminal"}),
                         browser=bool(names & {"browser", "computer", "computer_use"}),
                         code=bool(names & {"code", "code_execution", "execute_code"})),
    )


@dataclass(frozen=True)
class Prepared:
    files: list[tuple[str, str, bytes]]
    content_hash: str
    description: str
    description_hash: str
    manifest_hash: str
    commit: str
    objects: ObjectSet


def prepare(source: Path, *, description: str, owner: str, installation_id: str, staging: Path) -> Prepared:
    """Validate a local skill, add a generated manifest when absent, and build the private commit
    the Gateway reconstructs. ``staging`` receives the exact bytes that were hashed."""
    files = read_source(source)
    if not any(p == "skill.manifest.json" for p, _, _ in files):
        from hermes_cli import __version__
        skill_md = next((b for p, _, b in files if p == "SKILL.md"), b"").decode("utf-8", "replace")
        generated = PackageManifest(name=source.name, requirements=infer_spec(skill_md, hermes_version=__version__))
        files.append(("skill.manifest.json", "file", manifest_bytes(generated)))
    chash = verify_files(files)
    raw_manifest = next(b for p, _, b in files if p == "skill.manifest.json")
    try:
        parse_manifest(raw_manifest)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PackageError(f"skill.manifest.json is invalid: {exc}") from exc
    clean = sanitize_description(description)
    for rel, _, body in files:
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    objects = ObjectSet()
    tree = build_tree(staging, objects, max_object_bytes=MAX_FILE_BYTES)
    commit = build_commit(tree, [], owner=owner, device=installation_id,
                          message="Collective Wisdom owner-private draft", objects=objects)
    return Prepared(files=files, content_hash=chash, description=clean,
                    description_hash=sha256_address(clean.encode("utf-8")),
                    manifest_hash=sha256_address(raw_manifest), commit=commit, objects=objects)
