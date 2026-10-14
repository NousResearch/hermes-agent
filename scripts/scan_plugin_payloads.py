#!/usr/bin/env python3
"""Scan a pinned plugin tree for unscannable executable payloads (catalog rule 17).

A reviewer and the install scanner must be able to read everything shipped at the
pinned commit as source. This refuses:

  * compiled binaries, by magic header (ELF, Mach-O incl. universal, PE/``MZ``,
    WebAssembly) — never by extension, which a rename defeats;
  * committed Python bytecode (``.pyc``/``.pyo`` and sourceless caches), which the
    interpreter runs in place of the ``.py`` a reviewer read;
  * archives (zip/jar/whl, tar, gzip/xz/bzip2/zstd) whose members are not all
    themselves reviewable source — a tarball of ``.py``/``.md`` is fine, one hiding
    a binary is not.

stdlib only, so the no-install catalog validator and third-party CI can run it.
A native component is allowed only with a GitHub build attestation; that check
needs a token and lives in the workflow, not here.

Usage:
    scan_plugin_payloads.py <plugin-dir>
Exit 0 = clean, 1 = a payload was found (lines on stdout), 2 = bad usage.
"""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# Source extensions whose archives stay reviewable. Anything else inside an
# archive (a nested archive, a binary, an unknown blob) fails the whole archive.
_ARCHIVE_SOURCE_SUFFIXES = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".mjs", ".cjs", ".jsx", ".json", ".jsonc", ".jsonl",
    ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv",
    ".html", ".htm", ".css", ".scss", ".svg", ".sh", ".ps1", ".bat", ".sql", ".proto",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".rs", ".go", ".java", ".kt", ".rb", ".lua",
    ".patch", ".diff", ".lock", ".gitignore", ".gitattributes", ".env", ".service",
    ".xml", ".properties", ".gradle", ".cmake", ".mk", ".conf", ".d.ts", ".map",
}
# Inert, non-executable asset types that may ride along inside a source bundle:
# fonts and images carry no code the scanner needs to read. An unknown binary
# blob is NOT inert and still fails the archive.
_ARCHIVE_ASSET_SUFFIXES = {
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".avif",
}
# Extensionless filenames that are conventionally plain-text source/config.
_SOURCE_BASENAMES = {
    "dockerfile", "makefile", "license", "licence", "notice", "readme", "copying",
    "authors", "contributors", "changelog", "codeowners", "procfile", ".gitignore",
    ".gitattributes", ".dockerignore", ".npmignore", ".editorconfig",
}
_ARCHIVE_SUFFIXES = {".zip", ".jar", ".whl", ".egg", ".apk", ".war",
                     ".tar", ".tgz", ".gz", ".bz2", ".xz", ".zst", ".7z", ".rar"}
_BYTECODE_SUFFIXES = {".pyc", ".pyo"}


def _magic_kind(head: bytes) -> str | None:
    """Name the compiled-binary format a byte header indicates, else None."""
    if head.startswith(b"\x7fELF"):
        return "ELF executable/library"
    if head.startswith((b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
                        b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf")):
        return "Mach-O binary"
    if head.startswith(b"MZ"):
        return "PE (Windows) executable/DLL"
    if head.startswith(b"\x00asm"):
        return "WebAssembly module"
    return None


def _is_source_member(name: str) -> bool:
    low = name.lower().rstrip("/")
    if not low or low.endswith("/"):
        return True  # a directory entry
    base = low.rsplit("/", 1)[-1]
    if base in _SOURCE_BASENAMES:
        return True
    suffix = "." + base.rsplit(".", 1)[1] if "." in base else ""
    return suffix in _ARCHIVE_SOURCE_SUFFIXES or suffix in _ARCHIVE_ASSET_SUFFIXES


def _archive_problem(path: Path, data: bytes) -> str | None:
    """Return a reason the archive is unreviewable, or None if every member is source."""
    try:
        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                bad = [n for n in zf.namelist() if not _is_source_member(n)]
                return f"archive member not reviewable source: {bad[0]}" if bad else None
        bio = io.BytesIO(data)
        with tarfile.open(fileobj=bio) as tf:  # autodetects gzip/bz2/xz
            for member in tf.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    return f"archive has a non-regular member: {member.name}"
                if not _is_source_member(member.name):
                    return f"archive member not reviewable source: {member.name}"
                # A member named like source/asset must not actually be a compiled
                # binary (an .exe renamed .png rides in otherwise).
                fobj = tf.extractfile(member)
                if fobj is not None and _magic_kind(fobj.read(8)):
                    return f"archive member is a compiled binary: {member.name}"
            return None
    except tarfile.ReadError:
        return _single_stream_problem(path, data)
    except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError):
        # Not a parseable archive we recognise. A file carrying an archive
        # suffix we cannot open is itself unreviewable.
        return "archive could not be opened for review"
    return "archive could not be opened for review"


_SINGLE_STREAM = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}
# Only plain data may ship compressed. Code compressed out of sight would skip every
# text gate (self-updater, credential reads, reviewer greps), so it must be committed as-is.
_COMPRESSED_DATA_SUFFIXES = frozenset({".txt", ".tsv", ".csv", ".json", ".jsonl", ".ndjson"})
_COMPRESSED_TEXT_CAP = 64 * 1024 * 1024


def _single_stream_problem(path: Path, data: bytes) -> str | None:
    """A compressed single data file (``words.tsv.gz``) is reviewable when it
    decompresses to plain UTF-8 text under a data suffix."""
    opener = _SINGLE_STREAM.get(path.suffix.lower())
    if opener is None:
        return "archive could not be opened for review"
    inner = path.name[: -len(path.suffix)]
    if Path(inner).suffix.lower() not in _COMPRESSED_DATA_SUFFIXES:
        return f"compressed file is not plain data (commit code uncompressed): {inner}"
    try:
        with opener(io.BytesIO(data)) as stream:
            text = stream.read(_COMPRESSED_TEXT_CAP + 1)
    except (OSError, EOFError, lzma.LZMAError):
        return "archive could not be opened for review"
    if len(text) > _COMPRESSED_TEXT_CAP:
        return f"compressed file expands past {_COMPRESSED_TEXT_CAP >> 20} MiB: {inner}"
    if b"\0" in text:
        return f"compressed file is not text: {inner}"
    try:
        text.decode("utf-8")
    except UnicodeDecodeError:
        return f"compressed file is not UTF-8 text: {inner}"
    return None


def _file_problem(path: Path, rel: str) -> str | None:
    """Return a rule-17 reason this single file is unreviewable, or None."""
    if path.suffix.lower() in _BYTECODE_SUFFIXES:
        return f"{rel}: committed Python bytecode (ships instead of readable source)"
    try:
        head = path.read_bytes()[:8] if path.stat().st_size else b""
    except OSError:
        return None
    kind = _magic_kind(head)
    if kind:
        return f"{rel}: compiled binary — {kind} (not reviewable source)"
    if path.suffix.lower() in _ARCHIVE_SUFFIXES:
        try:
            reason = _archive_problem(path, path.read_bytes())
        except OSError:
            reason = "archive could not be read"
        if reason:
            return f"{rel}: {reason}"
    return None


def _shipped_files(plugin_dir: Path) -> list[Path]:
    """Files the pinned commit ships. In a git checkout that is the tracked set: the
    catalog gate runs `hermes plugins validate` (which imports the plugin) before this
    scan, so untracked `__pycache__/*.pyc` it wrote must not read as committed bytecode."""
    try:
        out = subprocess.run(["git", "-C", str(plugin_dir), "ls-files", "-z"],
                             capture_output=True, check=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        out = b""
    tracked = [plugin_dir / os.fsdecode(p) for p in out.split(b"\0") if p]
    # Not a checkout, or a tree nothing tracks (an untracked dir inside some other repo).
    return sorted(tracked) if tracked else sorted(plugin_dir.rglob("*"))


def scan(plugin_dir: Path) -> list[str]:
    findings: list[str] = []
    for path in _shipped_files(plugin_dir):
        if path.is_symlink() or not path.is_file():
            continue
        if "/.git/" in "/" + path.as_posix():
            continue
        problem = _file_problem(path, path.relative_to(plugin_dir).as_posix())
        if problem:
            findings.append(problem)
    return findings


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: scan_plugin_payloads.py <plugin-dir>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    findings = scan(root)
    for f in findings:
        print(f)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
