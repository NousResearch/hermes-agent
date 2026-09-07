#!/usr/bin/env python3
"""Assemble the standalone Windows installer without changing its script scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parent.parent
AUTHORING_ROOT = REPO_ROOT / "scripts" / "windows-installer"
OUTPUT = REPO_ROOT / "scripts" / "install.ps1"
LINE_LIMIT = 2000
INCLUDE = re.compile(r"# @include ([A-Za-z0-9][A-Za-z0-9._/-]*\.ps1)\n\Z")
WINDOWS_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)


class AssemblyError(ValueError):
    """The source graph cannot produce an unambiguous standalone artifact."""


def verify_history(base_ref: str, head_ref: str = "HEAD", repo_root: Path = REPO_ROOT) -> None:
    """Keep source debt monotonic across every manifest commit in a PR stack."""
    def git(*args: str) -> bytes:
        result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True)
        if result.returncode:
            raise AssemblyError(f"cannot verify installer history: {result.stderr.decode(errors='replace').strip()}")
        return result.stdout

    base = git("rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}").decode().strip()
    head = git("rev-parse", "--verify", "--end-of-options", f"{head_ref}^{{commit}}").decode().strip()
    base = git("merge-base", base, head).decode().strip()
    path = "scripts/windows-installer/manifest.json"
    # A path-filtered log can omit a merge that restores an older manifest.
    # Walk the complete DAG and check every parent edge carrying source debt.
    revisions = git("rev-list", "--reverse", "--topo-order", f"{base}..{head}").decode().splitlines()
    for revision in revisions:
        parents = git("rev-list", "--parents", "-n", "1", revision).decode().split()[1:]
        previous = []
        for parent in parents:
            if git("ls-tree", "--name-only", parent, "--", path).strip():
                previous.append(json.loads(git("show", f"{parent}:{path}"), object_pairs_hook=_object)["kill_track"])
        if not git("ls-tree", "--name-only", revision, "--", path).strip():
            if previous:
                raise AssemblyError(f"installer source ownership manifest was removed: {revision}")
            continue
        current = json.loads(git("show", f"{revision}:{path}"), object_pairs_hook=_object)
        ceilings = current["kill_track"]
        if not previous:
            original = git("show", f"{parents[0]}:scripts/install.ps1")
            source = git("show", f"{revision}:scripts/windows-installer/source/install.ps1")
            if source != original or ceilings != {"install.ps1": original.count(b"\n")}:
                raise AssemblyError("initial kill-track source must exactly preserve its parent installer")
        else:
            for prior in previous:
                if any(name not in prior or ceiling > prior[name] for name, ceiling in ceilings.items()):
                    raise AssemblyError(f"kill-track ceilings may only shrink on every parent edge: {revision}")


def _is_link(path: Path) -> bool:
    # Path.is_junction() only exists in Python 3.12+. lstat exposes the Windows
    # reparse attribute on the supported Python 3.11 baseline too.
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssemblyError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise AssemblyError("source paths must be nonempty strings")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(part.endswith(".") or WINDOWS_DEVICE.match(part) for part in value.split("/"))
        or path.suffix != ".ps1"
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value)
    ):
        raise AssemblyError(f"invalid source path: {value!r}")
    return value


def _plain_path(root: Path, name: str) -> Path:
    candidate = root / name
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise AssemblyError(f"source path escapes its root: {name}")
    parts = PurePosixPath(name).parts
    for checked in (root, *(root.joinpath(*parts[:i]) for i in range(1, len(parts) + 1))):
        if _is_link(checked):
            raise AssemblyError(f"source path contains a link: {name}")
    if not candidate.is_file():
        raise AssemblyError(f"source file is missing: {name}")
    return candidate


def _read_source(path: Path) -> str:
    text = path.read_bytes().decode("utf-8")
    if text.startswith("\ufeff"):
        raise AssemblyError(f"source must not contain a UTF-8 BOM: {path.name}")
    text = text.replace("\r\n", "\n")
    if "\r" in text or not text.endswith("\n"):
        raise AssemblyError(f"source requires LF/CRLF lines and a final newline: {path.name}")
    return text


def assemble(authoring_root: Path = AUTHORING_ROOT) -> bytes:
    """Validate and expand the declared source graph to canonical UTF-8/LF bytes."""
    manifest = json.loads(
        (authoring_root / "manifest.json").read_text(encoding="utf-8"),
        object_pairs_hook=_object,
    )
    required = {"version", "entry", "files", "kill_track"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise AssemblyError(f"manifest requires exactly these fields: {sorted(required)}")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise AssemblyError("unsupported manifest version")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise AssemblyError("manifest files must be a nonempty ordered list")
    names = [_name(value) for value in manifest["files"]]
    if len({name.casefold() for name in names}) != len(names):
        raise AssemblyError("duplicate or case-colliding manifest source paths")
    entry = _name(manifest["entry"])
    if entry not in names:
        raise AssemblyError("entry must be a declared source file")
    kill_track = manifest["kill_track"]
    if not isinstance(kill_track, dict) or not set(kill_track).issubset(names):
        raise AssemblyError("kill_track must map declared source paths to line ceilings")
    for name, ceiling in kill_track.items():
        if type(ceiling) is not int or ceiling <= LINE_LIMIT:
            raise AssemblyError(f"invalid kill-track ceiling: {name}")

    root = authoring_root / "source"
    sources = {}
    for name in names:
        text = _read_source(_plain_path(root, name))
        lines = text.count("\n")
        ceiling = kill_track.get(name, LINE_LIMIT)
        if lines > ceiling:
            raise AssemblyError(f"source exceeds its {ceiling}-line ceiling: {name} ({lines})")
        if name in kill_track and lines <= LINE_LIMIT:
            raise AssemblyError(f"remove completed source from kill_track: {name}")
        sources[name] = text

    # Everything in source/ is authored input. No extension-based size exemptions.
    inventory = set()
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if _is_link(path):
            raise AssemblyError(f"source inventory contains a link: {name}")
        if path.is_file():
            inventory.add(name)
    if inventory != set(names):
        raise AssemblyError(f"undeclared source files: {sorted(inventory - set(names))}")

    visited: list[str] = []
    active: list[str] = []

    def expand(name: str) -> str:
        if name in active:
            raise AssemblyError(f"source inclusion cycle: {' -> '.join([*active, name])}")
        if name in visited:
            raise AssemblyError(f"source included more than once: {name}")
        if name not in sources:
            raise AssemblyError(f"include is not declared in manifest: {name}")
        active.append(name)
        visited.append(name)
        output = []
        for line in sources[name].splitlines(keepends=True):
            if line.startswith("# @include"):
                match = INCLUDE.fullmatch(line)
                if match is None:
                    raise AssemblyError(f"malformed include directive in {name}: {line.rstrip()!r}")
                output.append(expand(_name(match[1])))
            else:
                output.append(line)
        active.pop()
        return "".join(output)

    result = expand(entry)
    if set(visited) != set(names):
        raise AssemblyError(f"orphaned declared sources: {sorted(set(names) - set(visited))}")
    if visited != names:
        raise AssemblyError("manifest files must follow source inclusion traversal order")
    return result.encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in artifact differs")
    parser.add_argument("--base-ref", help="verify monotonic source debt since this Git commit")
    parser.add_argument("--head-ref", default="HEAD", help="PR head for history verification (not its test merge)")
    args = parser.parse_args(argv)
    try:
        if Path(__file__).read_bytes().count(b"\n") > LINE_LIMIT:
            raise AssemblyError("installer assembler exceeds the authored source line ceiling")
        data = assemble(AUTHORING_ROOT)
        if args.base_ref:
            verify_history(args.base_ref, args.head_ref)
        if _is_link(OUTPUT):
            raise AssemblyError("installer output must not be a filesystem link")
        if args.check:
            if not OUTPUT.is_file() or OUTPUT.read_bytes() != data:
                raise AssemblyError("scripts/install.ps1 is stale; run python scripts/build_windows_installer.py")
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=OUTPUT.parent, prefix=".install-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                os.replace(temporary, OUTPUT)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        print(f"{'Verified' if args.check else 'Generated'} scripts/install.ps1: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
        return 0
    except (AssemblyError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"Windows installer assembly failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
