"""Verified optional wheels for a Windows ARM64 dependency generation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tomllib

from pm.downloader import Download, DownloadError, DownloadTransportError, HashError, Source
from pm.package import InstallError


@dataclass(frozen=True)
class WheelSelection:
    available: tuple[str, ...]
    missing: tuple[str, ...]
    versions: dict[str, str]


def _verify_wheel_file(name: str, wheel: Path, sha: str) -> None:
    if wheel.is_symlink() or wheel.parent.is_symlink() or not wheel.is_file():
        raise InstallError("venv", f"native wheel is missing: {name}")
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha:
        raise HashError(f"native wheel digest mismatch: {name}")


def stage_wheels(project: Path, lock_path: Path, wheels_dir: Path,
                 source_for: Callable[[str, str, Path], Source], *, target: str) -> WheelSelection:
    """Stage reviewed bytes, leaving missing wheels to the registry-locked sdist."""
    if target != "win32-arm64":
        return WheelSelection((), (), {})
    document = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8-sig"))
    manifest = document.get("tool", {}).get("hermes", {}).get("win-arm64-wheels", {})
    if not manifest:
        return WheelSelection((), (), {})
    packages = tomllib.loads(lock_path.read_text(encoding="utf-8-sig"))["package"]
    available: list[str] = []
    missing: list[str] = []
    versions: dict[str, str] = {}
    wheels_dir.mkdir(parents=True, exist_ok=True)
    for name, row in sorted(manifest.items()):
        if not isinstance(row, dict) or set(row) != {"filename", "sha256"}:
            raise InstallError("venv", f"invalid wheel pin for {name}")
        filename, sha = row["filename"], row["sha256"]
        if (not isinstance(filename, str) or Path(filename).name != filename or "\\" in filename
                or not filename.endswith("-win_arm64.whl") or not isinstance(sha, str)
                or not re.fullmatch(r"[a-f0-9]{64}", sha)):
            raise InstallError("venv", f"invalid wheel filename or hash for {name}")
        parts = filename.split("-")
        if len(parts) not in (5, 6) or re.sub(r"[-_.]+", "-", parts[0]).lower() != name:
            raise InstallError("venv", f"wheel filename does not name {name}")
        registry = [package for package in packages if package["name"] == name
                    and "registry" in package["source"] and package["version"] == parts[1]]
        if len(registry) != 1 or not registry[0].get("sdist", {}).get("hash", "").startswith("sha256:"):
            raise InstallError("venv", f"wheel {name} has no matching locked source distribution")
        versions[name] = parts[1]
        candidate = source_for(filename, sha, wheels_dir / filename)
        try:
            Download([candidate], partials_dir=wheels_dir / ".partials").run()
        except DownloadTransportError as exc:
            if not exc.fallback_allowed:
                raise
            missing.append(name)
        except DownloadError as exc:
            if isinstance(exc, HashError) or not isinstance(exc.__cause__, DownloadTransportError):
                raise
            if not exc.__cause__.fallback_allowed:
                raise
            missing.append(name)
        else:
            available.append(name)
    return WheelSelection(tuple(available), tuple(missing), versions)


def relocate_wheels(root: Path, replay: Path, *, target: str) -> tuple[str, ...]:
    """Keep recorded wheel bytes and versions while rebinding generation-local paths."""
    if target != "win32-arm64":
        return ()
    lock_path = root / "uv.lock"
    text = lock_path.read_text(encoding="utf-8-sig")
    lock = tomllib.loads(text)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8-sig"))
    pins = project.get("tool", {}).get("hermes", {}).get("win-arm64-wheels", {})
    old_dir, new_dir = replay / "wheels", root / "wheels"
    local = [row for row in lock["package"]
             if "registry" in row["source"] and Path(row["source"]["registry"]) == old_dir]
    wanted: dict[tuple[str, str], str] = {}
    for row in local:
        name = row["name"]
        if name not in pins:
            raise InstallError("venv", f"unrecognized recorded wheel: {name}")
        filename, sha = pins[name]["filename"], pins[name]["sha256"]
        wheels = row.get("wheels", [])
        if (row["version"] != filename.split("-")[1] or len(wheels) != 1
                or set(wheels[0]) != {"path"} or Path(wheels[0]["path"]) != old_dir / filename):
            raise InstallError("venv", f"recorded wheel does not match the reviewed pin: {name}")
        _verify_wheel_file(name, new_dir / filename, sha)
        wanted[("registry", row["source"]["registry"])] = str(new_dir)
        wanted[("path", wheels[0]["path"])] = str(new_dir / filename)
    if local:
        # uv's Windows TOML escapes backslashes; match decoded path values,
        # then replace only their quoted tokens without rewriting the lock.
        pattern = re.compile(r"""(?m)\b(?P<key>registry|path)\s*=\s*(?P<token>"(?:\\.|[^"\\])*"|'[^']*')""")
        spans = []
        for match in pattern.finditer(text):
            value = tomllib.loads("value = " + match.group("token"))["value"]
            replacement = wanted.get((match.group("key"), value))
            if replacement is not None:
                spans.append((match.start("token"), match.end("token"),
                              json.dumps(replacement, ensure_ascii=False)))
        if len(spans) != 2 * len(local):
            raise InstallError("venv", "recorded wheel paths do not match the generation")
        for start, end, rendered in reversed(spans):
            text = text[:start] + rendered + text[end:]
        tomllib.loads(text)
        lock_path.write_text(text, encoding="utf-8")
    return tuple(sorted(set(pins) - {row["name"] for row in local}))


def verify_selected_wheels(source_lock: Path, resolved_lock: Path, wheels_dir: Path,
                           selection: WheelSelection) -> None:
    """An optional local wheel may replace bytes, never the locked version or sdist."""
    source_rows = tomllib.loads(source_lock.read_text(encoding="utf-8-sig"))["package"]
    resolved_rows = tomllib.loads(resolved_lock.read_text(encoding="utf-8-sig"))["package"]
    document = tomllib.loads((resolved_lock.parent / "pyproject.toml").read_text(encoding="utf-8-sig"))
    pins = document["tool"]["hermes"]["win-arm64-wheels"]
    for name, version in selection.versions.items():
        original = [row for row in source_rows if row["name"] == name and row["version"] == version
                    and "registry" in row["source"] and "sdist" in row]
        selected = [row for row in resolved_rows if row["name"] == name and row["version"] == version]
        if len(original) != 1 or len(selected) != 1:
            raise InstallError("venv", f"native dependency version drift: {name}")
        candidate = selected[0]
        if name in selection.available:
            filename, sha = pins[name]["filename"], pins[name]["sha256"]
            if (candidate["source"].get("registry") != str(wheels_dir)
                    or candidate.get("wheels") != [{"path": str(wheels_dir / filename)}]):
                raise InstallError("venv", f"native wheel does not match its reviewed pin: {name}")
            _verify_wheel_file(name, wheels_dir / filename, sha)
        elif candidate["source"] != original[0]["source"] or candidate.get("sdist") != original[0]["sdist"]:
            raise InstallError("venv", f"source distribution changed for {name}")
