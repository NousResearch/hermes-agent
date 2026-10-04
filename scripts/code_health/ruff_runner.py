"""Run the pinned ruff and split its output into per-function CC and pattern hits.

The pin lives in pyproject's ``dev`` dependency group (the version CI installs); a different
local ruff is never used silently, because rule behaviour and CC counts drift between releases.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from scripts.code_health.config import RUFF_CODES

_CC_MESSAGE = re.compile(r"is too complex \((\d+) >")
_CHUNK = 400


@dataclass
class RuffFile:
    cc_by_line: dict[int, int] = field(default_factory=dict)
    hits: list[tuple[str, int]] = field(default_factory=list)
    # Ruff could not parse the file (code-less diagnostics): its CC and hits are missing.
    errors: list[str] = field(default_factory=list)


def pinned_version(repo: Path) -> str:
    data = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8-sig"))
    for spec in data.get("dependency-groups", {}).get("dev", []):
        if isinstance(spec, str) and spec.startswith("ruff=="):
            return spec.split("==", 1)[1].strip()
    raise RuntimeError("pyproject.toml [dependency-groups].dev has no ruff==<version> pin")


def _version_of(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(
            [*cmd, "--version"], capture_output=True, text=True, timeout=60,
            stdin=subprocess.DEVNULL, check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    parts = out.split()
    return parts[1] if len(parts) >= 2 and parts[0] == "ruff" else None


def resolve_ruff(repo: Path) -> list[str]:
    """Command prefix for the pinned ruff: a matching install, else ``uvx ruff@<pin>``."""
    pin = pinned_version(repo)
    candidates = [Path(sys.prefix) / "bin" / "ruff", Path(sys.prefix) / "Scripts" / "ruff.exe"]
    on_path = shutil.which("ruff")
    if on_path:
        candidates.insert(0, Path(on_path))
    for cand in candidates:
        if cand.exists() and _version_of([str(cand)]) == pin:
            return [str(cand)]
    uvx = shutil.which("uvx")
    if uvx and _version_of([uvx, f"ruff@{pin}"]) == pin:
        return [uvx, f"ruff@{pin}"]
    found = _version_of(["ruff"]) if on_path else None
    raise RuntimeError(
        f"code health needs ruff {pin} (pyproject dev group); found {found or 'none'}. "
        f"Install it (`uv tool install ruff=={pin}`) or put uv on PATH so `uvx ruff@{pin}` works."
    )


def run_ruff(ruff: list[str], root: Path, paths: list[str]) -> dict[str, RuffFile]:
    """Ruff over ``paths`` (relative to ``root``), keyed by the same relative path."""
    results: dict[str, RuffFile] = {p: RuffFile() for p in paths}
    select = ",".join(("C901", *RUFF_CODES))
    for start in range(0, len(paths), _CHUNK):
        chunk = paths[start : start + _CHUNK]
        proc = subprocess.run(
            [*ruff, "check", "--isolated", "--ignore-noqa", "--no-cache", "--preview",
             "--target-version", "py311", "--select", select,
             "--config", "lint.mccabe.max-complexity=0",
             "--output-format", "json", "--exit-zero", *chunk],
            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=600,
            stdin=subprocess.DEVNULL, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ruff failed: {proc.stderr.strip()[:2000]}")
        for diag in json.loads(proc.stdout or "[]"):
            _record(results, root, diag)
    return results


def _record(results: dict[str, RuffFile], root: Path, diag: dict) -> None:
    rel = Path(diag["filename"]).resolve().relative_to(root.resolve()).as_posix()
    entry = results.setdefault(rel, RuffFile())
    row = diag["location"]["row"]
    code = diag.get("code")
    if code == "C901":
        match = _CC_MESSAGE.search(diag.get("message", ""))
        if match:
            entry.cc_by_line[row] = int(match.group(1))
    elif code:
        entry.hits.append((code, row))
    else:
        entry.errors.append(f"line {row}: {diag.get('message', 'syntax error')}")
