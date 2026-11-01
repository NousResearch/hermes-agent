"""TypeScript measurement: runs ``ts_units.mjs`` under node with the lockfile's typescript."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from scripts.code_health.gitio import git

_SCRIPT = Path(__file__).with_name("ts_units.mjs")


def pinned_typescript(repo: Path) -> str:
    lock = json.loads((repo / "package-lock.json").read_text(encoding="utf-8-sig"))
    return lock["packages"]["node_modules/typescript"]["version"]


def _version_at(pkg_dir: Path) -> str | None:
    try:
        return json.loads((pkg_dir / "package.json").read_text(encoding="utf-8-sig"))["version"]
    except (OSError, ValueError, KeyError):
        return None


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "hermes-code-health"


def _install(repo: Path, prefix: Path, pin: str) -> None:
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("code health needs node + npm to measure TypeScript files")
    prefix.mkdir(parents=True, exist_ok=True)
    # With --prefix, npm reads <prefix>/.npmrc as the project config and never the repo's, so
    # the repo's install policy (min-release-age, registry) is copied in to apply here too.
    npmrc = repo / ".npmrc"
    if npmrc.is_file():
        shutil.copyfile(npmrc, prefix / ".npmrc")
    else:
        (prefix / ".npmrc").unlink(missing_ok=True)
    what = f"npm install typescript@{pin} into {prefix}"
    try:
        subprocess.run(
            [npm, "install", "--no-save", "--no-package-lock", "--no-audit", "--no-fund",
             "--prefix", str(prefix), f"typescript@{pin}"],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=300, stdin=subprocess.DEVNULL,
        )
    # cli.main reports a RuntimeError as a tool failure (exit 2), never as a finding.
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{what} failed: {(exc.stderr or '').strip()[-2000:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{what} timed out after {exc.timeout:.0f}s") from exc


def resolve_typescript(repo: Path) -> Path:
    """Directory of the pinned ``typescript`` package, installing it into a cache if absent."""
    pin = pinned_typescript(repo)
    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    for root in (repo, common.parent):
        cand = root / "node_modules" / "typescript"
        if _version_at(cand) == pin:
            return cand
    prefix = _cache_root() / f"typescript-{pin}"
    cand = prefix / "node_modules" / "typescript"
    if _version_at(cand) != pin:
        _install(repo, prefix, pin)
    return cand


def measure_ts(repo: Path, root: Path, paths: list[str]) -> dict[str, dict]:
    if not paths:
        return {}
    node = shutil.which("node")
    if not node:
        raise RuntimeError("code health needs node to measure TypeScript files")
    ts_dir = resolve_typescript(repo)
    proc = subprocess.run(
        [node, str(_SCRIPT), str(ts_dir), str(root)],
        input=json.dumps(paths), capture_output=True, text=True, encoding="utf-8",
        timeout=600, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ts_units.mjs failed: {proc.stderr.strip()[:2000]}")
    return json.loads(proc.stdout)
