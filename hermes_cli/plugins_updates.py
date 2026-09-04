"""Plugin update checks — the standard, read-only 'is it outdated?' verb.

Address resolution per plugin, in order, no derivation ever (settled
2026-09-03, plugin-auto-update plan):
  1. saved sidecar tag + matching manifest update_url → fetch the feed yml
  2. manifest update_url differs from the saved tag (or appeared where
     none was saved) → NEEDS-FIXING: fetch refused, tag untouched
  3. no update_url anywhere + git row → git ls-remote vs revision
  4. neither → manual/unupdatable
Plus the pip world, stateless: entry-point discovery → installed version
vs PyPI latest. NEVER mutates anything — no pulls, no row writes, no
saved-tag changes.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli.plugins_provenance import (
    Provenance,
    ProvenanceClass,
    plugins_provenance,
    read_sidecar_rows,
)

_FETCH_TIMEOUT = 10.0
_MAX_FEED_BYTES = 1 * 1024 * 1024


@dataclass
class CheckResult:
    name: str
    klass: str                      # provenance class value ('git', ...)
    current: Optional[str] = None
    latest: Optional[str] = None
    update_available: Optional[bool] = None   # None = unknown/uncheckable
    needs_fixing: Optional[str] = None        # mismatch reason when set
    min_hermes: Optional[str] = None          # feed's version floor, if any
    reason: str = ""

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "class": self.klass,
            "current": self.current,
            "latest": self.latest,
            "update_available": self.update_available,
            "needs_fixing": self.needs_fixing,
            "min_hermes": self.min_hermes,
            "reason": self.reason,
        }


def _read_manifest_field(plugin_dir: Path, key: str) -> Optional[str]:
    """One field from the installed plugin.yaml (claims, not provenance)."""
    import yaml

    manifest = plugin_dir / "plugin.yaml"
    if not manifest.is_file():
        return None
    try:
        with manifest.open(encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def check_provenanced(
    prov: Provenance,
    *,
    fetch: Callable[[str], str],           # url -> text (raises on failure)
    ls_remote: Callable[[str], str],       # source -> HEAD sha (raises)
) -> CheckResult:
    """Check ONE sidecar-provenanced plugin per the resolution order."""
    result = CheckResult(name=prov.name, klass=prov.klass.value)

    if prov.klass is ProvenanceClass.MANUAL:
        result.reason = "no provenance; not auto-updatable"
        return result
    if prov.klass is ProvenanceClass.DRIFT:
        result.reason = (
            f"provenance drift — sidecar records {prov.row.get('source')!r} "
            "but the dir has no .git; reinstall from the recorded source"
        )
        return result
    if prov.klass is ProvenanceClass.SELF_CLONED:
        result.reason = "self-cloned; run `hermes plugins adopt` first"
        return result

    row = prov.row or {}
    result.current = row.get("revision") or None
    if row.get("pinned") is True:
        result.update_available = False
        result.reason = f"pinned @ {(result.current or '')[:12] or 'sha'}"
        return result

    # ── the saved-tag comparison (the security heart) ──────────────
    saved = row.get("update_url") or None
    claimed = _read_manifest_field(prov.path, "update_url")
    if saved is None and claimed is not None:
        # a pulled commit introduced a url where none was saved — same
        # threat class as a swap; never adopt silently
        result.needs_fixing = (
            f"manifest declares update_url {claimed!r} but no url was saved "
            "at install; run `hermes plugins trust-update-url` after review"
        )
        return result
    if saved is not None and claimed != saved:
        result.needs_fixing = (
            f"update_url mismatch: saved {saved!r}, manifest declares "
            f"{claimed!r}; run `hermes plugins trust-update-url` after review"
        )
        return result

    # ── 1. matching saved tag → fetch the feed ─────────────────────
    if saved is not None:
        try:
            feed_text = fetch(saved)
        except Exception as exc:
            result.reason = f"feed fetch failed: {exc}"
            return result
        try:
            feed = parse_feed_yml(feed_text)
        except ValueError as exc:
            result.reason = f"feed unparseable: {exc}"
            return result
        result.latest = feed.get("version")
        result.min_hermes = feed.get("min_hermes")
        result.update_available = (
            result.latest is not None and result.latest != result.current
        )
        return result

    # ── 3. no update_url anywhere + git row → ls-remote ────────────
    source = row.get("source") or ""
    if not source:
        result.reason = "no recorded source"
        return result
    try:
        head = ls_remote(source)
    except Exception as exc:
        result.reason = f"ls-remote failed: {exc}"
        return result
    result.latest = head
    result.update_available = bool(head) and head != result.current
    return result


def parse_feed_yml(text: str) -> dict:
    """The electron-updater-derived feed shape: version, released,
    min_hermes, artifacts{git,bundle,bundle_sha256}, notes_url."""
    import yaml

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("feed must be a YAML mapping")
    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("feed missing 'version'")
    out: dict[str, Any] = {"version": version.strip()}
    for key in ("min_hermes", "notes_url"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    artifacts = data.get("artifacts")
    if isinstance(artifacts, dict):
        out["artifacts"] = {
            k: v for k, v in artifacts.items() if isinstance(v, str)
        }
    return out


def check_pip_plugins(
    *,
    installed_version: Callable[[str], str],   # dist name -> version
    pypi_latest: Callable[[str], Optional[str]],  # dist name -> latest
    entry_points: Optional[list] = None,       # injectable for tests
) -> list[CheckResult]:
    """The pip world, stateless: entry-point dists vs PyPI. Nothing
    recorded, nothing to drift."""
    if entry_points is None:
        entry_points = list(
            importlib.metadata.entry_points().select(group="hermes_agent.plugins")
        )
    results: list[CheckResult] = []
    for ep in entry_points:
        dist_name = getattr(ep, "dist_name", None) or (
            ep.value.split(":")[0].split(".")[0] if ep.value else ep.name
        )
        try:
            current = installed_version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            results.append(
                CheckResult(
                    name=ep.name,
                    klass="pip",
                    reason=f"distribution {dist_name!r} not importable",
                )
            )
            continue
        latest = pypi_latest(dist_name)
        # None (unknown / not on PyPI) must read as unknown — not False
        if latest is None:
            results.append(
                CheckResult(
                    name=ep.name,
                    klass="pip",
                    current=current,
                    latest=None,
                    update_available=None,
                    reason="unknown (not on PyPI)",
                )
            )
            continue
        results.append(
            CheckResult(
                name=ep.name,
                klass="pip",
                current=current,
                latest=latest,
                update_available=latest != current,
            )
        )
    return results


def run_checks(
    plugins_dir: Path,
    *,
    fetch: Callable[[str], str],
    ls_remote: Callable[[str], str],
    include_pip: bool = True,
    pip_installed_version: Callable[[str], str] = importlib.metadata.version,
    pip_pypi_latest: Optional[Callable[[str], Optional[str]]] = None,
    pip_entry_points: Optional[list] = None,
) -> list[CheckResult]:
    """All checks for one plugins dir. NEVER mutates anything."""
    results = [
        check_provenanced(p, fetch=fetch, ls_remote=ls_remote)
        for p in plugins_provenance(plugins_dir)
    ]
    if include_pip:
        if pip_pypi_latest is None:
            pip_pypi_latest = _default_pypi_latest
        results.extend(
            check_pip_plugins(
                installed_version=pip_installed_version,
                pypi_latest=pip_pypi_latest,
                entry_points=pip_entry_points,
            )
        )
    return results


def _default_pypi_latest(dist: str) -> Optional[str]:
    """PyPI JSON API — the real fetcher (injectable in tests)."""
    import urllib.request

    url = f"https://pypi.org/pypi/{dist}/json"
    try:
        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read(_MAX_FEED_BYTES))
        return (data.get("info") or {}).get("version")
    except Exception:
        return None