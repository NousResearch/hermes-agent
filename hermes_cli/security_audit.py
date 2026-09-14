"""On-demand supply-chain audit for Hermes Agent installs.

Vulnerabilities are looked up against OSV.dev (``api.osv.dev/v1/querybatch`` + ``/v1/vulns/{id}``).
Single-shot, on-demand, never daily — see ``references/security-disclosure-triage.md``.

Durable lazy-install copies (``HERMES_LAZY_INSTALL_TARGET``) are classified
separately: the importable copy is the effective component; a shadowed lazy
copy is ``lazy-shadowed`` and does not trip ``--fail-on``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from hermes_constants import get_hermes_home

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{vid}"
OSV_BATCH_MAX = 1000  # OSV documented hard cap per request
HTTP_TIMEOUT = 20
DETAIL_PARALLELISM = 8

# Severity ordering for --fail-on gating. UNKNOWN sits below LOW so it never blocks.
SEVERITY_ORDER = {"UNKNOWN": 0, "LOW": 1, "MODERATE": 2, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


@dataclass(frozen=True)
class Component:
    """A single (name, version, ecosystem) tuple discovered on disk."""

    name: str
    version: str
    ecosystem: str  # "PyPI" | "npm" — exactly as OSV expects
    source: str    # "venv" | "lazy" | "lazy-shadowed" | "plugin:foo" | "mcp:bar"


@dataclass
class Vulnerability:
    osv_id: str
    severity: str = "UNKNOWN"
    summary: str = ""
    fixed_versions: list[str] = field(default_factory=list)


@dataclass
class Finding:
    component: Component
    vuln: Vulnerability


# Source labels for installed PyPI dists. ``venv`` is the sealed/core
# environment (anything not under the durable lazy target). ``lazy`` is a
# dist that lives in HERMES_LAZY_INSTALL_TARGET *and* would actually be
# imported (no earlier sys.path collision). ``lazy-shadowed`` is a stale
# copy in that target that cannot win imports because the target is
# append-only on sys.path — reported for cleanup, ignored by --fail-on.
SOURCE_VENV = "venv"
SOURCE_LAZY = "lazy"
SOURCE_LAZY_SHADOWED = "lazy-shadowed"

# Matches tools.lazy_deps._LAZY_TARGET_ENV. Read here directly so the
# audit module does not import the lazy-install machinery.
_LAZY_TARGET_ENV = "HERMES_LAZY_INSTALL_TARGET"


def _lazy_install_target_path() -> Optional[Path]:
    raw = os.environ.get(_LAZY_TARGET_ENV, "").strip()
    if not raw:
        return None
    try:
        return Path(raw).resolve()
    except OSError:
        return Path(raw)


def _is_under(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` or a descendant, after resolving."""
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        resolved_path, resolved_root = path, root
    if resolved_path == resolved_root:
        return True
    try:
        return resolved_path.is_relative_to(resolved_root)
    except (ValueError, TypeError):
        return False


def _dist_install_root(dist) -> Optional[Path]:
    """sys.path entry that owns this distribution (parent of dist-info)."""
    try:
        located = dist.locate_file("")
        if located is not None:
            return Path(str(located)).resolve()
    except Exception:
        pass
    path_attr = getattr(dist, "_path", None)
    if path_attr is None:
        return None
    try:
        p = Path(path_attr)
        name = p.name
        if name.endswith(".dist-info") or name.endswith(".egg-info"):
            p = p.parent
        return p.resolve()
    except Exception:
        return None


def _path_rank(install_root: Optional[Path], search_path: Sequence[str]) -> int:
    """Lower rank = earlier on sys.path = wins imports. Missing → last."""
    if install_root is None:
        return len(search_path)
    try:
        resolved = install_root.resolve()
    except OSError:
        resolved = install_root
    for i, entry in enumerate(search_path):
        if not entry:
            continue
        try:
            entry_path = Path(entry).resolve()
        except OSError:
            continue
        if resolved == entry_path or _is_under(resolved, entry_path):
            return i
    return len(search_path)


def _fail_on_applies(source: str) -> bool:
    """Shadowed durable-target copies cannot be imported; they don't fail the audit."""
    return source != SOURCE_LAZY_SHADOWED


def _classify_installed_distributions(
    dists: Iterable,
    *,
    search_path: Sequence[str],
    lazy_target: Optional[Path],
) -> list[Component]:
    """Turn raw distributions into Components using path + import precedence.

    For each normalized package name the copy whose install root appears
    earliest on ``search_path`` is the effective/runtime version. A later
    copy under ``lazy_target`` is kept as ``lazy-shadowed`` (different
    version only — same-version dupes are dropped). Other non-winning
    copies are skipped so they cannot be mislabeled as active ``venv``.
    """
    candidates: list[tuple[int, str, Component]] = []
    for dist in dists:
        try:
            name = (dist.metadata["Name"] or "").strip()
        except Exception:
            continue
        version = (dist.version or "").strip()
        if not name or not version:
            continue
        root = _dist_install_root(dist)
        if lazy_target is not None and root is not None and _is_under(root, lazy_target):
            source = SOURCE_LAZY
        else:
            source = SOURCE_VENV
        rank = _path_rank(root, search_path)
        candidates.append((rank, name.lower(), Component(
            name=name, version=version, ecosystem="PyPI", source=source,
        )))

    best_rank: dict[str, int] = {}
    for rank, nkey, _comp in candidates:
        prev = best_rank.get(nkey)
        if prev is None or rank < prev:
            best_rank[nkey] = rank

    # Process earlier-on-sys.path first so the effective copy of a given
    # (name, version) is recorded before a shadowed duplicate is considered.
    candidates.sort(key=lambda item: item[0])

    out: list[Component] = []
    seen_effective: set[tuple[str, str]] = set()
    seen_shadowed: set[tuple[str, str]] = set()
    for rank, nkey, comp in candidates:
        source = comp.source
        if source == SOURCE_LAZY and rank > best_rank[nkey]:
            source = SOURCE_LAZY_SHADOWED
        elif source == SOURCE_VENV and rank > best_rank[nkey]:
            # A non-lazy copy that loses import precedence is not the
            # running package; don't label it ``venv``.
            continue
        nv = (nkey, comp.version)
        if source == SOURCE_LAZY_SHADOWED:
            if nv in seen_effective or nv in seen_shadowed:
                continue
            seen_shadowed.add(nv)
        else:
            if nv in seen_effective:
                continue
            seen_effective.add(nv)
        if source != comp.source:
            comp = Component(
                name=comp.name,
                version=comp.version,
                ecosystem=comp.ecosystem,
                source=source,
            )
        out.append(comp)
    return out


def _discover_venv() -> list[Component]:
    """Every dist on the running Python's import path, classified by origin.

    The durable lazy-install target is appended at the end of ``sys.path``,
    so a stale copy there cannot shadow the sealed venv. That copy is
    reported as ``lazy-shadowed`` rather than ``venv``.
    """
    from importlib.metadata import distributions

    return _classify_installed_distributions(
        distributions(),
        search_path=sys.path,
        lazy_target=_lazy_install_target_path(),
    )


# ``name[extras]==version ; marker`` — an exact pin, optionally with extras and an environment marker.
_REQ_LINE = re.compile(r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?\s*==\s*(?P<version>[A-Za-z0-9._+!-]+)\s*(?:;.*)?$")


def _match_pins(specs: Iterable[str]) -> list[tuple[str, str]]:
    """``name==version`` pairs for every spec that is an exact pin; all others are skipped."""
    return [(m.group("name"), m.group("version")) for spec in specs if (m := _REQ_LINE.match(spec))]


def _parse_requirements(text: str) -> list[tuple[str, str]]:
    """Extract ``name==version`` pins. Loose specs (>=, ~=, no pin) are skipped: they can't map to
    a single OSV query, and false positives train users to ignore an audit tool's output.
    """
    lines = (raw.strip() for raw in text.splitlines())
    return _match_pins(line for line in lines if line and not line.startswith(("#", "-")))


def _parse_pyproject_pins(text: str) -> list[tuple[str, str]]:
    """Pull ``name==version`` pins from a ``pyproject.toml`` ``dependencies`` list."""
    try:
        import tomllib
        project = tomllib.loads(text).get("project") or {}
    except Exception:
        return []
    optional = project.get("optional-dependencies") or {}
    groups = [project.get("dependencies")] + (list(optional.values()) if isinstance(optional, dict) else [])
    return _match_pins(str(x) for group in groups if isinstance(group, list) for x in group)


_PLUGIN_PIN_FILES = (("requirements.txt", _parse_requirements), ("requirements-dev.txt", _parse_requirements),
                     ("pyproject.toml", _parse_pyproject_pins))


def _discover_plugins(hermes_home: Path) -> list[Component]:
    """Python deps declared by plugins under ``~/.hermes/plugins``. Plugins typically don't install
    into the venv, so their stated requirements are audit surface the venv scan misses.
    """
    plugins_dir = hermes_home / "plugins"
    if not plugins_dir.is_dir():
        return []
    out: list[Component] = []
    for plugin_dir in sorted(plugins_dir.iterdir()):
        if not plugin_dir.is_dir() or plugin_dir.name.startswith("."):
            continue
        for filename, parse in _PLUGIN_PIN_FILES:
            path = plugin_dir / filename
            try:
                pins = parse(path.read_text(encoding="utf-8", errors="replace")) if path.is_file() else []
            except OSError:
                continue
            out.extend(Component(name=n, version=v, ecosystem="PyPI", source=f"plugin:{plugin_dir.name}") for n, v in pins)
    return out


# Recognised pinned refs: ``npx [-y|--yes] [@scope/]pkg@1.2.3`` and ``uvx [--with] pkg==1.2.3``.
# Unversioned names map to "latest" at runtime and aren't a stable audit subject.
_NPX_PKG = re.compile(r"^(@[A-Za-z0-9._-]+/[A-Za-z0-9._-]+|[A-Za-z0-9._-]+)@([A-Za-z0-9._+-]+)$")
_UVX_PKG = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9._+!-]+)$")
# launcher basename -> (package-ref regex, OSV ecosystem)
_MCP_LAUNCHERS = {"npx": (_NPX_PKG, "npm"), "uvx": (_UVX_PKG, "PyPI")}


def _extract_mcp_component(server_name: str, command: str, args: list[str]) -> Optional[Component]:
    """Parse `command/args` into a Component, or None when the entry doesn't pin an auditable
    version (local paths, Docker images, unversioned npx, ...) — stay silent rather than guess.
    """
    cmd = (command or "").strip().lower()
    launcher = next((k for k in _MCP_LAUNCHERS if cmd.endswith(k)), None)  # any prefix path
    # Skip flag tokens; the first non-flag token must be a pinned ref or we stay silent.
    ref = next((token for token in args if not token.startswith("-")), None)
    if launcher is None or ref is None:
        return None
    pattern, ecosystem = _MCP_LAUNCHERS[launcher]
    m = pattern.match(ref)
    return m and Component(name=m.group(1), version=m.group(2), ecosystem=ecosystem, source=f"mcp:{server_name}")


def _discover_mcp() -> list[Component]:
    """Pinned MCP server packages from ``config.yaml``."""
    try:
        from hermes_cli.mcp_config import _get_mcp_servers
    except Exception:
        return []
    servers = _get_mcp_servers()
    if not isinstance(servers, dict):
        return []
    comps = (
        _extract_mcp_component(name, cfg.get("command", "") or "", [str(a) for a in cfg.get("args") or []])
        for name, cfg in servers.items()
        if isinstance(cfg, dict) and isinstance(cfg.get("args") or [], list)
    )
    return [c for c in comps if c]


_HTTP_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError)


def _http_json(url: str, payload: Optional[dict] = None) -> dict:
    """GET ``url`` (or POST ``payload`` as JSON when given) and decode the JSON body."""
    req = urllib.request.Request(url, method="GET") if payload is None else urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _osv_query_batch(components: list[Component]) -> dict[Component, list[str]]:
    """Return {component -> [osv_id, ...]} for components with any vulns."""
    findings: dict[Component, list[str]] = {}
    for chunk_start in range(0, len(components), OSV_BATCH_MAX):
        chunk = components[chunk_start:chunk_start + OSV_BATCH_MAX]
        payload = {"queries": [{"package": {"name": c.name, "ecosystem": c.ecosystem}, "version": c.version} for c in chunk]}
        try:
            resp = _http_json(OSV_BATCH_URL, payload)
        except _HTTP_ERRORS as exc:
            raise RuntimeError(f"OSV batch query failed: {exc}") from exc
        for comp, result in zip(chunk, resp.get("results") or []):
            if ids := [v.get("id") for v in (result or {}).get("vulns") or [] if v.get("id")]:
                findings[comp] = ids
    return findings


def _osv_severity_from_record(record: dict) -> str:
    """CVSS-derived severity tier from an OSV vuln record.

    Top-level ``severity`` holds CVSS vector strings we can't tier without a lib, so use the GHSA
    ``database_specific`` bucket first, then the per-affected ``ecosystem_specific`` one.
    """
    candidates = [(record.get("database_specific") or {}).get("severity")] + [
        (entry.get("ecosystem_specific") or {}).get("severity") for entry in record.get("affected") or []]
    tiers = (sev.strip().upper() for sev in candidates if isinstance(sev, str))
    return next((tier for tier in tiers if tier in SEVERITY_ORDER), "UNKNOWN")


def _osv_fixed_versions(record: dict) -> list[str]:
    fixes = [str(event["fixed"]) for entry in record.get("affected") or [] for rng in entry.get("ranges") or []
             for event in rng.get("events") or [] if "fixed" in event]
    return list(dict.fromkeys(fixes))  # dedupe, preserve order


def _osv_fetch_details(vuln_ids: Iterable[str]) -> dict[str, Vulnerability]:
    """Fetch summary/severity for each unique vuln id, in parallel."""
    def _fetch_one(vid: str) -> Vulnerability:
        try:
            rec = _http_json(OSV_VULN_URL.format(vid=vid))
        except _HTTP_ERRORS:
            return Vulnerability(osv_id=vid)
        return Vulnerability(vid, _osv_severity_from_record(rec), (rec.get("summary") or "").strip(), _osv_fixed_versions(rec))

    unique = sorted({vid for vid in vuln_ids if vid})
    if not unique:
        return {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=DETAIL_PARALLELISM) as pool:
        return {vuln.osv_id: vuln for vuln in pool.map(_fetch_one, unique)}


def _discover_components(
    *, skip_venv: bool = False, skip_plugins: bool = False, skip_mcp: bool = False, hermes_home: Optional[Path] = None
) -> list[Component]:
    """Discover all scannable components across the enabled sources."""
    home = hermes_home or Path(get_hermes_home())
    sources = ((skip_venv, _discover_venv), (skip_plugins, lambda: _discover_plugins(home)), (skip_mcp, _discover_mcp))
    return [c for skip, discover in sources if not skip for c in discover()]


def run_audit(*, components: Optional[list[Component]] = None, **discover_kwargs) -> list[Finding]:
    """Query OSV for ``components`` (or discover them with ``discover_kwargs`` when None; passing
    an already-discovered list avoids scanning the venv/plugins/MCP config a second time).
    """
    if components is None:
        components = _discover_components(**discover_kwargs)
    raw = _osv_query_batch(components) if components else {}
    if not raw:
        return []
    details = _osv_fetch_details(vid for ids in raw.values() for vid in ids)
    findings = [Finding(comp, details.get(vid) or Vulnerability(osv_id=vid)) for comp, ids in raw.items() for vid in ids]
    findings.sort(key=lambda f: (
        -SEVERITY_ORDER.get(f.vuln.severity, 0), f.component.source, f.component.name.lower(), f.vuln.osv_id
    ))
    return findings


def _render_human(findings: list[Finding], total_components: int) -> str:
    if not findings:
        return f"No known vulnerabilities found across {total_components} component(s)."

    lines = [f"Found {len(findings)} known vulnerability finding(s) across {total_components} component(s):", ""]
    last_source = None
    for f in findings:
        c, v = f.component, f.vuln
        if c.source != last_source:
            lines.append(f"[{c.source}]")
            last_source = c.source
        lines.append(f"  {v.severity.ljust(8)}  {c.name}=={c.version}  {v.osv_id}")
        if summary := v.summary:
            lines.append(f"           {summary if len(summary) <= 100 else summary[:97] + '...'}")
        if v.fixed_versions:
            lines.append(f"           fixed in: {', '.join(v.fixed_versions[:3])}")
    return "\n".join(lines)


def _render_json(findings: list[Finding], total_components: int) -> str:
    payload = {
        "total_components_scanned": total_components,
        "finding_count": len(findings),
        "findings": [{
            "package": f.component.name, "version": f.component.version,
            "ecosystem": f.component.ecosystem, "source": f.component.source,
            "vuln_id": f.vuln.osv_id, "severity": f.vuln.severity,
            "summary": f.vuln.summary, "fixed_versions": f.vuln.fixed_versions,
        } for f in findings],
    }
    return json.dumps(payload, indent=2)


def cmd_security_audit(args: argparse.Namespace) -> int:
    """Implementation of `hermes security audit`."""
    home = Path(get_hermes_home())
    output_json = bool(getattr(args, "json", False))
    fail_on = (getattr(args, "fail_on", None) or "critical").upper()
    if fail_on not in SEVERITY_ORDER:
        print(f"unknown --fail-on value: {fail_on.lower()} (choose from: low, moderate, high, critical)", file=sys.stderr)
        return 2

    skips = {k: bool(getattr(args, k, False)) for k in ("skip_venv", "skip_plugins", "skip_mcp")}
    components = _discover_components(hermes_home=home, **skips)
    total = len(components)
    if total == 0:
        print(json.dumps({"total_components_scanned": 0, "finding_count": 0, "findings": []}) if output_json
              else "No components discovered (everything skipped, or empty environment).")
        return 0

    try:
        findings = run_audit(hermes_home=home, components=components)
    except RuntimeError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2

    print((_render_json if output_json else _render_human)(findings, total))
    # Exit code: 1 iff any *importable* finding meets or exceeds --fail-on.
    # Shadowed lazy-target copies are reported in the output but cannot be
    # imported (append-only sys.path), so they must not keep the audit red
    # after the sealed venv is already patched.
    threshold = SEVERITY_ORDER[fail_on]
    return int(any(
        _fail_on_applies(f.component.source)
        and SEVERITY_ORDER.get(f.vuln.severity, 0) >= threshold
        for f in findings
    ))
