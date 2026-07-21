"""On-demand supply-chain audit for Hermes Agent installs.

Scans three surfaces a Hermes user actually controls and we can map to
upstream advisories without auth or extra binaries:

1. The Hermes venv (every PyPI dist via ``importlib.metadata``).
2. Python deps declared by user-installed plugins under ``~/.hermes/plugins``
   (``requirements.txt`` + ``pyproject.toml`` best-effort pin extraction).
3. MCP servers wired in ``config.yaml`` whose ``command/args`` look like
   ``npx -y <pkg>@<ver>`` or ``uvx <pkg>==<ver>``.

Vulnerabilities are looked up against OSV.dev (``api.osv.dev/v1/querybatch``
+ ``/v1/vulns/{id}``). Single-shot, on-demand, never daily — see the design
notes in ``references/security-disclosure-triage.md``.

Out of scope on purpose: global pip/npm, editor/browser extensions,
daily background scans, auto-blocking installs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from hermes_constants import get_hermes_home

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{vid}"
OSV_BATCH_MAX = 1000  # OSV documented hard cap per request
HTTP_TIMEOUT = 20
DETAIL_PARALLELISM = 8

# Severity ordering for --fail-on gating. UNKNOWN sits below LOW so it
# never blocks unless --fail-on is passed something even lower (we don't
# expose that).
SEVERITY_ORDER = {
    "UNKNOWN": 0,
    "LOW": 1,
    "MODERATE": 2,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


# ─── Data shapes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Component:
    """A single (name, version, ecosystem) tuple discovered on disk."""

    name: str
    version: str
    ecosystem: str  # "PyPI" | "npm" — exactly as OSV expects
    source: str    # human-readable origin, e.g. "venv", "plugin:foo", "mcp:bar"


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


# ─── Component discovery ──────────────────────────────────────────────────────


def _discover_venv() -> list[Component]:
    """Every dist installed in the running Python's import path."""
    from importlib.metadata import distributions

    out: list[Component] = []
    seen: set[tuple[str, str]] = set()
    for dist in distributions():
        try:
            name = (dist.metadata["Name"] or "").strip()
        except Exception:
            continue
        version = (dist.version or "").strip()
        if not name or not version:
            continue
        key = (name.lower(), version)
        if key in seen:
            continue
        seen.add(key)
        out.append(Component(name=name, version=version, ecosystem="PyPI", source="venv"))
    return out


# requirements.txt line: drop comments, environment markers, options, extras
_REQ_LINE = re.compile(
    r"""^\s*
        (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)
        (?:\[[^\]]+\])?              # extras
        \s*==\s*
        (?P<version>[A-Za-z0-9._+!-]+)
        \s*(?:;.*)?$
    """,
    re.VERBOSE,
)


def _parse_requirements(text: str) -> list[tuple[str, str]]:
    """Extract ``name==version`` pins. Everything else (>=, ~=, no pin) is skipped.

    A loose pin can't be mapped to a single OSV query, and getting it wrong
    is worse than missing a finding for an audit tool — false positives
    train users to ignore output.
    """
    pins: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = _REQ_LINE.match(line)
        if m:
            pins.append((m.group("name"), m.group("version")))
    return pins


def _parse_pyproject_pins(text: str) -> list[tuple[str, str]]:
    """Pull ``name==version`` pins from a ``pyproject.toml`` ``dependencies`` list.

    Uses stdlib ``tomllib`` (3.11+). Same exact-pin policy as requirements.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover - 3.10 only
        return []
    try:
        data = tomllib.loads(text)
    except Exception:
        return []
    deps: list[str] = []
    project = data.get("project") or {}
    if isinstance(project.get("dependencies"), list):
        deps.extend(str(x) for x in project["dependencies"])
    optional = project.get("optional-dependencies") or {}
    if isinstance(optional, dict):
        for group in optional.values():
            if isinstance(group, list):
                deps.extend(str(x) for x in group)
    pins: list[tuple[str, str]] = []
    for dep in deps:
        m = _REQ_LINE.match(dep)
        if m:
            pins.append((m.group("name"), m.group("version")))
    return pins


def _discover_plugins(hermes_home: Path) -> list[Component]:
    """Python deps declared by plugins under ``~/.hermes/plugins``.

    Plugins typically don't install into the venv (they're directory-based
    with relative imports), so their stated requirements are useful audit
    surface even when the venv scan misses them.
    """
    plugins_dir = hermes_home / "plugins"
    if not plugins_dir.is_dir():
        return []

    out: list[Component] = []
    for plugin_dir in sorted(plugins_dir.iterdir()):
        if not plugin_dir.is_dir() or plugin_dir.name.startswith("."):
            continue
        source = f"plugin:{plugin_dir.name}"
        for req_file in ("requirements.txt", "requirements-dev.txt"):
            path = plugin_dir / req_file
            if path.is_file():
                try:
                    pins = _parse_requirements(path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
                for name, version in pins:
                    out.append(Component(name=name, version=version, ecosystem="PyPI", source=source))
        pyproject = plugin_dir / "pyproject.toml"
        if pyproject.is_file():
            try:
                pins = _parse_pyproject_pins(pyproject.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            for name, version in pins:
                out.append(Component(name=name, version=version, ecosystem="PyPI", source=source))
    return out


# npx forms we recognise:
#   npx -y @scope/pkg@1.2.3
#   npx --yes pkg@1.2.3
#   npx pkg@1.2.3 [...args]
# We deliberately don't try to resolve unversioned names — that maps to
# "latest" at runtime and isn't a stable audit subject.
_NPX_PKG = re.compile(r"^(@[A-Za-z0-9._-]+/[A-Za-z0-9._-]+|[A-Za-z0-9._-]+)@([A-Za-z0-9._+-]+)$")
# uvx forms:
#   uvx pkg==1.2.3
#   uvx --with pkg==1.2.3 entrypoint
_UVX_PKG = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9._+!-]+)$")


def _extract_mcp_component(server_name: str, command: str, args: list[str]) -> Optional[Component]:
    """Best-effort: parse `command/args` into a (name, version, ecosystem).

    Returns None when the entry doesn't pin a version we can audit (local
    paths, Docker images, unversioned npx, etc.). Audit output stays silent
    rather than guess.
    """
    cmd = (command or "").strip().lower()
    if not args:
        return None
    # npx (any prefix path)
    if cmd.endswith("npx") or cmd == "npx":
        # Skip flag tokens until we see the first thing that looks like a pkg ref
        for token in args:
            if token.startswith("-"):
                continue
            m = _NPX_PKG.match(token)
            if m:
                return Component(
                    name=m.group(1),
                    version=m.group(2),
                    ecosystem="npm",
                    source=f"mcp:{server_name}",
                )
            return None  # First non-flag token isn't a pinned ref
    # uvx (any prefix path)
    if cmd.endswith("uvx") or cmd == "uvx":
        for token in args:
            if token.startswith("-"):
                continue
            m = _UVX_PKG.match(token)
            if m:
                return Component(
                    name=m.group(1),
                    version=m.group(2),
                    ecosystem="PyPI",
                    source=f"mcp:{server_name}",
                )
            return None
    return None


def _discover_mcp() -> list[Component]:
    """Pinned MCP server packages from ``config.yaml``."""
    try:
        from hermes_cli.mcp_config import _get_mcp_servers
    except Exception:
        return []

    out: list[Component] = []
    servers = _get_mcp_servers()
    if not isinstance(servers, dict):
        return []
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        command = cfg.get("command", "") or ""
        args = cfg.get("args") or []
        if not isinstance(args, list):
            continue
        comp = _extract_mcp_component(name, command, [str(a) for a in args])
        if comp is not None:
            out.append(comp)
    return out


# ─── OSV client ───────────────────────────────────────────────────────────────


def _http_post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        body = resp.read()
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"OSV API returned non-JSON response ({resp.status}): {body[:200]!r}"
        ) from exc


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        body = resp.read()
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"OSV API returned non-JSON response ({resp.status}): {body[:200]!r}"
        ) from exc


def _osv_query_batch(components: list[Component]) -> dict[Component, list[str]]:
    """Return {component -> [osv_id, ...]} for components with any vulns.

    Components without findings are omitted from the result dict.
    """
    if not components:
        return {}
    findings: dict[Component, list[str]] = {}
    for chunk_start in range(0, len(components), OSV_BATCH_MAX):
        chunk = components[chunk_start:chunk_start + OSV_BATCH_MAX]
        payload = {
            "queries": [
                {
                    "package": {"name": c.name, "ecosystem": c.ecosystem},
                    "version": c.version,
                }
                for c in chunk
            ]
        }
        try:
            resp = _http_post_json(OSV_BATCH_URL, payload)
        except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError) as exc:
            raise RuntimeError(f"OSV batch query failed: {exc}") from exc
        results = resp.get("results") or []
        for comp, result in zip(chunk, results):
            vulns = (result or {}).get("vulns") or []
            ids = [v.get("id") for v in vulns if v.get("id")]
            if ids:
                findings[comp] = ids
    return findings


def _osv_severity_from_record(record: dict) -> str:
    """Extract CVSS-derived severity tier from an OSV vuln record."""
    # OSV puts CVSS in `severity` (top-level or per-affected) and a
    # human-readable bucket in `database_specific.severity` for GHSAs.
    db_specific = record.get("database_specific") or {}
    raw = db_specific.get("severity")
    if isinstance(raw, str) and raw.strip():
        upper = raw.strip().upper()
        if upper in SEVERITY_ORDER:
            return upper
    # Fall back to CVSS score → tier
    score: Optional[float] = None
    for sev_entry in record.get("severity") or []:
        s = sev_entry.get("score")
        if isinstance(s, str):
            # CVSS vector strings look like "CVSS:3.1/AV:N/..." — we can't
            # parse without a lib. Look for an explicit numeric in
            # affected[].ecosystem_specific later if present.
            continue
    affected = record.get("affected") or []
    for entry in affected:
        eco_spec = entry.get("ecosystem_specific") or {}
        sev = eco_spec.get("severity")
        if isinstance(sev, str) and sev.strip().upper() in SEVERITY_ORDER:
            return sev.strip().upper()
    if score is not None:
        if score >= 9.0:
            return "CRITICAL"
        if score >= 7.0:
            return "HIGH"
        if score >= 4.0:
            return "MODERATE"
        if score > 0:
            return "LOW"
    return "UNKNOWN"


def _osv_fixed_versions(record: dict) -> list[str]:
    fixes: list[str] = []
    for entry in record.get("affected") or []:
        for rng in entry.get("ranges") or []:
            for event in rng.get("events") or []:
                if "fixed" in event:
                    fixes.append(str(event["fixed"]))
    # Dedupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for f in fixes:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _osv_fetch_details(vuln_ids: Iterable[str]) -> dict[str, Vulnerability]:
    """Fetch summary/severity for each unique vuln id, in parallel."""
    unique = sorted({vid for vid in vuln_ids if vid})
    if not unique:
        return {}
    out: dict[str, Vulnerability] = {}

    def _fetch_one(vid: str) -> Vulnerability:
        try:
            rec = _http_get_json(OSV_VULN_URL.format(vid=vid))
        except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError):
            return Vulnerability(osv_id=vid)
        return Vulnerability(
            osv_id=vid,
            severity=_osv_severity_from_record(rec),
            summary=(rec.get("summary") or "").strip(),
            fixed_versions=_osv_fixed_versions(rec),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=DETAIL_PARALLELISM) as pool:
        for vuln in pool.map(_fetch_one, unique):
            out[vuln.osv_id] = vuln
    return out


# ─── Orchestration ────────────────────────────────────────────────────────────


def run_audit(
    *,
    skip_venv: bool = False,
    skip_plugins: bool = False,
    skip_mcp: bool = False,
    hermes_home: Optional[Path] = None,
) -> list[Finding]:
    """Discover components, query OSV, return findings sorted by severity desc."""
    home = hermes_home or Path(get_hermes_home())
    components: list[Component] = []
    if not skip_venv:
        components.extend(_discover_venv())
    if not skip_plugins:
        components.extend(_discover_plugins(home))
    if not skip_mcp:
        components.extend(_discover_mcp())

    if not components:
        return []

    raw = _osv_query_batch(components)
    if not raw:
        return []

    all_ids: list[str] = []
    for ids in raw.values():
        all_ids.extend(ids)
    details = _osv_fetch_details(all_ids)

    findings: list[Finding] = []
    for comp, ids in raw.items():
        for vid in ids:
            vuln = details.get(vid) or Vulnerability(osv_id=vid)
            findings.append(Finding(component=comp, vuln=vuln))

    findings.sort(
        key=lambda f: (
            -SEVERITY_ORDER.get(f.vuln.severity, 0),
            f.component.source,
            f.component.name.lower(),
            f.vuln.osv_id,
        )
    )
    return findings


# ─── Rendering ────────────────────────────────────────────────────────────────


def _render_human(findings: list[Finding], total_components: int) -> str:
    if not findings:
        return f"No known vulnerabilities found across {total_components} component(s)."

    lines: list[str] = []
    lines.append(
        f"Found {len(findings)} known vulnerability finding(s) "
        f"across {total_components} component(s):"
    )
    lines.append("")
    last_source = None
    for f in findings:
        if f.component.source != last_source:
            lines.append(f"[{f.component.source}]")
            last_source = f.component.source
        sev = f.vuln.severity.ljust(8)
        head = f"  {sev}  {f.component.name}=={f.component.version}  {f.vuln.osv_id}"
        lines.append(head)
        if f.vuln.summary:
            summary = f.vuln.summary
            if len(summary) > 100:
                summary = summary[:97] + "..."
            lines.append(f"           {summary}")
        if f.vuln.fixed_versions:
            lines.append(f"           fixed in: {', '.join(f.vuln.fixed_versions[:3])}")
    return "\n".join(lines)


def _render_json(findings: list[Finding], total_components: int) -> str:
    payload = {
        "total_components_scanned": total_components,
        "finding_count": len(findings),
        "findings": [
            {
                "package": f.component.name,
                "version": f.component.version,
                "ecosystem": f.component.ecosystem,
                "source": f.component.source,
                "vuln_id": f.vuln.osv_id,
                "severity": f.vuln.severity,
                "summary": f.vuln.summary,
                "fixed_versions": f.vuln.fixed_versions,
            }
            for f in findings
        ],
    }
    return json.dumps(payload, indent=2)


def _count_components(
    *, skip_venv: bool, skip_plugins: bool, skip_mcp: bool, hermes_home: Path
) -> int:
    total = 0
    if not skip_venv:
        total += len(_discover_venv())
    if not skip_plugins:
        total += len(_discover_plugins(hermes_home))
    if not skip_mcp:
        total += len(_discover_mcp())
    return total


# ─── CLI entrypoint ───────────────────────────────────────────────────────────


def cmd_security_audit(args: argparse.Namespace) -> int:
    """Implementation of `hermes security audit`."""
    home = Path(get_hermes_home())
    skip_venv = bool(getattr(args, "skip_venv", False))
    skip_plugins = bool(getattr(args, "skip_plugins", False))
    skip_mcp = bool(getattr(args, "skip_mcp", False))
    output_json = bool(getattr(args, "json", False))
    fail_on = (getattr(args, "fail_on", None) or "critical").upper()
    if fail_on not in SEVERITY_ORDER:
        print(
            f"unknown --fail-on value: {fail_on.lower()} "
            f"(choose from: low, moderate, high, critical)",
            file=sys.stderr,
        )
        return 2

    total = _count_components(
        skip_venv=skip_venv, skip_plugins=skip_plugins, skip_mcp=skip_mcp, hermes_home=home
    )
    if total == 0:
        msg = "No components discovered (everything skipped, or empty environment)."
        if output_json:
            print(json.dumps({"total_components_scanned": 0, "finding_count": 0, "findings": []}))
        else:
            print(msg)
        return 0

    try:
        findings = run_audit(
            skip_venv=skip_venv,
            skip_plugins=skip_plugins,
            skip_mcp=skip_mcp,
            hermes_home=home,
        )
    except RuntimeError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2

    if output_json:
        print(_render_json(findings, total))
    else:
        print(_render_human(findings, total))

    # Exit code: 1 iff any finding meets or exceeds the --fail-on threshold.
    threshold = SEVERITY_ORDER[fail_on]
    for f in findings:
        if SEVERITY_ORDER.get(f.vuln.severity, 0) >= threshold:
            return 1
    return 0
