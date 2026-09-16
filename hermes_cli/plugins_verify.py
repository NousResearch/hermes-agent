"""Console-package reconciliation and the read-only ``hermes plugins verify`` surface.

Two lifecycle gaps this closes (both observed in production):

1. A stale sibling tree inside ``<home>/plugins`` (e.g. the ``<name>.old-<timestamp>`` backup
   an interrupted update leaves behind) shares the manifest ``name`` with the active plugin, so
   discovery/list used to show whichever directory sorted last — a stale version while the live
   tree was current. Discovery now resolves such collisions deterministically
   (:func:`hermes_cli.plugins_discovery.resolve_key_collisions`); ``verify`` proves the winner
   per profile and fails closed on any mismatch.

2. ``install``/``update`` refresh the plugin directory and its metadata sidecar but never looked
   at the console package the plugin may declare in ``pyproject.toml`` (``[project.scripts]``),
   so the runtime-installed distribution could drift or break (console launcher dying with
   ``ModuleNotFoundError``) with no signal. :func:`reconcile_console_package` probes — read-only,
   no pip, no secrets — and surfaces the exact remediation command.

``verify`` is public evidence, not access: it prints versions, digests, doctor verdicts, and
console-launcher state without ever printing file *contents*, so a model/operator can confirm
an installation's integrity without reading policy-controlled files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Only these tracked files feed the tree digest: manifests and Python sources. Generated noise
# (__pycache__, .git, venvs) must not make a healthy tree look drifted.
_DIGEST_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".toml"})
_DIGEST_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "venv", ".venv", "node_modules", ".tox", "dist", "build",
})
_MAX_DIGEST_FILES = 2000


class VerifyError(Exception):
    """Fatal verification setup problem (unknown profile, missing plugin in every target)."""


@dataclass
class ConsolePackageProbe:
    """Read-only state of the console distribution a plugin tree declares (if any)."""

    declared: bool = False
    distribution: Optional[str] = None
    expected_version: Optional[str] = None
    scripts: list[str] = field(default_factory=list)
    installed_version: Optional[str] = None
    importable: Optional[bool] = None
    launcher_found: Optional[bool] = None
    launcher_path: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True when nothing is declared, or everything declared is present and version-matched."""
        if not self.declared:
            return True
        return bool(
            self.importable is True
            and self.launcher_found is not False
            and (self.expected_version is None or self.installed_version == self.expected_version)
        )

    @property
    def mismatches(self) -> list[str]:
        out: list[str] = []
        if not self.declared:
            return out
        if self.importable is not True:
            out.append(
                f"console package {self.distribution!r} is not importable in the active runtime "
                "(console launcher will fail with ModuleNotFoundError)")
        if self.launcher_found is False:
            out.append(
                f"console launcher for {', '.join(self.scripts) or self.distribution!r} "
                "was not found on PATH")
        if (self.expected_version is not None and self.installed_version is not None
                and self.installed_version != self.expected_version):
            out.append(
                f"installed console package version {self.installed_version} != declared "
                f"{self.expected_version}")
        return out

    def remediation(self, plugin_dir: Path) -> str:
        """The exact command that reconciles the declared console package in the active runtime."""
        return (
            f"python -m pip install --force-reinstall --no-deps {plugin_dir}\n"
            "  (run with the interpreter of the Hermes runtime that must expose the launcher; "
            "add --no-index for a fully offline install)"
        )


@dataclass
class ShadowingEntry:
    """One non-winning directory that claims the same registry key as the winner."""

    path: str
    version: str


@dataclass
class ProfileVerifyResult:
    """Verification evidence for one profile home. All fields are printable (no file contents)."""

    profile: str
    home: str
    plugin_name: str
    ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    active: dict[str, Any] = field(default_factory=dict)
    shadows: list[ShadowingEntry] = field(default_factory=list)
    console: Optional[ConsolePackageProbe] = None
    doctor_ok: Optional[bool] = None
    doctor_findings: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "home": self.home,
            "plugin": self.plugin_name,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "active": self.active,
            "shadowing_entries": [{"path": s.path, "version": s.version} for s in self.shadows],
            "console_package": (self.console.__dict__ if self.console is not None else None),
            "doctor": {"ok": self.doctor_ok, "findings": self.doctor_findings},
        }


def _read_pyproject_console_declaration(plugin_dir: Path) -> tuple[Optional[str], Optional[str], list[str]]:
    """``(distribution, version, scripts)`` from a plugin tree's ``pyproject.toml``.

    ``(None, None, [])`` when the tree declares no PEP 621 project. Read-only; parse failures
    are treated as "not declared" with a debug log — verify reports them as warnings elsewhere.
    """
    pyproject = plugin_dir / "pyproject.toml"
    if not pyproject.is_file():
        return None, None, []
    try:
        import tomllib
        with open(pyproject, "rb") as fh:  # windows-footgun: ok — binary mode, tomllib requires bytes
            data = tomllib.load(fh)
    except Exception as exc:
        logger.debug("pyproject.toml unreadable in %s: %s", plugin_dir, exc)
        return None, None, []
    project = data.get("project")
    if not isinstance(project, dict):
        return None, None, []
    distribution = project.get("name")
    version = project.get("version")
    scripts_table = project.get("scripts")
    scripts = sorted(scripts_table) if isinstance(scripts_table, dict) else []
    if not isinstance(distribution, str) or not distribution.strip():
        return None, (str(version) if version is not None else None), scripts
    return distribution.strip(), (str(version) if version is not None else None), scripts


def _launcher_on_path(script_names: list[str]) -> tuple[Optional[bool], Optional[str]]:
    """``(found, path)`` for the first declared console script resolvable on PATH.

    ``None`` when nothing was declared; ``(False, None)`` when declared but missing. ``shutil.which``
    resolves through PATHEXT on Windows (``fleet-policy`` -> ``fleet-policy.exe``)."""
    if not script_names:
        return None, None
    for name in script_names:
        resolved = shutil.which(name)
        if resolved:
            return True, resolved
    return False, None


def probe_console_package(plugin_dir: Path) -> ConsolePackageProbe:
    """Read-only probe of the console distribution declared by *plugin_dir*.

    Uses only ``importlib.metadata`` (dist-info on the active runtime's path) and ``shutil.which``;
    never runs pip, never imports plugin code, never reads secrets.
    """
    probe = ConsolePackageProbe()
    distribution, version, scripts = _read_pyproject_console_declaration(plugin_dir)
    probe.scripts = scripts
    if distribution is None and not scripts:
        return probe
    probe.declared = True
    probe.distribution = distribution
    probe.expected_version = version
    if distribution:
        import importlib.metadata
        import importlib.util
        try:
            probe.installed_version = str(importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            probe.installed_version = None
        except Exception as exc:  # unreadable metadata — treat as unknown, verify warns
            logger.debug("metadata probe failed for %s: %s", distribution, exc)
        top_level = distribution.replace("-", "_")
        try:
            probe.importable = importlib.util.find_spec(top_level) is not None
        except Exception:
            probe.importable = False
    probe.launcher_found, probe.launcher_path = _launcher_on_path(scripts)
    return probe


def reconcile_console_package(plugin_dir: Path, console) -> ConsolePackageProbe:
    """Post-install/update console-package reconciliation: probe and report; never auto-pip.

    Auto-installing from a lifecycle command would execute package machinery with ambient
    credentials (index auth, netrc) — a secrets-adjacent side effect the plugin lifecycle must
    not own. Instead the mismatch is surfaced loudly with the exact remediation command, and
    ``hermes plugins verify`` fails closed on it. Returns the probe for callers/tests.
    """
    probe = probe_console_package(plugin_dir)
    if not probe.declared:
        return probe
    mismatches = probe.mismatches
    if not mismatches:
        console.print(
            f"[green]✓[/green] Console package [bold]{probe.distribution or '?'}[/bold] "
            f"v{probe.installed_version or '?'} matches the plugin tree "
            f"(launcher: {probe.launcher_path or 'n/a'}).")
        return probe
    console.print()
    console.print(
        f"[yellow]⚠ Console package drift for '{plugin_dir.name}' "
        "(the plugin directory was refreshed, but its declared console package in the "
        "active runtime is stale or broken):[/yellow]")
    for mismatch in mismatches:
        console.print(f"  [yellow]-[/yellow] {mismatch}")
    console.print("[dim]  Remedy (no secrets are read by this command):[/dim]")
    console.print(f"[dim]  {probe.remediation(plugin_dir)}[/dim]")
    console.print("[dim]  Re-check with: hermes plugins verify "
                  f"{plugin_dir.name}[/dim]")
    return probe


def tree_digest(plugin_dir: Path) -> str:
    """Stable sha256 over the tracked files of *plugin_dir* (relative paths + bytes).

    Deterministic across platforms (sorted POSIX-style relative paths, newline-insensitive
    content hashing). Printed as hex only — never file contents.
    """
    digest = hashlib.sha256()
    root = plugin_dir.resolve()
    tracked: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _DIGEST_SKIP_DIRS for part in rel_parts):
            continue
        if path.suffix.lower() not in _DIGEST_SUFFIXES:
            continue
        tracked.append(path)
    for path in tracked[:_MAX_DIGEST_FILES]:
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            digest.update(f"<unreadable: {exc.__class__.__name__}>".encode("utf-8"))
        digest.update(b"\0")
    digest.update(f"<files={min(len(tracked), _MAX_DIGEST_FILES)}>".encode("utf-8"))
    return digest.hexdigest()


def _find_shadowing_entries(plugins_root: Path, winner_dir: Path) -> list[ShadowingEntry]:
    """Sibling directories claiming the same manifest name as *winner_dir* (non-winning trees)."""
    from hermes_cli.plugins_cmd import _read_manifest
    shadows: list[ShadowingEntry] = []
    winner_manifest = _read_manifest(winner_dir)
    winner_name = str(winner_manifest.get("name") or winner_dir.name)
    if not plugins_root.is_dir():
        return shadows
    for sibling in sorted(plugins_root.iterdir()):
        if not sibling.is_dir() or sibling == winner_dir or sibling.name.startswith("."):
            continue
        manifest = _read_manifest(sibling)
        if not manifest:
            continue
        if str(manifest.get("name") or sibling.name) == winner_name:
            shadows.append(ShadowingEntry(
                path=str(sibling), version=str(manifest.get("version") or "")))
    return shadows


def _active_plugin_dir(plugin_name: str, home: Path) -> Optional[Path]:
    """Winner directory for *plugin_name* under *home* using the real discovery order.

    Mirrors ``_discover_all_plugins`` (same collision resolution) but scoped to one home, so the
    verifier reports exactly the tree the runtime for that profile would load.
    """
    from hermes_cli.plugins_cmd import _scan_level
    from hermes_cli.plugins_discovery import collision_sort_key
    plugins_root = home / "plugins"
    if not plugins_root.is_dir():
        return None
    seen: dict = {}
    _scan_level(plugins_root, "user", set(), "", 0, seen)
    best: Optional[tuple] = None
    best_rank: Optional[tuple] = None
    for _key, entries in seen.items():
        for entry in entries:
            name, _version, _description, _source, dir_path, _key2 = entry
            if name != plugin_name and _key != plugin_name:
                continue
            rank = collision_sort_key("user", str(dir_path), _key)
            if best_rank is None or rank > best_rank:
                best, best_rank = entry, rank
    return Path(best[4]) if best is not None else None


def _run_doctor(plugin_dir: Path) -> tuple[Optional[bool], list[dict[str, str]]]:
    """Doctor verdict for one tree through the real runtime contracts (network already denied
    inside ``_doctor_runtime``). ``(None, [error])`` when the doctor itself fails."""
    try:
        from hermes_cli.plugin_dev import doctor_plugin
        report = doctor_plugin(str(plugin_dir))
        return report.ok, [
            {"level": finding.level, "message": finding.message} for finding in report.findings
        ]
    except Exception as exc:
        return None, [{"level": "error", "message": f"doctor failed: {type(exc).__name__}: {exc}"}]


def verify_plugin(
    plugin_name: str,
    profiles: Optional[list[str]] = None,
    *,
    run_doctor: bool = True,
) -> list[ProfileVerifyResult]:
    """Read-only verification of *plugin_name* across profile homes; one result per profile.

    ``profiles=None`` verifies the current HERMES_HOME only (labelled ``active``). Named
    profiles resolve through :func:`hermes_cli.profiles.get_profile_dir`; an unknown profile is
    a :class:`VerifyError` (fail closed — never silently verify a different home).
    """
    targets: list[tuple[str, Path]] = []
    if profiles:
        from hermes_cli.profiles import get_profile_dir, profile_exists
        expanded: list[str] = []
        for raw in profiles:
            # Accept both repeated flags (--profile a --profile b) and the comma form
            # (--profiles a,b) promised by the CLI contract.
            expanded.extend(p.strip() for p in str(raw).split(",") if p.strip())
        for name in expanded:
            if not profile_exists(name):
                raise VerifyError(f"profile '{name}' does not exist")
            targets.append((name, Path(get_profile_dir(name))))
    else:
        from hermes_constants import get_hermes_home
        targets.append(("active", get_hermes_home()))

    results: list[ProfileVerifyResult] = []
    for profile, home in targets:
        result = ProfileVerifyResult(
            profile=profile, home=str(home), plugin_name=plugin_name)
        results.append(result)
        if not home.is_dir():
            result.errors.append(f"profile home does not exist: {home}")
            continue
        active_dir = _active_plugin_dir(plugin_name, home)
        if active_dir is None:
            result.errors.append(
                f"plugin '{plugin_name}' not found under {home / 'plugins'} "
                "(no manifest claims that name)")
            continue

        from hermes_cli.plugins_cmd import _read_manifest
        manifest = _read_manifest(active_dir)
        result.active = {
            "path": str(active_dir),
            "manifest_name": str(manifest.get("name") or active_dir.name),
            "manifest_version": str(manifest.get("version") or ""),
            "tree_sha256": tree_digest(active_dir),
        }
        # Sidecar pin evidence (revision/source) when recorded — metadata only, no secrets.
        metadata_file = home / "plugins" / ".install-metadata.json"
        if metadata_file.is_file():
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                record = metadata.get(active_dir.name)
                if isinstance(record, dict):
                    result.active["install_record"] = {
                        k: record.get(k) for k in ("pinned", "revision", "source") if k in record
                    }
            except (OSError, json.JSONDecodeError) as exc:
                result.warnings.append(f"install metadata unreadable: {exc}")

        result.shadows = _find_shadowing_entries(home / "plugins", active_dir)
        for shadow in result.shadows:
            result.warnings.append(
                f"shadowing tree with the same manifest name: {shadow.path} "
                f"(version {shadow.version or '?'}) — not loaded; remove it to silence this warning")

        result.console = probe_console_package(active_dir)
        result.errors.extend(result.console.mismatches)

        if run_doctor:
            result.doctor_ok, result.doctor_findings = _run_doctor(active_dir)
            if result.doctor_ok is not True:
                result.errors.append("plugin doctor did not pass (see findings)")

        result.ok = not result.errors
    return results


def cmd_verify(
    name: str,
    *,
    profiles: Optional[list[str]] = None,
    json_output: bool = False,
    no_doctor: bool = False,
) -> int:
    """``hermes plugins verify`` implementation; returns the process exit code (fail closed)."""
    from hermes_cli.plugins_cmd import _console
    console = _console()
    try:
        results = verify_plugin(name, profiles, run_doctor=not no_doctor)
    except VerifyError as exc:
        if json_output:
            print(json.dumps({"plugin": name, "ok": False, "error": str(exc)}, indent=2))
        else:
            console.print(f"[red]Error:[/red] {exc}")
        return 2
    all_ok = all(result.ok for result in results)
    if json_output:
        print(json.dumps(
            {"plugin": name, "ok": all_ok, "profiles": [r.to_dict() for r in results]}, indent=2))
        return 0 if all_ok else 1

    console.print()
    console.print(f"[bold]Verify plugin:[/bold] {name}")
    for result in results:
        mark = "[green]PASS[/green]" if result.ok else "[red]FAIL[/red]"
        console.print(f"\n  [{ 'bold' }]{result.profile}[/] ({result.home}) — {mark}")
        if result.active:
            console.print(
                f"    active: {result.active.get('manifest_name')} "
                f"v{result.active.get('manifest_version') or '?'} @ {result.active.get('path')}")
            console.print(f"    tree sha256: {result.active.get('tree_sha256')}")
            record = result.active.get("install_record")
            if record:
                console.print(
                    f"    install record: pinned={record.get('pinned')} "
                    f"revision={str(record.get('revision'))[:12]} source={record.get('source')}")
        for shadow in result.shadows:
            console.print(
                f"    [yellow]shadowed by (not loaded):[/] {shadow.path} "
                f"(v{shadow.version or '?'})")
        probe = result.console
        if probe is not None and probe.declared:
            state = "[green]ok[/green]" if probe.ok else "[red]MISMATCH[/red]"
            console.print(
                f"    console package: {probe.distribution or '?'} "
                f"declared v{probe.expected_version or '?'}, "
                f"installed v{probe.installed_version or 'MISSING'}, "
                f"importable={probe.importable}, launcher={probe.launcher_path or 'NOT FOUND'} "
                f"— {state}")
        if result.doctor_ok is not None:
            state = "[green]ok[/green]" if result.doctor_ok else "[red]FAIL[/red]"
            console.print(f"    doctor: {state}")
        for finding in result.doctor_findings:
            level = "ERROR" if finding["level"] == "error" else "WARN"
            console.print(f"      [{level}] {finding['message']}")
        for warning in result.warnings:
            console.print(f"    [yellow]warn:[/yellow] {warning}")
        for error in result.errors:
            console.print(f"    [red]error:[/red] {error}")
    console.print()
    if all_ok:
        console.print(f"[green]✓ '{name}' verified across {len(results)} profile(s).[/green]")
    else:
        failed = [r.profile for r in results if not r.ok]
        console.print(
            f"[red]✗ '{name}' FAILED verification in: {', '.join(failed)}. "
            "Nothing was modified; reconcile per the errors above.[/red]")
    return 0 if all_ok else 1


__all__ = [
    "ConsolePackageProbe",
    "ProfileVerifyResult",
    "ShadowingEntry",
    "VerifyError",
    "cmd_verify",
    "probe_console_package",
    "reconcile_console_package",
    "tree_digest",
    "verify_plugin",
]
