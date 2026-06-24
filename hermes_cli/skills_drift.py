"""Skill drift detection across per-profile skill copies.

A "drift" is the failure mode where the same skill name ships in multiple
profiles with **different SKILL.md content**.  The audit that motivated
this module found ``agent-handoff`` distributed across 13 profile copies
with 3 distinct content versions — the canonical case the detector must
catch.

The scanner walks ``~/.hermes/profiles/*/skills/*/SKILL.md`` (plus the
default ``~/.hermes/skills/`` root), MD5-hashes each file, and groups by
``skill name``.  Any group with more than one distinct hash is reported
as a finding.

Designed to be invoked:

1. **Explicitly** via ``hermes skills drift-check [--skill <name>] [--json]``
   from the CLI — exits non-zero on drift so CI can gate on it.
2. **At startup** via :func:`warn_on_skill_drift` — logs a warning to the
   regular logging surface so drifted installs still load but the
   operator sees the discrepancy in the boot transcript.

The MD5 is intentionally cheap and content-only — we want the detector
to run on every boot without measurable cost.  Two SKILL.md files with
identical hashes are, for drift purposes, identical.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from hermes_constants import display_hermes_home, get_default_hermes_root


logger = logging.getLogger(__name__)


# Env knob for tests / CI: point at a hermes root other than the platform
# default so we don't accidentally scan the user's real install.
DRIFT_ROOT_ENV = "HERMES_DRIFT_ROOT"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _resolve_root(root: Optional[Path]) -> Path:
    """Pick the Hermes root to scan.

    Precedence: explicit ``root`` arg > ``HERMES_DRIFT_ROOT`` env > default
    Hermes root.  Returning the default root is the right answer for the
    user-facing ``hermes skills drift-check`` command; tests pass an
    explicit ``root`` to a tmp dir.
    """
    if root is not None:
        return Path(root)
    env_root = os.environ.get(DRIFT_ROOT_ENV)
    if env_root:
        return Path(env_root)
    return get_default_hermes_root()


def _iter_skill_index_files(skills_dir: Path) -> Iterable[Path]:
    """Yield every ``SKILL.md`` under ``skills_dir``, skipping exclusions.

    Mirrors the conservative exclusion policy used by
    :mod:`agent.skill_utils` so we don't trip on hidden state directories.
    """
    from agent.skill_utils import is_excluded_skill_path

    if not skills_dir.is_dir():
        return
    try:
        for skill_md in skills_dir.rglob("SKILL.md"):
            try:
                if is_excluded_skill_path(skill_md):
                    continue
            except Exception:
                # If the exclusion predicate blows up, don't take down the
                # whole scan — just include the file (better over-report
                # than miss).
                pass
            yield skill_md
    except OSError as exc:
        logger.debug("drift scan: cannot walk %s: %s", skills_dir, exc)


def _candidate_skills_dirs(root: Path) -> List[Tuple[str, Path]]:
    """Return ``(profile_name, skills_dir)`` for every skills tree under root.

    Includes the default profile (``<root>/skills``) under the synthetic
    name ``"default"`` and one entry per named profile
    (``<root>/profiles/<name>/skills``).  Non-existent or non-directory
    entries are silently skipped — that mirrors what callers want when
    they invoke this on a partial install.
    """
    candidates: List[Tuple[str, Path]] = []

    default_skills = root / "skills"
    if default_skills.is_dir():
        candidates.append(("default", default_skills))

    profiles_root = root / "profiles"
    if profiles_root.is_dir():
        try:
            for entry in sorted(profiles_root.iterdir()):
                if not entry.is_dir():
                    continue
                pskills = entry / "skills"
                if pskills.is_dir():
                    candidates.append((entry.name, pskills))
        except OSError as exc:
            logger.debug("drift scan: cannot list %s: %s", profiles_root, exc)

    return candidates


def _md5_of_file(path: Path) -> str:
    """MD5 a file's bytes — small SKILL.md, read in one shot is fine."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _name_from_path(skill_md: Path) -> str:
    """Skill name = the parent directory of SKILL.md.

    This is the canonical name resolution for skills stored on disk —
    see :mod:`tools.skills_tool` for the same convention.  We deliberately
    do NOT parse YAML frontmatter here: the drift question is "is this
    file byte-for-byte the same?", not "do these two skills claim the
    same name in their frontmatter?".
    """
    return skill_md.parent.name


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def scan_skill_drift(
    root: Optional[Path] = None,
    skill_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a structured drift report for ``root`` (default Hermes root).

    Args:
        root: Hermes root to scan.  ``None`` = use the platform default.
        skill_filter: If set, only include this skill name in the report.
            Useful for ``--skill <name>`` scoping.

    Returns:
        Dict with shape::

            {
                "ok": bool,            # True iff no drift findings
                "root": str,           # abs path scanned (display form)
                "profiles_scanned": int,
                "skills_scanned": int,
                "findings": [
                    {
                        "name": str,
                        "distinct_versions": int,
                        "versions": [
                            {
                                "md5": str,
                                "profiles": [str, ...],
                                "first_path": str,
                            },
                            ...
                        ],
                    },
                    ...
                ],
                "scanned_skills": [
                    {"name": str, "profiles": [str, ...], "md5": str},
                    ...
                ],
            }

    The ``scanned_skills`` list is the full population (clean + drifted);
    ``findings`` is the subset that drifted.  Either can be empty.
    """
    root_path = _resolve_root(root)
    candidates = _candidate_skills_dirs(root_path)

    # Group (skill_name, md5) -> [profile_name, ...]
    # AND (skill_name) -> {md5 -> [profile_name, ...]}
    by_hash: Dict[str, Dict[str, List[str]]] = {}
    profiles_seen: set = set()
    skipped: List[Tuple[str, str]] = []  # (profile, path) — for diagnostics

    for profile_name, skills_dir in candidates:
        profiles_seen.add(profile_name)
        for skill_md in _iter_skill_index_files(skills_dir):
            name = _name_from_path(skill_md)
            if skill_filter and name != skill_filter:
                continue
            try:
                md5 = _md5_of_file(skill_md)
            except OSError as exc:
                logger.debug(
                    "drift scan: cannot read %s: %s", skill_md, exc
                )
                skipped.append((profile_name, str(skill_md)))
                continue
            by_hash.setdefault(name, {}).setdefault(md5, []).append(profile_name)

    findings: List[Dict[str, Any]] = []
    scanned: List[Dict[str, Any]] = []

    # Deterministic ordering — sort by skill name so output is stable for
    # diffing across runs (matters for --json CI consumers).
    for name in sorted(by_hash):
        versions = by_hash[name]
        profiles_for_skill = sorted({p for ps in versions.values() for p in ps})
        entry = {
            "name": name,
            "profiles": profiles_for_skill,
            "md5": _canonical_md5(versions),
        }
        scanned.append(entry)
        if len(versions) > 1:
            version_payload = []
            for md5 in sorted(versions):
                version_payload.append(
                    {
                        "md5": md5,
                        "profiles": sorted(versions[md5]),
                        "first_path": _first_path_for(
                            root_path, md5, versions[md5]
                        ),
                    }
                )
            findings.append(
                {
                    "name": name,
                    "distinct_versions": len(versions),
                    "versions": version_payload,
                }
            )

    return {
        "ok": not findings,
        "root": str(root_path),
        "profiles_scanned": len(profiles_seen),
        "skills_scanned": len(scanned),
        "findings": findings,
        "scanned_skills": scanned,
        "skipped_unreadable": [
            {"profile": p, "path": path} for p, path in skipped
        ],
    }


def _canonical_md5(versions: Dict[str, List[str]]) -> str:
    """Pick a single representative MD5 for a clean skill.

    Used for the ``scanned_skills`` list — drift-less skills have exactly
    one entry, so this just returns that key.  Split out so the contract
    is explicit when refactoring.
    """
    return next(iter(sorted(versions)))


def _first_path_for(
    root: Path, md5: str, profiles: List[str]
) -> str:
    """Best-effort: return the absolute path of one SKILL.md for a version.

    Used only inside the JSON ``findings`` block.  We don't need a real
    canonical path — just any one of the copies so operators can ``cat``
    it.  Profile-name order is stable because ``profiles`` is sorted.
    """
    for profile_name in profiles:
        if profile_name == "default":
            skills_dir = root / "skills"
        else:
            skills_dir = root / "profiles" / profile_name / "skills"
        candidate = skills_dir / _candidate_skill_dir_name(root, profile_name, profiles, md5)
        if candidate.exists():
            return str(candidate)
    return ""


def _candidate_skill_dir_name(
    root: Path, profile_name: str, profiles: List[str], md5: str
) -> str:
    """Find the skill-dir name under ``profile_name`` for the given md5.

    The drift report's ``first_path`` is informational; we resolve it by
    re-scanning that profile's skills dir for the matching MD5.  Cheap
    enough at this layer — drift reports are small.
    """
    if profile_name == "default":
        skills_dir = root / "skills"
    else:
        skills_dir = root / "profiles" / profile_name / "skills"
    if not skills_dir.is_dir():
        return ""
    for skill_md in _iter_skill_index_files(skills_dir):
        try:
            if _md5_of_file(skill_md) == md5:
                return skill_md.parent.name
        except OSError:
            continue
    return ""


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def format_human_report(report: Dict[str, Any]) -> str:
    """Pretty-print a drift report for terminal reading."""
    lines: List[str] = []
    if report["ok"]:
        lines.append(
            f"[bold green]No skill drift detected[/] "
            f"across {report['profiles_scanned']} profile(s) "
            f"and {report['skills_scanned']} skill(s)."
        )
        return "\n".join(lines) + "\n"

    lines.append(
        f"[bold red]Skill drift detected[/] "
        f"across {report['profiles_scanned']} profile(s), "
        f"{report['skills_scanned']} skill(s) scanned, "
        f"{len(report['findings'])} drifted."
    )
    lines.append("")

    for finding in report["findings"]:
        lines.append(f"[bold cyan]{finding['name']}[/] — "
                     f"{finding['distinct_versions']} distinct version(s):")
        for version in finding["versions"]:
            profiles_str = ", ".join(version["profiles"])
            lines.append(f"  [yellow]{version['md5']}[/]  {profiles_str}")
            if version.get("first_path"):
                lines.append(f"    [dim]{version['first_path']}[/]")
        lines.append("")

    return "\n".join(lines)


def format_json_report(report: Dict[str, Any]) -> str:
    """Render the report as JSON for machine consumers."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# CLI dispatch (called from hermes_cli/skills_hub.skills_command)
# ---------------------------------------------------------------------------


def run_drift_check(
    *,
    skill_filter: Optional[str] = None,
    as_json: bool = False,
    console: Optional[Console] = None,
    root: Optional[Path] = None,
) -> int:
    """CLI entry point.  Returns the process exit code."""
    c = console or Console()
    report = scan_skill_drift(root=root, skill_filter=skill_filter)
    if as_json:
        c.print(format_json_report(report))
    else:
        c.print(format_human_report(report))
    return 0 if report["ok"] else 1


# ---------------------------------------------------------------------------
# Startup-time warning (called from prompt_builder / system_prompt path)
# ---------------------------------------------------------------------------


def warn_on_skill_drift(
    *,
    root: Optional[Path] = None,
    skill_filter: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Log a warning if skill drift is present.  Never raises.

    Called at startup.  Drift is a *warning*, not an error — drifted
    installs must continue to load, otherwise existing users with the
    very problem we're trying to surface would silently lose their
    agent.  Returns the report (or ``None`` on clean / error) for
    callers that want to forward it into a structured log channel.
    """
    try:
        report = scan_skill_drift(root=root, skill_filter=skill_filter)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("drift scan raised at startup: %s", exc)
        return None
    if report["ok"]:
        return report
    names = ", ".join(f["name"] for f in report["findings"])
    logger.warning(
        "skill drift: %d skill(s) drifted across profiles "
        "(%s). Run `hermes skills drift-check` for the breakdown. "
        "Hermes will still load — fix drift to silence this warning.",
        len(report["findings"]),
        names,
    )
    return report


def display_hermes_home_safe() -> str:
    """Small helper so startup code can show the scan root on warning."""
    try:
        return display_hermes_home()
    except Exception:
        return "<hermes root>"