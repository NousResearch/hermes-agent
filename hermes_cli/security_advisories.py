"""
Security advisory checker for Hermes Agent.

Detects known-compromised Python packages installed in the active venv
(supply-chain attacks like the Mini Shai-Hulud worm of May 2026 that
poisoned ``mistralai 2.4.6`` on PyPI) and surfaces remediation guidance to
the user.

Design goals:

- **Cheap.** A single ``importlib.metadata.version()`` call per advisory
  package. Safe to run on every CLI startup.
- **Loud when it matters, silent otherwise.** If no compromised package is
  installed, the user sees nothing.
- **Acknowledgeable.** Once the user has read and acted on an advisory they
  can dismiss it via ``hermes doctor --ack <id>``; the ack is persisted to
  ``config.security.acked_advisories`` and survives restart.
- **Extensible.** Adding a new advisory is one entry in ``ADVISORIES``;
  adding a new compromised version is a one-line edit. No code changes
  needed when the next worm hits.

The check is invoked from three places:

1. ``hermes doctor`` (and ``hermes doctor --ack <id>``)
2. CLI startup banner (one short line, then full guidance via
   ``hermes doctor``)
3. Gateway startup (logged to gateway.log; first interactive message gets
   a one-line operator banner)

This module is intentionally dependency-free beyond the stdlib so it can
run in environments where the rest of Hermes failed to import.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Advisory catalog
#
# Each advisory is a community-facing security warning about one or more
# specific package versions that are known to be compromised. To add a new
# advisory:
#
#   1. Append a new ``Advisory`` to ``ADVISORIES`` below
#   2. Set ``compromised`` to a tuple of ``(pkg_name, frozenset_of_versions)``
#      — version strings must match what ``importlib.metadata.version()``
#      returns. Use an empty frozenset to flag *any installed version*
#      (rare; only when the maintainer namespace itself is compromised).
#   3. Write 2-4 short ``remediation`` lines a non-expert can copy/paste.
#
# Do NOT remove old advisories. Once an advisory ships, leave it in place so
# users running an older release with the compromised package still get
# warned. Mark superseded ones via ``superseded_by`` if needed.
# =============================================================================


@dataclass(frozen=True)
class Advisory:
    """One security advisory entry.

    Attributes:
        id: stable identifier used for acks (e.g. ``shai-hulud-2026-05``).
            Lowercase-hyphen, never reused.
        title: one-line headline shown in banners.
        summary: 1-3 sentence description of what was compromised and how.
        url: reference URL (Socket advisory, GitHub advisory, PyPI page).
        compromised: tuple of ``(package_name, frozenset_of_versions)``
            pairs. Empty frozenset means "any version of this package is
            considered suspect" — use sparingly.
        remediation: ordered list of steps the user should take. First step
            should be the uninstall command; subsequent steps the credential
            audit / rotation guidance.
        published: ISO date string for sort order.
    """

    id: str
    title: str
    summary: str
    url: str
    compromised: tuple[tuple[str, frozenset[str]], ...]
    remediation: tuple[str, ...]
    published: str = ""
    severity: str = "high"  # low / medium / high / critical


ADVISORIES: tuple[Advisory, ...] = (
    Advisory(
        id="shai-hulud-2026-05",
        title="Mini Shai-Hulud worm — mistralai 2.4.6 compromised on PyPI",
        summary=(
            "PyPI quarantined the mistralai package on 2026-05-12 after a "
            "malicious 2.4.6 release. The worm steals credentials from "
            "environment variables and credential files (~/.npmrc, ~/.pypirc, "
            "~/.aws/credentials, GitHub PATs, cloud SDK tokens) and exfils "
            "them to a hardcoded webhook. If you ran any Python process that "
            "imported mistralai 2.4.6 — including hermes when configured "
            "with provider=mistral for TTS or STT — assume those credentials "
            "are exposed."
        ),
        url="https://socket.dev/blog/mini-shai-hulud-worm-pypi",
        compromised=(
            ("mistralai", frozenset({"2.4.6"})),
        ),
        remediation=(
            "Run: pip uninstall -y mistralai  (or: uv pip uninstall mistralai)",
            "Rotate API keys in ~/.hermes/.env (OpenRouter, Anthropic, OpenAI, "
            "Nous, GitHub, AWS, Google, Mistral, etc.).",
            "Audit ~/.npmrc, ~/.pypirc, ~/.aws/credentials, ~/.config/gh/hosts.yml, "
            "and any other credential files for tokens that may have been read.",
            "Check GitHub for unexpected new SSH keys, deploy keys, or webhook "
            "additions on repos you have admin on.",
            "After cleanup: hermes doctor --ack shai-hulud-2026-05  to dismiss "
            "this warning.",
        ),
        published="2026-05-12",
        severity="critical",
    ),
)


# =============================================================================
# Detection
# =============================================================================


@dataclass(frozen=True)
class AdvisoryHit:
    """One package-version match against an advisory."""

    advisory: Advisory
    package: str
    installed_version: str


def _installed_version(pkg_name: str) -> Optional[str]:
    """Return the installed version of ``pkg_name``, or None if not installed.

    Uses ``importlib.metadata`` so we don't depend on pip being importable
    inside the active venv (uv-created venvs may lack pip).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # py<3.8 — Hermes requires 3.10+ but defensive.
        return None
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return None
    except Exception:
        # Some metadata corruption modes raise ValueError or OSError. Don't
        # let advisory checking crash the CLI startup path.
        logger.debug("importlib.metadata.version(%s) raised", pkg_name, exc_info=True)
        return None


def detect_compromised(
    advisories: Iterable[Advisory] = ADVISORIES,
) -> list[AdvisoryHit]:
    """Scan installed packages and return all advisory hits.

    A "hit" means an advisory's listed package is installed AND the version
    is in the compromised set (or the compromised set is empty, meaning
    *any* version is suspect).
    """
    hits: list[AdvisoryHit] = []
    for advisory in advisories:
        for pkg_name, bad_versions in advisory.compromised:
            installed = _installed_version(pkg_name)
            if installed is None:
                continue
            if not bad_versions or installed in bad_versions:
                hits.append(AdvisoryHit(
                    advisory=advisory,
                    package=pkg_name,
                    installed_version=installed,
                ))
    return hits


# =============================================================================
# Acknowledgement persistence
#
# Acks live under ``security.acked_advisories`` in config.yaml as a list of
# advisory IDs. The list is the only state — no per-host data, no
# timestamps, no fingerprints. Users sharing a config.yaml across machines
# (rare but possible) get the same dismissal everywhere, which is the
# correct behavior for a global advisory.
# =============================================================================


def get_acked_ids() -> set[str]:
    """Return the set of advisory IDs the user has dismissed.

    Returns an empty set if config can't be loaded (don't block startup
    just because config is broken — the advisory will keep firing until
    config is repaired, which is fine).
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        logger.debug("Could not load config for advisory acks", exc_info=True)
        return set()
    sec = cfg.get("security") or {}
    raw = sec.get("acked_advisories") or []
    if not isinstance(raw, list):
        return set()
    return {str(x).strip() for x in raw if str(x).strip()}


def ack_advisory(advisory_id: str) -> bool:
    """Persist an ack for ``advisory_id``. Returns True on success.

    Idempotent — acking an already-acked ID is a no-op.
    """
    advisory_id = advisory_id.strip()
    if not advisory_id:
        return False
    try:
        from hermes_cli.config import load_config, save_config
    except Exception:
        logger.warning("Could not import config module to persist ack")
        return False
    try:
        cfg = load_config()
        sec = cfg.setdefault("security", {})
        existing = sec.get("acked_advisories") or []
        if not isinstance(existing, list):
            existing = []
        if advisory_id not in existing:
            existing.append(advisory_id)
            sec["acked_advisories"] = existing
            save_config(cfg)
        return True
    except Exception:
        logger.exception("Failed to persist advisory ack for %s", advisory_id)
        return False


def filter_unacked(hits: list[AdvisoryHit]) -> list[AdvisoryHit]:
    """Return only hits whose advisories the user has not dismissed."""
    if not hits:
        return []
    acked = get_acked_ids()
    return [h for h in hits if h.advisory.id not in acked]


# =============================================================================
# Rendering helpers
# =============================================================================


def _term_supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


def short_banner_lines(hits: list[AdvisoryHit]) -> list[str]:
    """Return 1-3 short lines suitable for a startup banner.

    Caller is responsible for color/styling. Always names the worst hit
    explicitly so the user knows what's wrong without running doctor.
    """
    if not hits:
        return []
    primary = hits[0]
    lines = [
        f"SECURITY ADVISORY [{primary.advisory.id}]: {primary.advisory.title}",
        f"  Detected: {primary.package}=={primary.installed_version}",
        "  Run 'hermes doctor' for remediation steps.",
    ]
    if len(hits) > 1:
        lines.insert(1, f"  ({len(hits) - 1} additional advisor"
                       f"{'ies' if len(hits) > 2 else 'y'} also active.)")
    return lines


def full_remediation_text(hit: AdvisoryHit) -> list[str]:
    """Return a multi-line block describing the advisory + remediation."""
    a = hit.advisory
    lines = [
        f"=== {a.title} ===",
        f"ID:        {a.id}    Severity: {a.severity}    Published: {a.published}",
        f"Detected:  {hit.package}=={hit.installed_version}",
        f"Reference: {a.url}",
        "",
        a.summary,
        "",
        "Remediation:",
    ]
    for i, step in enumerate(a.remediation, 1):
        lines.append(f"  {i}. {step}")
    return lines


# =============================================================================
# Startup-banner gating
#
# We do NOT want to hammer the user with the banner on every command. Once
# they've seen it inside a 24h window we cache that fact in
# ``~/.hermes/cache/advisory_banner_seen`` (a single line per advisory ID:
# ``<id> <iso8601_timestamp>``).
#
# Acked advisories never re-banner. Cached-but-not-acked advisories
# re-banner after 24h so the user doesn't fully forget.
# =============================================================================


_BANNER_CACHE_FILE = "advisory_banner_seen"
_BANNER_REPEAT_HOURS = 24


def _banner_cache_path() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home
        cache_dir = Path(get_hermes_home()) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / _BANNER_CACHE_FILE
    except Exception:
        return None


def _read_banner_cache() -> dict[str, float]:
    p = _banner_cache_path()
    if p is None or not p.exists():
        return {}
    out: dict[str, float] = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            advisory_id, ts = parts
            try:
                out[advisory_id] = float(ts)
            except ValueError:
                continue
    except Exception:
        return {}
    return out


def _write_banner_cache(seen: dict[str, float]) -> None:
    p = _banner_cache_path()
    if p is None:
        return
    try:
        lines = [f"{aid} {ts}" for aid, ts in seen.items()]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        logger.debug("Could not write advisory banner cache", exc_info=True)


def hits_due_for_banner(
    hits: list[AdvisoryHit],
    *,
    repeat_hours: int = _BANNER_REPEAT_HOURS,
) -> list[AdvisoryHit]:
    """Return only hits whose banner is due (not acked, not recently shown).

    Side effect: stamps the banner cache for any hit that's about to be
    shown. Callers should subsequently render the result.
    """
    import time

    fresh = filter_unacked(hits)
    if not fresh:
        return []
    now = time.time()
    cache = _read_banner_cache()
    cutoff = now - (repeat_hours * 3600)

    due: list[AdvisoryHit] = []
    for hit in fresh:
        last = cache.get(hit.advisory.id, 0.0)
        if last < cutoff:
            due.append(hit)
            cache[hit.advisory.id] = now
    if due:
        _write_banner_cache(cache)
    return due


# =============================================================================
# Public entry points used by doctor / CLI / gateway
# =============================================================================


def render_doctor_section(hits: list[AdvisoryHit]) -> tuple[bool, list[str]]:
    """Render the security-advisory section for ``hermes doctor``.

    Returns ``(has_problems, lines)``. Caller is responsible for printing
    with whatever color scheme it uses.
    """
    fresh = filter_unacked(hits)
    if not fresh:
        return False, ["No active security advisories.  ✓"]

    lines: list[str] = []
    for i, hit in enumerate(fresh):
        if i:
            lines.append("")
        lines.extend(full_remediation_text(hit))
    return True, lines


def startup_banner(hits: list[AdvisoryHit]) -> Optional[str]:
    """Return a printable startup banner, or None if nothing is due.

    Updates the banner cache as a side effect (so the next call within
    24h returns None for the same hit).
    """
    due = hits_due_for_banner(hits)
    if not due:
        return None
    lines = short_banner_lines(due)
    if _term_supports_color():
        red = "\x1b[1;31m"
        reset = "\x1b[0m"
        return red + "\n".join(lines) + reset
    return "\n".join(lines)


def gateway_log_message(hits: list[AdvisoryHit]) -> Optional[str]:
    """Return a one-line log message for gateway operators, or None."""
    fresh = filter_unacked(hits)
    if not fresh:
        return None
    if len(fresh) == 1:
        h = fresh[0]
        return (f"Security advisory [{h.advisory.id}] active: "
                f"{h.package}=={h.installed_version} matches {h.advisory.title}. "
                f"See {h.advisory.url}")
    return (f"{len(fresh)} security advisories active "
            f"(IDs: {', '.join(h.advisory.id for h in fresh)}). "
            f"Run `hermes doctor` on the gateway host for details.")
