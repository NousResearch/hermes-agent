"""Security advisory checker for Hermes Agent.

Cheap (one ``importlib.metadata.version()`` call per advisory package, safe on every CLI startup)
and silent unless a compromised package is actually installed.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# Advisory catalog. To add one: append an ``Advisory`` to ``ADVISORIES`` with ``compromised`` as
# ``(pkg_name, frozenset_of_versions)`` pairs (versions exactly as ``importlib.metadata.version()``
# returns them; an empty frozenset flags ANY installed version — rare, namespace compromise only)
# and 2-4 copy/pasteable ``remediation`` lines. Never remove old advisories: users on an older
# release with the compromised package must still get warned.


@dataclass(frozen=True)
class Advisory:
    """``id`` is lowercase-hyphen, stable and never reused (it is what acks key on). ``remediation``
    is ordered: uninstall command first, then credential audit/rotation guidance.
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
            "are exposed. PyPI has since removed 2.4.6 and the project ships "
            "clean releases again (2.4.7, 2.4.8); this advisory only fires if "
            "the compromised 2.4.6 is still installed."
        ),
        url="https://socket.dev/blog/mini-shai-hulud-worm-pypi",
        compromised=(("mistralai", frozenset({"2.4.6"})),),
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


@dataclass(frozen=True)
class AdvisoryHit:
    """One package-version match against an advisory."""

    advisory: Advisory
    package: str
    installed_version: str


def _installed_version(pkg_name: str) -> Optional[str]:
    """Installed version of ``pkg_name`` via ``importlib.metadata`` (uv venvs may lack pip), or
    None if not installed or metadata is corrupt — never crash the CLI startup path.
    """
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return None
    except Exception:
        logger.debug("importlib.metadata.version(%s) raised", pkg_name, exc_info=True)
        return None


def detect_compromised(advisories: Iterable[Advisory] = ADVISORIES) -> list[AdvisoryHit]:
    """All hits: package installed AND version in the compromised set (or the set is empty)."""
    return [
        AdvisoryHit(advisory, pkg_name, installed)
        for advisory in advisories
        for pkg_name, bad_versions in advisory.compromised
        if (installed := _installed_version(pkg_name)) is not None and (not bad_versions or installed in bad_versions)
    ]


# ─── Acknowledgement persistence ──────────────────────────────────────────────
# Acks live under ``security.acked_advisories`` in config.yaml as a list of advisory IDs — the only
# state (no per-host data or timestamps), so a shared config.yaml dismisses everywhere.

def get_acked_ids() -> set[str]:
    """Advisory IDs the user has dismissed; empty when config can't be loaded (never block startup —
    the advisory keeps firing until config is repaired).
    """
    try:
        from hermes_cli.config import load_config
        raw = (load_config().get("security") or {}).get("acked_advisories") or []
    except Exception:
        logger.debug("Could not load config for advisory acks", exc_info=True)
        return set()
    return {str(x).strip() for x in raw if str(x).strip()} if isinstance(raw, list) else set()


def ack_advisory(advisory_id: str) -> bool:
    """Persist an ack for ``advisory_id``. Returns True on success."""
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
            sec["acked_advisories"] = existing + [advisory_id]
            save_config(cfg)
        return True
    except Exception:
        logger.exception("Failed to persist advisory ack for %s", advisory_id)
        return False


def filter_unacked(hits: list[AdvisoryHit]) -> list[AdvisoryHit]:
    """Only hits whose advisories the user has not dismissed."""
    acked = get_acked_ids() if hits else set()
    return [h for h in hits if h.advisory.id not in acked]


def _term_supports_color() -> bool:
    return not os.environ.get("NO_COLOR") and sys.stdout.isatty()


def short_banner_lines(hits: list[AdvisoryHit]) -> list[str]:
    """1-3 short unstyled lines for a startup banner; always names the worst hit explicitly."""
    if not hits:
        return []
    primary = hits[0]
    lines = [
        f"SECURITY ADVISORY [{primary.advisory.id}]: {primary.advisory.title}",
        f"  Detected: {primary.package}=={primary.installed_version}",
        "  Run 'hermes doctor' for remediation steps.",
    ]
    if len(hits) > 1:
        lines.insert(1, f"  ({len(hits) - 1} additional advisor{'ies' if len(hits) > 2 else 'y'} also active.)")
    return lines


def full_remediation_text(hit: AdvisoryHit) -> list[str]:
    """Multi-line block describing the advisory + remediation."""
    a = hit.advisory
    return [
        f"=== {a.title} ===",
        f"ID:        {a.id}    Severity: {a.severity}    Published: {a.published}",
        f"Detected:  {hit.package}=={hit.installed_version}",
        f"Reference: {a.url}",
        "",
        a.summary,
        "",
        "Remediation:",
        *(f"  {i}. {step}" for i, step in enumerate(a.remediation, 1)),
    ]


# ─── Startup-banner gating ────────────────────────────────────────────────────
# Once the banner is seen we cache that in ``~/.hermes/cache/advisory_banner_seen`` (one
# ``<id> <timestamp>`` line per advisory). Acked advisories never re-banner; cached-but-not-acked
# ones re-banner after 24h so the user doesn't fully forget.

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
    try:
        lines = p.read_text(encoding="utf-8").splitlines() if p is not None and p.exists() else []
    except Exception:
        return {}
    out: dict[str, float] = {}
    for parts in (line.split(None, 1) for line in lines):
        try:
            if len(parts) == 2:
                out[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return out


def _write_banner_cache(seen: dict[str, float]) -> None:
    try:
        if (p := _banner_cache_path()) is not None:
            p.write_text("\n".join(f"{aid} {ts}" for aid, ts in seen.items()) + "\n", encoding="utf-8")
    except Exception:
        logger.debug("Could not write advisory banner cache", exc_info=True)


def hits_due_for_banner(hits: list[AdvisoryHit], *, repeat_hours: int = _BANNER_REPEAT_HOURS) -> list[AdvisoryHit]:
    """Hits whose banner is due (not acked, not recently shown). Side effect: stamps the banner
    cache for every returned hit, so callers must render the result.
    """
    import time

    fresh = filter_unacked(hits)
    if not fresh:
        return []
    now = time.time()
    cache = _read_banner_cache()
    cutoff = now - (repeat_hours * 3600)
    due = [hit for hit in fresh if cache.get(hit.advisory.id, 0.0) < cutoff]
    for hit in due:
        cache[hit.advisory.id] = now
    if due:
        _write_banner_cache(cache)
    return due


def startup_banner(hits: list[AdvisoryHit]) -> Optional[str]:
    """Printable startup banner, or None if nothing is due (updates the banner cache)."""
    due = hits_due_for_banner(hits)
    if not due:
        return None
    text = "\n".join(short_banner_lines(due))
    return f"\x1b[1;31m{text}\x1b[0m" if _term_supports_color() else text


def gateway_log_message(hits: list[AdvisoryHit]) -> Optional[str]:
    """One-line log message for gateway operators, or None."""
    fresh = filter_unacked(hits)
    if not fresh:
        return None
    if len(fresh) == 1:
        h = fresh[0]
        return (f"Security advisory [{h.advisory.id}] active: {h.package}=={h.installed_version} "
                f"matches {h.advisory.title}. See {h.advisory.url}")
    return (f"{len(fresh)} security advisories active (IDs: {', '.join(h.advisory.id for h in fresh)}). "
            "Run `hermes doctor` on the gateway host for details.")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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
# ---- END PLUGIN-COMPAT ----
