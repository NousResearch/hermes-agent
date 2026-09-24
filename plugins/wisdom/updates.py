"""Update policy and local-edit conflicts for Wisdom-managed installations.

The Gateway assigns every installation an update mode — ``MANUAL`` (the user reviews each
version), ``AUTO_WITH_NOTICE`` (the org wants it applied, the user is told) or ``REQUIRED``
(the org mandates it). A managed tree whose bytes no longer match the hash the user installed
carries local edits: an automatic update never overwrites those silently. ``REQUIRED`` still
lands, with the edited copy parked (``service._swap_in``); ``AUTO_WITH_NOTICE`` turns into a
conflict the user resolves (replace, keeping a copy of the edits, or keep theirs). An unknown
or non-passing security verdict never counts as passing for an automatic apply.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODES = ("MANUAL", "AUTO_WITH_NOTICE", "REQUIRED")
SWEEP_INTERVAL = 600.0


def is_modified(entry: dict) -> bool:
    """True when the installed tree differs from the hash in the ledger (or is gone)."""
    from plugins.wisdom.service import _hash_tree
    path = Path(entry.get("path") or "")
    if not path.is_dir():
        return True
    return _hash_tree(path) != entry.get("content_hash")


def classify(row: dict, *, modified: bool, deferred_version: int | None) -> str:
    """``auto`` (policy applies it), ``conflict`` (edits block a policy update), ``deferred``
    (user said keep mine / not now for exactly this version) or ``manual``."""
    mode = row.get("mode") or "MANUAL"
    if deferred_version == row.get("latest"):
        return "deferred"
    if mode in ("AUTO_WITH_NOTICE", "REQUIRED") and (not modified or mode == "REQUIRED"):
        return "auto"
    if modified and mode != "MANUAL":
        return "conflict"
    return "manual"


def pending(svc) -> list[dict]:
    """Every available update with its policy verdict; the raw Gateway rows plus ``mode``,
    ``modified``, ``action``."""
    ledger = svc._ledger()
    deferred = svc.state.get("deferred_updates") or {}
    out = []
    for row in svc.status(include_paths=False)["updates"]:
        entry = ledger.get(row["skill_id"]) or {}
        modified = is_modified(entry)
        row = dict(row, modified=modified)  # ``mode`` is the Gateway's live policy from status()
        row["action"] = classify(row, modified=modified, deferred_version=deferred.get(row["skill_id"]))
        out.append(row)
    return out


def defer(state, skill_id: str, version: int) -> None:
    """Keep the local copy for this version; a newer version asks again."""
    deferred = dict(state.get("deferred_updates") or {})
    deferred[skill_id] = int(version)
    state.set("deferred_updates", deferred)


def _security_passed(_title: str, detail: str) -> bool:
    return any(line == "security: pass" or line.startswith("security: pass ") for line in detail.splitlines())


def apply_automatic(svc, rows: list[dict] | None = None) -> dict[str, list[dict]]:
    """Apply policy-authorized updates; report everything else for a human.

    Consent for an automatic update is the org policy the user saw when installing (the
    install plan names the update mode), so ``confirm`` only re-checks the plan's security
    verdict: anything but ``pass`` falls back to the manual queue."""
    from plugins.wisdom.service import NotConfirmed
    rows = pending(svc) if rows is None else rows
    report: dict[str, list[dict]] = {"applied": [], "conflicts": [], "manual": [], "failed": []}
    for row in rows:
        if row["action"] == "auto":
            try:
                result = svc.install(row["skill_id"], version=row["latest"], confirm=_security_passed)
            except NotConfirmed:
                report["manual"].append(dict(row, note="security verdict is not pass; review manually"))
                continue
            except Exception as exc:
                logger.warning("wisdom automatic update of %s failed: %s", row["slug"], exc)
                report["failed"].append(dict(row, error=str(exc)))
                continue
            report["applied"].append(dict(row, preserved_local_edits=result.get("preserved_local_edits")))
        elif row["action"] == "conflict":
            report["conflicts"].append(row)
        elif row["action"] == "manual":
            report["manual"].append(row)
    return report


def sweep(state, *, now: float | None = None, service=None) -> dict[str, list[dict]]:
    """Rate-limited policy pass: apply automatic updates, persist what needs a human as
    ``update_notices`` (read by the prompt section, the chat cards and the Desktop page)."""
    now = now or time.time()
    if not (state.get("installed") or {}) or now - state.get("updates_checked_at", 0) < SWEEP_INTERVAL:
        return {"applied": [], "conflicts": state.get("update_notices", {}).get("conflicts", []),
                "manual": state.get("update_notices", {}).get("manual", []), "failed": []}
    state.set("updates_checked_at", now)
    if service is None:
        from plugins.wisdom.service import Wisdom
        service = Wisdom(state)
    report = apply_automatic(service)
    state.set("update_notices", {"conflicts": report["conflicts"], "manual": report["manual"],
                                 "applied": report["applied"], "at": now})
    return report


def summary_lines(report: dict[str, Any]) -> list[str]:
    lines = []
    for r in report.get("applied") or []:
        note = " (your edited copy was kept aside)" if r.get("preserved_local_edits") else ""
        lines.append(f"updated {r['slug']} v{r['installed']} → v{r['latest']} ({r['mode']}){note}")
    for r in report.get("conflicts") or []:
        lines.append(f"{r['slug']} v{r['latest']} is available ({r['mode']}) but you edited your copy; "
                     f"resolve with `wisdom update {r['slug']} --replace` or `--keep`")
    for r in report.get("manual") or []:
        lines.append(f"{r['slug']} v{r['installed']} → v{r['latest']} available" + (" (REQUIRED)" if r.get("required") else ""))
    return lines
