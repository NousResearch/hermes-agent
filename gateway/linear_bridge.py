"""Linear -> Kanban bridge.

Mirrors Linear issues that carry an explicit ``agent:<profile>`` routing label
onto the kanban board, so Linear remains the reference plane while Kanban stays
the fleet's single execution-authority plane. This module is called from the
gateway's embedded dispatcher tick loop (``gateway/kanban_watchers.py``), which
means:

  * Only the dispatcher-LOCK-HOLDING gateway ever polls Linear — the exact
    same single-poller gating kanban dispatch itself relies on. A standby
    that takes over the lock starts polling on its next tick.
  * A bridge failure must never break dispatch: the caller wraps the tick in
    a broad try/except, and everything here is best-effort with loud logs.

``kanban.linear_bridge.dry_run`` defaults true and only reports cards it WOULD
create. When set false, mapped issues create Kanban cards via ``create_task``
with ``idempotency_key="linear:<issue-uuid>"`` so a retried poll cannot create
duplicates. Live mode has a safe default cap of one create per tick. The
optional ``issue_id_allowlist`` is Linear UUID-only and filters candidates; it
never bypasses the live cap. The seen-store remains a cheap poll-level skip,
while DB-level idempotency is the final safety rail.

Routing is reference-based, never assignee-based. Exactly one
``agent:<profile>`` label resolves to a real Hermes profile; no routing label
is skipped without error. Unknown, empty, or conflicting routing labels are
``UNROUTABLE`` and FAIL LOUD every poll, never silently dropped or marked seen.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger("gateway.linear_bridge")

LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

# GraphQL: read-only query for issues in the watched teams. Deliberately the
# ONLY Linear operation in this module — the bridge never mutates Linear.
_ISSUES_QUERY = """
query BridgeIssues($teamKeys: [String!]!, $after: String) {
  issues(
    first: 50
    after: $after
    filter: { team: { key: { in: $teamKeys } } }
  ) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      identifier
      title
      url
      priority
      state { name type }
      labels { nodes { name } }
      team { key }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Config / key resolution
# ---------------------------------------------------------------------------

def bridge_config() -> dict:
    """Return the ``kanban.linear_bridge`` config dict (defaults merged)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        bcfg = kcfg.get("linear_bridge", {})
        return bcfg if isinstance(bcfg, dict) else {}
    except Exception:
        return {}


def resolve_linear_api_key(bcfg: Optional[dict] = None) -> "tuple[Optional[str], str]":
    """Resolve the Linear API key as an INTENTIONAL dispatcher-role grant.

    Resolution order:
      1. the process environment (``api_key_env``, default LINEAR_API_KEY) —
         covers Doppler-injected service env;
      2. the shared ``~/.hermes/.env`` secrets store via
         ``hermes_cli.config.load_env()`` — covers EVERY gateway on the host,
         so a standby that wins the dispatcher lock after a failover can
         still poll. This is the fix for the survey finding that Linear
         access only worked because one specific gateway's process happened
         to inherit the key.

    Returns ``(key, source)`` where source is ``"env"`` / ``"hermes-env"`` /
    ``"missing"``. The key VALUE is never logged by any caller in this
    module; only the source label is.
    """
    bcfg = bcfg if bcfg is not None else bridge_config()
    env_name = str(bcfg.get("api_key_env") or "LINEAR_API_KEY")
    key = (os.environ.get(env_name) or "").strip()
    if key:
        return key, "env"
    try:
        from hermes_cli.config import load_env

        key = (load_env().get(env_name) or "").strip()
        if key:
            return key, "hermes-env"
    except Exception:
        logger.debug("linear bridge: ~/.hermes/.env resolution failed", exc_info=True)
    return None, "missing"


# ---------------------------------------------------------------------------
# Reference-label classification
# ---------------------------------------------------------------------------

def classify_linear_labels(
    label_names: "list[str]",
    bcfg: dict,
) -> "tuple[str, Optional[str], list[str]]":
    """Resolve an ``agent:<profile>`` Linear label to a Hermes profile.

    Returns ``(disposition, hermes_profile, routing_labels)`` where disposition
    is ``mapped``, ``unlabeled``, or ``unroutable``. A routing label is only
    valid when exactly one is present and its suffix names a real Hermes
    profile. Unknown and conflicting references stay unroutable every tick.
    """
    prefix = str(bcfg.get("routing_label_prefix") or "agent:").strip()
    if not prefix:
        prefix = "agent:"
    folded_prefix = prefix.casefold()
    routing_labels = [
        str(name).strip()
        for name in (label_names or [])
        if str(name).strip().casefold().startswith(folded_prefix)
    ]
    if not routing_labels:
        return "unlabeled", None, []
    if len(routing_labels) != 1:
        return "unroutable", None, routing_labels

    target = routing_labels[0][len(prefix):].strip().casefold()
    if not target:
        return "unroutable", None, routing_labels
    allowed_profiles_raw = bcfg.get("allowed_profiles")
    if allowed_profiles_raw:
        try:
            allowed_profiles = {
                str(profile).strip().casefold()
                for profile in allowed_profiles_raw
                if str(profile).strip()
            }
        except TypeError:
            return "unroutable", None, routing_labels
        if target not in allowed_profiles:
            return "unroutable", None, routing_labels
    try:
        from hermes_cli.kanban_db import classify_assignee

        if classify_assignee(target) != "profile":
            return "unroutable", None, routing_labels
    except Exception:
        logger.warning(
            "linear bridge: Hermes profile validation failed for routing label",
            exc_info=True,
        )
        return "unroutable", None, routing_labels
    return "mapped", target, routing_labels


# ---------------------------------------------------------------------------
# Seen-store (dedup on Linear issue id; atomic like the dispatcher heartbeat)
# ---------------------------------------------------------------------------

def seen_store_path() -> Path:
    from hermes_cli.kanban_db import kanban_home

    return kanban_home() / "kanban" / ".linear_bridge_seen.json"


def load_seen() -> "dict[str, dict]":
    try:
        raw = seen_store_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("linear bridge: seen-store unreadable; treating as empty", exc_info=True)
        return {}


def save_seen(seen: "dict[str, dict]") -> None:
    """Atomic write (tmp + os.replace) so a reader never sees a torn file."""
    try:
        path = seen_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(seen, indent=0, sort_keys=True), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception:
        logger.warning("linear bridge: seen-store write failed", exc_info=True)


def _task_body_for_linear_issue(issue: dict, routing_label: str) -> str:
    ident = str(issue.get("identifier") or issue.get("id") or "?")
    issue_uuid = _linear_issue_uuid(issue)
    title = str(issue.get("title") or "").strip()
    url = str(issue.get("url") or "").strip()
    state = (issue.get("state") or {}) if isinstance(issue.get("state"), dict) else {}
    state_name = str(state.get("name") or "").strip()
    state_type = str(state.get("type") or "").strip()
    team = (issue.get("team") or {}) if isinstance(issue.get("team"), dict) else {}
    team_key = str(team.get("key") or "").strip()
    priority = issue.get("priority")
    lines = [
        f"Linear: {ident}",
        f"Linear UUID: {issue_uuid}",
        f"Title: {title}" if title else None,
        f"URL: {url}" if url else None,
        f"Team: {team_key}" if team_key else None,
        f"State: {state_name} ({state_type})" if state_name or state_type else None,
        f"Priority: {priority}" if priority is not None else None,
        f"Routing label: {routing_label}",
        "",
        "This Kanban card was created automatically from Linear by the Linear -> Kanban bridge.",
    ]
    return "\n".join(line for line in lines if line is not None)


def _linear_issue_uuid(issue: dict) -> str:
    """Return the canonical Linear UUID for an issue, or fail loud."""
    raw_id = str(issue.get("id") or "").strip()
    if not raw_id:
        raise ValueError("missing Linear issue UUID")
    try:
        return str(UUID(raw_id))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Linear issue UUID: {raw_id!r}") from exc


def _linear_issue_idempotency_key(issue: dict) -> str:
    """Stable Kanban idempotency key for a Linear issue.

    Linear identifiers like ``BUI-123`` can change if an issue is moved between
    teams. The issue UUID is stable, so it is the DB-level duplicate guard.
    """
    return f"linear:{_linear_issue_uuid(issue)}"


def _parse_issue_id_allowlist(raw_allowlist: Any) -> "tuple[Optional[set[str]], Optional[str]]":
    if raw_allowlist in (None, ""):
        raw_allowlist = []
    if isinstance(raw_allowlist, str):
        allowlist_values = [item.strip() for item in raw_allowlist.split(",")]
    elif isinstance(raw_allowlist, (list, tuple, set)):
        allowlist_values = [str(item).strip() for item in raw_allowlist]
    else:
        return None, "invalid issue_id_allowlist"

    issue_id_allowlist: set[str] = set()
    for item in allowlist_values:
        if not item:
            continue
        try:
            issue_id_allowlist.add(str(UUID(item)))
        except (TypeError, ValueError):
            return None, "invalid issue_id_allowlist"
    return issue_id_allowlist, None


def _parse_live_create_cap(bcfg: dict) -> "tuple[Optional[int], Optional[str]]":
    if "max_creates_per_tick" not in bcfg:
        return 1, None
    raw_cap = bcfg.get("max_creates_per_tick")
    if isinstance(raw_cap, bool) or raw_cap is None:
        return None, "invalid max_creates_per_tick"
    if isinstance(raw_cap, int):
        cap = raw_cap
    elif isinstance(raw_cap, str):
        stripped = raw_cap.strip()
        if not stripped.isdecimal():
            return None, "invalid max_creates_per_tick"
        cap = int(stripped)
    else:
        return None, "invalid max_creates_per_tick"
    if cap < 1:
        return None, "invalid max_creates_per_tick"
    return cap, None


def _existing_kanban_task_id(conn: Any, idempotency_key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? "
        "AND status != 'archived' "
        "ORDER BY created_at DESC LIMIT 1",
        (idempotency_key,),
    ).fetchone()
    return str(row["id"]) if row else None


def _create_kanban_card_for_issue(
    issue: dict,
    *,
    assignee: str,
    routing_label: str,
    bcfg: dict,
) -> "tuple[str, bool]":
    """Create (or idempotently retrieve) the Kanban task for a Linear issue.

    Returns ``(task_id, duplicate)``. ``duplicate`` is true when the Kanban DB
    already had a non-archived task for this Linear UUID before this tick.
    """
    from hermes_cli import kanban_db as kb

    ident = str(issue.get("identifier") or issue.get("id") or "?")
    title = str(issue.get("title") or ident).strip() or ident
    idempotency_key = _linear_issue_idempotency_key(issue)
    board = bcfg.get("board")
    board = str(board).strip() if board else None
    conn = kb.connect(board=board)
    try:
        existing_task_id = _existing_kanban_task_id(conn, idempotency_key)
        if existing_task_id:
            return existing_task_id, True
        task_id = kb.create_task(
            conn,
            title=title,
            body=_task_body_for_linear_issue(issue, routing_label),
            assignee=assignee,
            created_by="linear_bridge",
            idempotency_key=idempotency_key,
            validate_assignee=True,
            board=board,
        )
        return task_id, False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Linear fetch (read-only)
# ---------------------------------------------------------------------------

def fetch_issues(api_key: str, team_keys: "list[str]") -> "list[dict]":
    """Fetch issues for the watched teams via GraphQL. READ-ONLY, paginated."""
    out: list[dict] = []
    after: Optional[str] = None
    for _page in range(20):  # hard cap: 1000 issues/poll
        payload = json.dumps({
            "query": _ISSUES_QUERY,
            "variables": {"teamKeys": list(team_keys), "after": after},
        }).encode("utf-8")
        req = urllib.request.Request(
            LINEAR_GRAPHQL_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": api_key,
                "User-Agent": "hermes-linear-bridge",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        if body.get("errors"):
            raise RuntimeError(f"Linear GraphQL errors: {body['errors']!r}")
        conn = (body.get("data") or {}).get("issues") or {}
        out.extend(conn.get("nodes") or [])
        page = conn.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        after = page.get("endCursor")
    return out


# ---------------------------------------------------------------------------
# The tick
# ---------------------------------------------------------------------------

def run_bridge_tick(
    bcfg: Optional[dict] = None,
    *,
    issues: "Optional[list[dict]]" = None,
    now: Optional[int] = None,
) -> dict:
    """Run one bridge poll and return a structured report.

    ``issues`` may be injected (tests / offline dry-runs against pre-fetched
    Linear data); otherwise they are fetched live with the resolved key.

    Report shape:
      {"ok": bool, "would_create": [...], "created": [...],
       "duplicates": [...], "unroutable": [...], "skipped_unlabeled": int,
       "skipped_status": int, "skipped_allowlist": int, "skipped_cap": int,
       "already_seen": int, "invalid_issue_ids": [...], "error": str|None}
    """
    bcfg = bcfg if bcfg is not None else bridge_config()
    dry_run = bool(bcfg.get("dry_run", True))
    now = int(now if now is not None else time.time())
    report: dict[str, Any] = {
        "ok": True, "would_create": [], "created": [], "duplicates": [],
        "unroutable": [], "skipped_unlabeled": 0, "skipped_status": 0,
        "skipped_allowlist": 0, "skipped_cap": 0, "already_seen": 0,
        "invalid_issue_ids": [], "disabled": False, "error": None,
    }

    if not bool(bcfg.get("enabled", False)):
        report["disabled"] = True
        return report

    issue_id_allowlist, allowlist_error = _parse_issue_id_allowlist(
        bcfg.get("issue_id_allowlist", [])
    )
    if allowlist_error is not None or issue_id_allowlist is None:
        report["ok"] = False
        report["error"] = allowlist_error or "invalid issue_id_allowlist"
        return report

    max_creates_per_tick: Optional[int] = None
    if not dry_run:
        max_creates_per_tick, cap_error = _parse_live_create_cap(bcfg)
        if cap_error is not None or max_creates_per_tick is None:
            report["ok"] = False
            report["error"] = cap_error or "invalid max_creates_per_tick"
            return report

    if issues is None:
        key, source = resolve_linear_api_key(bcfg)
        if not key:
            # FAIL LOUD: an armed bridge with no key must be visible — this
            # is exactly the silent-failover-break the key-resolution order
            # exists to prevent.
            logger.critical(
                "linear bridge: enabled but %s is set in neither the process "
                "env nor ~/.hermes/.env on this gateway — polling DISABLED "
                "here. Grant the key to the dispatcher role on every gateway "
                "host so failover cannot silently stop the bridge.",
                str(bcfg.get("api_key_env") or "LINEAR_API_KEY"),
            )
            report["ok"] = False
            report["error"] = "linear api key missing"
            return report
        logger.info("linear bridge: polling Linear (key source: %s)", source)
        try:
            issues = fetch_issues(key, list(bcfg.get("team_keys") or ["BUI"]))
        except Exception as exc:
            logger.warning("linear bridge: Linear fetch failed: %s", exc)
            report["ok"] = False
            report["error"] = f"fetch failed: {exc}"
            return report

    wanted_status = {str(s) for s in (bcfg.get("status_types") or ["unstarted"])}
    seen = load_seen()
    new_seen_entries: dict[str, dict] = {}

    for issue in issues:
        state_type = ((issue.get("state") or {}).get("type") or "").strip()
        if state_type not in wanted_status:
            report["skipped_status"] += 1
            continue
        label_nodes = ((issue.get("labels") or {}).get("nodes") or [])
        label_names = [
            str(label.get("name") or "")
            for label in label_nodes
            if isinstance(label, dict)
        ]
        disposition, hermes_assignee, routing_labels = classify_linear_labels(
            label_names, bcfg,
        )
        if disposition == "unlabeled":
            report["skipped_unlabeled"] += 1
            continue
        ident = str(issue.get("identifier") or issue.get("id") or "?")
        if disposition == "unroutable":
            entry = {
                "identifier": ident,
                "title": str(issue.get("title") or "")[:80],
                "routing_labels": routing_labels,
            }
            report["unroutable"].append(entry)
            logger.warning(
                "linear bridge UNROUTABLE: %s (%r) has routing label(s) %r, "
                "but they do not resolve to exactly one valid Hermes profile. "
                "It will NOT be bridged until the label is corrected. "
                "(fail-loud every poll)",
                ident, entry["title"], routing_labels,
            )
            continue
        try:
            issue_id = _linear_issue_uuid(issue)
        except ValueError as exc:
            entry = {
                "identifier": ident,
                "title": str(issue.get("title") or "")[:80],
                "error": str(exc),
            }
            report["invalid_issue_ids"].append(entry)
            report["ok"] = False
            msg = f"invalid Linear issue UUID for {ident}: {exc}"
            report["error"] = msg if not report["error"] else f"{report['error']}; {msg}"
            logger.warning(
                "linear bridge: %s has no valid Linear UUID; it will NOT be "
                "bridged or marked seen until Linear returns a stable UUID. "
                "(fail-loud every poll)",
                ident,
            )
            continue
        if issue_id_allowlist and issue_id not in issue_id_allowlist:
            report["skipped_allowlist"] += 1
            continue
        seen_entry = seen.get(issue_id)
        if dry_run:
            if seen_entry:
                report["already_seen"] += 1
                continue
        elif seen_entry and not bool(seen_entry.get("dry_run", False)):
            report["already_seen"] += 1
            continue
        idempotency_key = _linear_issue_idempotency_key(issue)
        card = {
            "identifier": ident,
            "linear_id": issue_id,
            "title": str(issue.get("title") or "")[:200],
            "url": issue.get("url"),
            "hermes_assignee": hermes_assignee,

            "routing_label": routing_labels[0],
            "state": (issue.get("state") or {}).get("name"),
            "priority": issue.get("priority"),
            "planned_idempotency_key": idempotency_key,
        }
        if dry_run:
            report["would_create"].append(card)
            new_seen_entries[issue_id] = {
                "identifier": ident,
                "first_seen": now,
                "dry_run": True,
            }
            logger.info(
                "linear bridge DRY-RUN: WOULD CREATE Kanban card: %s %r -> "
                "assignee=%s via label=%s (idempotency_key=%s). No card was created.",
                ident, card["title"][:60], hermes_assignee,
                card["routing_label"], idempotency_key,
            )
            continue

        if max_creates_per_tick is not None and len(report["created"]) >= max_creates_per_tick:
            report["skipped_cap"] += 1
            continue

        try:
            task_id, duplicate = _create_kanban_card_for_issue(
                issue,
                assignee=str(hermes_assignee),
                routing_label=routing_labels[0],
                bcfg=bcfg,
            )
        except Exception as exc:
            logger.warning(
                "linear bridge: failed to create Kanban card for %s: %s",
                ident, exc,
                exc_info=True,
            )
            report["ok"] = False
            msg = f"create failed for {ident}: {exc}"
            report["error"] = msg if not report["error"] else f"{report['error']}; {msg}"
            continue

        created_card = {
            "identifier": ident,
            "linear_issue_id": issue_id,
            "kanban_task_id": task_id,
            "title": card["title"],
            "hermes_assignee": hermes_assignee,
            "routing_label": routing_labels[0],
            "idempotency_key": idempotency_key,
        }
        if duplicate:
            report["duplicates"].append(created_card)
        else:
            report["created"].append(created_card)
        new_seen_entries[issue_id] = {
            "identifier": ident,
            "first_seen": (seen_entry or {}).get("first_seen", now),
            "bridged_at": now,
            "dry_run": False,
            "kanban_task_id": task_id,
            "idempotency_key": idempotency_key,
        }
        logger.info(
            "linear bridge: ensured Kanban card: %s %r -> task=%s assignee=%s "
            "via label=%s (idempotency_key=%s).",
            ident, card["title"][:60], task_id, hermes_assignee,
            routing_labels[0], idempotency_key,
        )

    if new_seen_entries:
        seen.update(new_seen_entries)
        save_seen(seen)
    return report
