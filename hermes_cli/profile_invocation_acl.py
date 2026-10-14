"""Optional application-level admission for agent-originated profile dispatch.

Configure on the install root's config.yaml::

    bot_mode:
      invocation_acl:
        forge: [forge, forge-worker, forge-reviewer]
        forge-worker: [forge, forge-worker, forge-reviewer]

Keys are target profile ids; values are source profile ids. An absent key keeps
legacy behavior. A malformed configured ACL denies all agent-originated targets;
unknown source identities never satisfy a configured target. This does not guard
operator shell access or direct filesystem writes by the same Unix account.
"""
from __future__ import annotations

import json
from pathlib import Path

from hermes_constants import PROFILE_ID_RE

# Application-level provenance, not protection from a process that can modify
# this user's SQLite database or execute arbitrary Python as the same UID.
NATIVE_KANBAN_ADMISSION = "native_kanban_admission"


def admitted_kanban_source(conn, task_id: str, *, target: str, lane: str) -> str | None:
    """Only a native admission for this target and *current* handoff counts."""
    latest_review = None
    if lane == "review":
        row = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
            "ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        if row is None:
            return None
        latest_review = row["id"]
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC", (task_id, NATIVE_KANBAN_ADMISSION),
    )
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("origin") != "native_kanban_tool":
            continue
        if payload.get("target_profile") != target or payload.get("lane") != lane:
            continue
        if lane == "review" and payload.get("review_event_id") != latest_review:
            continue
        source = payload.get("source_profile")
        if isinstance(source, str) and PROFILE_ID_RE.fullmatch(source):
            return source
    return None


def _policy(root: Path) -> dict | None:
    path = root / "config.yaml"
    if not path.exists():
        return None
    try:
        import hermes_yaml as yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        mode = data.get("bot_mode", {}) if isinstance(data, dict) else {}
        if not isinstance(mode, dict) or "invocation_acl" not in mode:
            return None
        acl = mode["invocation_acl"]
        if not isinstance(acl, dict) or not acl or any(
            not isinstance(target, str) or not PROFILE_ID_RE.fullmatch(target)
            or not isinstance(sources, list) or not sources
            or any(not isinstance(source, str) or not PROFILE_ID_RE.fullmatch(source)
                   for source in sources)
            for target, sources in acl.items()
        ):
            return {}  # invalid configured policy: no agent admission
        return acl
    except Exception:
        # A broken file must not silently lift a policy. The operator must restore it.
        return {}


def permits(source: str | None, target: str, *, root: Path) -> bool:
    """Check canonical profile ids; caller must derive source from runtime, not args."""
    acl = _policy(root)
    if acl is None:
        return True
    if not acl:
        return False
    if target not in acl:
        return True
    return bool(source and source in acl[target])


def require(source: str | None, target: str, *, root: Path) -> None:
    if not permits(source, target, root=root):
        raise PermissionError(f"agent invocation of profile {target!r} is denied by bot_mode.invocation_acl")


def install_root() -> Path:
    from hermes_constants import get_routing_process_hermes_home
    from tools.bot_mode_probe import _hermes_root
    return _hermes_root(Path(get_routing_process_hermes_home()))
