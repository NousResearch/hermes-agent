from __future__ import annotations

import argparse
import fnmatch
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

VALID_WAKE_EVENTS = {
    "new_repair_packet",
    "pr_status_check_change",
    "review_request",
    "queue_priority_change",
    "stale_claim_expiry",
    "failed_ci_transition",
    "governance_unblock",
    "explicit_operator_command",
}

TERMINAL_CLAIM_STATUSES = {
    "completed",
    "blocked",
    "refused",
    "superseded",
    "no-op with evidence",
    "expired/recovered",
}

ACTIVE_CLAIM_STATUS = "active"


class DevLaneStore:
    """Durable file-backed heartbeat, claim, and pickup store for developer lanes.

    The store is deliberately small and JSON-only so cron/controller wrappers can
    use it as a restart-safe coordination layer without adding a new daemon or
    timer loop.  All mutating operations read current state, validate governance
    constraints, write atomically, and leave receipt pointers behind.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for child in ("heartbeats", "continuity", "claims", "work", "receipts"):
            (self.root / child).mkdir(parents=True, exist_ok=True)

    def path_for(self, relative: str | Path) -> Path:
        return self.root / relative

    def read_json(self, relative: str | Path, default: Any | None = None) -> Any:
        path = self.path_for(relative)
        if not path.exists():
            return [] if default is None else default
        return json.loads(path.read_text())

    def write_json(self, relative: str | Path, value: Any) -> Path:
        path = self.path_for(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
        return path

    def register_lane(self, lane_id: str, *, repo_scope: str, authorized_scopes: list[str]) -> dict[str, Any]:
        lanes = self.read_json("lanes.json", {})
        lanes[lane_id] = {
            "lane_id": lane_id,
            "repo_scope": repo_scope,
            "authorized_scopes": authorized_scopes,
            "registered_at": lanes.get(lane_id, {}).get("registered_at") or utc_now(),
        }
        self.write_json("lanes.json", lanes)
        return lanes[lane_id]

    def latest_heartbeat(self, lane_id: str) -> dict[str, Any] | None:
        value = self.read_json(Path("heartbeats") / f"{lane_id}.json", None)
        return value

    def emit_heartbeat(
        self,
        lane_id: str,
        agent_session_id: str,
        repo_scope: str,
        current_state: str,
        claimed_work_item_id: str | None,
        last_successful_activity_at: str,
        next_eligible_wake_at: str,
        evidence_pointer: str,
        last_error_at: str | None = None,
        last_error_message: str | None = None,
        last_event_consumed: str | None = None,
    ) -> dict[str, Any]:
        heartbeat = {
            "schema_version": "dev_lane_heartbeat.v1",
            "lane_id": lane_id,
            "agent_session_id": agent_session_id,
            "repo_project_scope": repo_scope,
            "current_state": current_state,
            "claimed_work_item_id": claimed_work_item_id,
            "last_successful_activity_at": last_successful_activity_at,
            "last_error_at": last_error_at,
            "last_error_message": last_error_message,
            "next_eligible_wake_at": next_eligible_wake_at,
            "evidence_pointer": evidence_pointer,
            "last_event_consumed": last_event_consumed,
            "emitted_at": utc_now(),
        }
        self.write_json(Path("heartbeats") / f"{lane_id}.json", heartbeat)
        return heartbeat

    def write_continuity_packet(self, lane_id: str, packet: dict[str, Any]) -> dict[str, Any]:
        required = {
            "current_objective",
            "current_repo_branch_pr",
            "files_touched_or_planned",
            "active_blocker",
            "last_verified_command_check",
            "next_safe_action",
            "explicit_non_claims",
            "operator_approvals_relied_on",
        }
        missing = sorted(required - set(packet))
        if missing:
            raise ValueError(f"continuity packet missing required fields: {', '.join(missing)}")
        document = {"schema_version": "dev_lane_continuity.v1", "lane_id": lane_id, **packet, "updated_at": utc_now()}
        self.write_json(Path("continuity") / f"{lane_id}.json", document)
        return document

    def continuity_packet(self, lane_id: str) -> dict[str, Any] | None:
        return self.read_json(Path("continuity") / f"{lane_id}.json", None)

    def active_claim_for_work(self, work_item_id: str, *, now: str) -> dict[str, Any] | None:
        claim = self.read_json(Path("claims") / f"{safe_id(work_item_id)}.json", None)
        if not claim or claim.get("claim_status") != ACTIVE_CLAIM_STATUS:
            return None
        if utc_parse(claim["claim_expires_at"]) <= utc_parse(now):
            return None
        return claim

    def claim_for_work(self, work_item_id: str) -> dict[str, Any] | None:
        return self.read_json(Path("claims") / f"{safe_id(work_item_id)}.json", None)

    def claim_work(
        self,
        work_item_id: str,
        lane_id: str,
        owner_session_id: str,
        now: str,
        ttl_seconds: int,
        evidence_path: str,
        recovery_evidence_path: str | None = None,
    ) -> dict[str, Any]:
        current = self.claim_for_work(work_item_id)
        now_dt = utc_parse(now)
        if current and current.get("claim_status") == ACTIVE_CLAIM_STATUS:
            expires = utc_parse(current["claim_expires_at"])
            if expires > now_dt:
                raise ValueError(
                    f"active claim already exists for {work_item_id}: "
                    f"{current.get('claim_owner_session_id')} until {current.get('claim_expires_at')}"
                )
            if not recovery_evidence_path:
                raise ValueError(f"expired claim for {work_item_id} requires recovery_evidence_path")
            recovered = dict(current)
            recovered.update(
                {
                    "claim_status": "expired/recovered",
                    "recovered_by_session_id": owner_session_id,
                    "recovered_at": now,
                    "recovery_evidence_path": recovery_evidence_path,
                }
            )
            self._append_history(recovered)
        expires_at = format_utc(now_dt + timedelta(seconds=ttl_seconds))
        claim = {
            "schema_version": "dev_lane_claim.v1",
            "work_item_id": work_item_id,
            "lane_id": lane_id,
            "claim_owner_session_id": owner_session_id,
            "claim_started_at": now,
            "claim_expires_at": expires_at,
            "claim_status": ACTIVE_CLAIM_STATUS,
            "claim_evidence_path": evidence_path,
        }
        self.write_json(Path("claims") / f"{safe_id(work_item_id)}.json", claim)
        return claim

    def close_claim(self, work_item_id: str, status: str, evidence_path: str, now: str, message: str | None = None) -> dict[str, Any]:
        if status not in TERMINAL_CLAIM_STATUSES:
            raise ValueError(f"invalid terminal claim status: {status}")
        claim = self.claim_for_work(work_item_id)
        if not claim:
            raise ValueError(f"no claim exists for {work_item_id}")
        closed = dict(claim)
        closed.update({"claim_status": status, "closed_at": now, "closeout_evidence_path": evidence_path})
        if message:
            closed["closeout_message"] = message
        self.write_json(Path("claims") / f"{safe_id(work_item_id)}.json", closed)
        self._append_history(closed)
        return closed

    def add_work_item(self, item: dict[str, Any]) -> dict[str, Any]:
        if "work_item_id" not in item:
            raise ValueError("work item missing work_item_id")
        items = self.read_json("work/items.json", {})
        item = {"schema_version": "dev_lane_work_item.v1", **item}
        item.setdefault("events", [])
        item.setdefault("status", "queued")
        item.setdefault("created_at", utc_now())
        items[item["work_item_id"]] = item
        self.write_json("work/items.json", items)
        return item

    def record_event(self, work_item_id: str, event_type: str, created_at: str, event_id: str | None = None) -> dict[str, Any]:
        if event_type not in VALID_WAKE_EVENTS:
            raise ValueError(f"invalid wake event: {event_type}")
        items = self.read_json("work/items.json", {})
        if work_item_id not in items:
            raise ValueError(f"unknown work item: {work_item_id}")
        event = {"event_id": event_id or f"{event_type}:{created_at}", "event_type": event_type, "created_at": created_at}
        items[work_item_id].setdefault("events", []).append(event)
        self.write_json("work/items.json", items)
        return event

    def start_session(self, lane_id: str, session_id: str, now: str) -> dict[str, Any]:
        heartbeat = self.latest_heartbeat(lane_id)
        continuity = self.continuity_packet(lane_id)
        claim = None
        if heartbeat and heartbeat.get("claimed_work_item_id"):
            claim = self.claim_for_work(heartbeat["claimed_work_item_id"])
            if claim and claim.get("claim_status") == ACTIVE_CLAIM_STATUS and utc_parse(claim["claim_expires_at"]) > utc_parse(now):
                return {
                    "action": "resume",
                    "lane_id": lane_id,
                    "session_id": session_id,
                    "heartbeat": heartbeat,
                    "continuity_packet": continuity,
                    "claim": claim,
                }
        return self.pick_next_work(lane_id, session_id, now=now)

    def pick_next_work(self, lane_id: str, session_id: str, *, now: str) -> dict[str, Any]:
        lane = self.read_json("lanes.json", {}).get(lane_id)
        if not lane:
            raise ValueError(f"unknown lane: {lane_id}")
        items = list(self.read_json("work/items.json", {}).values())
        candidates: list[tuple[tuple[Any, ...], dict[str, Any], str]] = []
        skip_rows: list[dict[str, Any]] = []
        for item in items:
            allowed, reason = self._is_item_pickable_for_lane(item, lane, now)
            if not allowed:
                if reason in {"held_by_do_not_merge", "awaiting_human_review", "refused_without_sidecar_evidence"}:
                    skip_rows.append({"work_item_id": item["work_item_id"], "reason": reason, "skipped_at": now, "lane_id": lane_id})
                continue
            wake_reason = self._wake_reason(item, lane_id, now)
            if not wake_reason:
                continue
            candidates.append((self._rank_key(item), item, wake_reason))
        if skip_rows:
            existing = self.read_json("receipts/skips.json", [])
            existing.extend(skip_rows)
            self.write_json("receipts/skips.json", existing)
        if not candidates:
            next_wake = format_utc(utc_parse(now) + timedelta(minutes=30))
            self.emit_heartbeat(
                lane_id,
                session_id,
                lane["repo_scope"],
                "idle-no-work",
                None,
                now,
                next_wake,
                "receipts/skips.json",
            )
            return {"action": "idle-no-work", "lane_id": lane_id, "next_eligible_wake_at": next_wake}
        _, item, wake_reason = sorted(candidates, key=lambda row: row[0])[0]
        evidence_path = f"claims/{safe_id(item['work_item_id'])}.json"
        claim = self.claim_work(item["work_item_id"], lane_id, session_id, now, 3600, evidence_path)
        latest_event = latest_event_id(item)
        next_wake = format_utc(utc_parse(now) + timedelta(minutes=5))
        self.emit_heartbeat(
            lane_id,
            session_id,
            lane["repo_scope"],
            "working",
            item["work_item_id"],
            now,
            next_wake,
            evidence_path,
            last_event_consumed=latest_event,
        )
        return {
            "action": "claimed",
            "lane_id": lane_id,
            "work_item_id": item["work_item_id"],
            "claim": claim,
            "wake_reason": wake_reason,
            "next_eligible_wake_at": next_wake,
        }

    def status_report(self, *, now: str, stale_after_seconds: int = 1800) -> dict[str, Any]:
        lanes = self.read_json("lanes.json", {})
        rows = []
        for lane_id, lane in sorted(lanes.items()):
            hb = self.latest_heartbeat(lane_id)
            active_claims = self._active_claims_for_lane(lane_id, now)
            stale_claims = self._stale_claims_for_lane(lane_id, now)
            if not hb:
                status = "failed-heartbeat"
                freshness_seconds = None
            else:
                freshness_seconds = max(0, int((utc_parse(now) - utc_parse(hb["emitted_at"])).total_seconds()))
                if freshness_seconds > stale_after_seconds:
                    status = "failed-heartbeat"
                elif hb["current_state"] in {"working", "idle-no-work", "blocked", "awaiting-human", "refused-closeout"}:
                    status = hb["current_state"]
                elif stale_claims:
                    status = "stale"
                else:
                    status = hb["current_state"]
            rows.append(
                {
                    "lane_id": lane_id,
                    "repo_scope": lane.get("repo_scope"),
                    "status": status,
                    "heartbeat_freshness_seconds": freshness_seconds,
                    "active_claims": [c["work_item_id"] for c in active_claims],
                    "stale_claims": [c["work_item_id"] for c in stale_claims],
                    "next_scheduled_wake": hb.get("next_eligible_wake_at") if hb else None,
                    "last_event_consumed": hb.get("last_event_consumed") if hb else None,
                    "last_evidence_receipt": hb.get("evidence_pointer") if hb else None,
                }
            )
        return {"schema_version": "dev_lane_status_report.v1", "generated_at": now, "lanes": rows}

    def _append_history(self, claim: dict[str, Any]) -> None:
        history = self.read_json("claims/history.json", [])
        history.append(claim)
        self.write_json("claims/history.json", history)

    def _active_claims_for_lane(self, lane_id: str, now: str) -> list[dict[str, Any]]:
        claims = []
        for path in (self.root / "claims").glob("*.json"):
            if path.name == "history.json":
                continue
            claim = json.loads(path.read_text())
            if claim.get("lane_id") == lane_id and claim.get("claim_status") == ACTIVE_CLAIM_STATUS and utc_parse(claim["claim_expires_at"]) > utc_parse(now):
                claims.append(claim)
        return claims

    def _stale_claims_for_lane(self, lane_id: str, now: str) -> list[dict[str, Any]]:
        claims = []
        for path in (self.root / "claims").glob("*.json"):
            if path.name == "history.json":
                continue
            claim = json.loads(path.read_text())
            if claim.get("lane_id") == lane_id and claim.get("claim_status") == ACTIVE_CLAIM_STATUS and utc_parse(claim["claim_expires_at"]) <= utc_parse(now):
                claims.append(claim)
        return claims

    def _is_item_pickable_for_lane(self, item: dict[str, Any], lane: dict[str, Any], now: str) -> tuple[bool, str | None]:
        if item.get("repo_scope") != lane.get("repo_scope"):
            return False, "outside_repo_scope"
        if not scopes_overlap(lane.get("authorized_scopes", []), item.get("authorized_scopes", [])):
            return False, "outside_lane_capability"
        if self.active_claim_for_work(item["work_item_id"], now=now):
            return False, "already_actively_claimed"
        labels = {str(label).lower() for label in item.get("labels", [])}
        if "do-not-merge" in labels or "do_not_merge" in labels:
            return False, "held_by_do_not_merge"
        if item.get("awaiting_human_review"):
            return False, "awaiting_human_review"
        if item.get("status") == "refused" and not item.get("sidecar_evidence_path"):
            return False, "refused_without_sidecar_evidence"
        if item.get("status") == "blocked" and latest_event_id(item) == item.get("blocked_at_event_id"):
            return False, "blocked_without_new_event"
        if item.get("status") in {"completed", "superseded"}:
            return False, "terminal"
        return True, None

    def _wake_reason(self, item: dict[str, Any], lane_id: str, now: str) -> str | None:
        hb = self.latest_heartbeat(lane_id)
        consumed = hb.get("last_event_consumed") if hb else None
        new_events = [event for event in item.get("events", []) if event_id(event) != consumed]
        valid_events = [event for event in new_events if event.get("event_type") in VALID_WAKE_EVENTS]
        if valid_events:
            event = sorted(valid_events, key=lambda e: e.get("created_at", ""), reverse=True)[0]
            return f"event:{event['event_type']}"
        if hb and utc_parse(now) >= utc_parse(hb.get("next_eligible_wake_at", "1970-01-01T00:00:00Z")):
            return "timer:fallback"
        if not hb:
            return "event:fresh_session"
        return None

    def _rank_key(self, item: dict[str, Any]) -> tuple[Any, ...]:
        operator_priority = item.get("operator_priority")
        if operator_priority is None:
            operator_priority = 1000
        created = item.get("created_at", "9999-01-01T00:00:00Z")
        return (
            operator_priority,
            0 if item.get("failed_required_ci") else 1,
            0 if item.get("stale_open_pr") else 1,
            0 if item.get("governance_blocker") else 1,
            0 if item.get("review_gap") else 1,
            0 if item.get("dependency_unblocked") else 1,
            created,
        )


def utc_now() -> str:
    return format_utc(datetime.now(timezone.utc))


def utc_parse(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def event_id(event: dict[str, Any]) -> str:
    return str(event.get("event_id") or f"{event.get('event_type')}:{event.get('created_at')}")


def latest_event_id(item: dict[str, Any]) -> str | None:
    events = item.get("events", [])
    if not events:
        return None
    event = sorted(events, key=lambda e: e.get("created_at", ""), reverse=True)[0]
    return event_id(event)


def scopes_overlap(lane_scopes: list[str], item_scopes: list[str]) -> bool:
    if not lane_scopes or not item_scopes:
        return False
    for lane_scope in lane_scopes:
        for item_scope in item_scopes:
            if lane_scope == "**" or lane_scope == item_scope:
                return True
            if fnmatch.fnmatch(item_scope, lane_scope) or fnmatch.fnmatch(lane_scope, item_scope):
                return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Developer lane heartbeat and pickup control")
    parser.add_argument("--root", default=os.getenv("HERMES_DEV_LANE_STATE", str(Path.home() / ".hermes/state/dev-lane-heartbeat")))
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register-lane")
    reg.add_argument("lane_id")
    reg.add_argument("--repo-scope", required=True)
    reg.add_argument("--authorized-scope", action="append", required=True)

    hb = sub.add_parser("heartbeat")
    hb.add_argument("lane_id")
    hb.add_argument("--session", required=True)
    hb.add_argument("--repo-scope", required=True)
    hb.add_argument("--state", required=True)
    hb.add_argument("--claimed-work-item-id")
    hb.add_argument("--last-success", required=True)
    hb.add_argument("--next-wake", required=True)
    hb.add_argument("--evidence", required=True)
    hb.add_argument("--last-error-at")
    hb.add_argument("--last-error-message")

    pick = sub.add_parser("pickup")
    pick.add_argument("lane_id")
    pick.add_argument("--session", required=True)
    pick.add_argument("--now", default=utc_now())

    status = sub.add_parser("status")
    status.add_argument("--now", default=utc_now())
    status.add_argument("--stale-after-seconds", type=int, default=1800)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = DevLaneStore(args.root)
    if args.command == "register-lane":
        result = store.register_lane(args.lane_id, repo_scope=args.repo_scope, authorized_scopes=args.authorized_scope)
    elif args.command == "heartbeat":
        result = store.emit_heartbeat(
            args.lane_id,
            args.session,
            args.repo_scope,
            args.state,
            args.claimed_work_item_id,
            args.last_success,
            args.next_wake,
            args.evidence,
            args.last_error_at,
            args.last_error_message,
        )
    elif args.command == "pickup":
        result = store.pick_next_work(args.lane_id, args.session, now=args.now)
    elif args.command == "status":
        result = store.status_report(now=args.now, stale_after_seconds=args.stale_after_seconds)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
