"""Local weekly selection work on the session-owned Wisdom queue.

The producer never invokes a model or sends a message. A leased session reviews
the evidence; selected exact hashes and their advice commit together.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agent_led.evidence import build_evidence
from .agent_led.policy import load_policy
from .mediation_store import MediationStore
from .preferences import WisdomPreferences, suppression_key

MAX_REVIEW_CANDIDATES = 100


def _week_key(now: datetime) -> str:
    day = now.astimezone(timezone.utc).date()
    return f"weekly:{day - timedelta(days=day.weekday())}"


def _handled(db, org: str, skill_id: str, content_hash: str) -> bool:
    return bool(
        db.execute(
            """SELECT 1 FROM candidate WHERE skill_id=? AND content_hash=? AND state='dismissed'
            UNION ALL SELECT 1 FROM local_draft WHERE skill_id=? AND source_hash=?
            UNION ALL SELECT 1 FROM wisdom_assessment a
            LEFT JOIN local_event e ON e.id=json_extract(a.reference_json,'$.event_id')
            WHERE a.organization_id=? AND a.state!='retired'
            AND json_extract(a.reference_json,'$.kind')='candidate'
            AND COALESCE(json_extract(a.reference_json,'$.local_skill_id'), e.skill_id)=?
            AND json_extract(a.reference_json,'$.content_hash')=? LIMIT 1""",
            (
                skill_id,
                content_hash,
                skill_id,
                content_hash,
                org,
                skill_id,
                content_hash,
            ),
        ).fetchone()
    )


def enqueue_weekly_review(
    service, *, now=None, skills_root=None, dry_run=False
) -> dict:
    """One UTC-week review per organization, across all local processes.

    A manual run does not bypass the weekly cap or policy. Dry runs are local
    evidence previews and never create a queue item or call a model.
    """
    policy = load_policy(client=service.client)
    if (
        not policy.enabled
        or not policy.max_candidates
        or not policy.notification_defaults.get("skill_ready_to_share")
    ):
        return {
            "queued": False,
            "skipped_reason": "organization_notifications_disabled",
        }
    service.require_setup()
    org = service.store.active_org_id()
    current = now or datetime.now(timezone.utc)
    queue = MediationStore(service.store, clock=current.timestamp)
    queue._require_org(org)
    event_key = _week_key(current)
    with service.store.transaction() as db:
        queue._check_org(db, org)
        existing = db.execute(
            "SELECT id FROM wisdom_assessment WHERE organization_id=? AND event_key=?",
            (org, event_key),
        ).fetchone()
    if existing and not dry_run:
        return {
            "queued": False,
            "assessment_id": existing[0],
            "skipped_reason": "already_queued",
        }
    signals = {}
    for event in service.store.local_events(kind="wisdom.candidate"):
        qualification = event.get("qualification")
        if (qualification == "high_usage" and policy.consecutive_day_usage_counts) or (
            qualification == "refinement" and policy.repeated_edits_count
        ):
            signals.setdefault(
                (event.get("payload") or {}).get("skill_name"), []
            ).append(qualification)
    evidence = build_evidence(
        store=service.store,
        policy=policy,
        history=None,
        organization={"id": org},
        at=current,
        skills_root=skills_root,
        supporting_signals=signals,
    )
    candidates = []
    excluded = dict(evidence.excluded)
    with service.store.transaction() as db:
        queue._check_org(db, org)
        for item in evidence.candidates:
            local = db.execute(
                "SELECT id,current_hash FROM local_skill WHERE canonical_path=? AND deleted_at IS NULL AND source_kind='local'",
                (item.path,),
            ).fetchone()
            reason = None
            if not local or local["current_hash"] != item.content_hash:
                reason = "changed"
            elif _handled(db, org, local["id"], item.content_hash):
                reason = "previously_handled"
            elif len(candidates) >= MAX_REVIEW_CANDIDATES:
                reason = "review_batch_limit"
            if reason:
                excluded.setdefault(reason, []).append(item.skill_name)
                continue
            # Keep identity separate from prose; names are not unique in a tree.
            candidates.append({
                "local_skill_id": local["id"],
                "content_hash": item.content_hash,
                "evidence": {
                    **item.as_dict(),
                    "skill_name": local["id"],
                    "display_name": item.skill_name,
                },
            })
    reference = {
        "kind": "weekly_review",
        "candidates": candidates,
        "window": {
            "start": evidence.window_start,
            "end": evidence.window_end,
            "days": policy.window_days,
        },
        "excluded": excluded,
        "max_candidates": policy.max_candidates,
    }
    if len(json.dumps(reference).encode()) > 256_000:
        raise ValueError("weekly evidence exceeds the bounded review limit")
    if dry_run:
        return {"queued": False, "dry_run": True, **reference}
    identity = queue.enqueue(org, event_key, reference)
    return {"queued": True, "assessment_id": identity, "considered": len(candidates)}


def _session_call(runtime):
    from .agent_led.agent import session_model_call

    return session_model_call(runtime, name="wisdom_weekly_review")


def process_weekly_review(
    mediation, org: str, job: dict, *, runtime, history, reviewer=None
) -> None:
    from .agent_led.agent import review_candidates
    from .mediation import Advice, conversation_context
    from .qualification import snapshot_tree

    service, queue = mediation.service, mediation.queue
    policy = load_policy(client=service.client)
    if (
        not policy.enabled
        or not policy.max_candidates
        or not policy.notification_defaults.get("skill_ready_to_share")
    ):
        queue.defer_for_preferences(org, job, 0)
        return
    candidates = job["reference"]["candidates"]
    prefs = WisdomPreferences(service, clock=queue.clock).check(
        org,
        [
            {"kind": "candidate", "content_hash": item["content_hash"]}
            for item in candidates
        ],
    )
    if not prefs["available"] or prefs["muted"]:
        queue.defer_for_preferences(org, job, 0)
        return
    available = {}
    for item in candidates:
        ref = {"kind": "candidate", "content_hash": item["content_hash"]}
        if suppression_key(ref) in prefs["suppressed"]:
            continue
        local = service.store.local_skill(item["local_skill_id"])
        if (
            not local
            or local.get("deleted_at")
            or local["current_hash"] != item["content_hash"]
        ):
            continue
        if snapshot_tree(Path(local["canonical_path"]))[0] != item["content_hash"]:
            continue
        available[item["local_skill_id"]] = item
    if job["state"] == "fallback":
        # One deterministic notice, never arbitrary model-free selection.
        with service.store.transaction() as db:
            queue._check_org(db, org)
            db.execute(
                """UPDATE wisdom_assessment SET state='ready',reference_json=?,advice_json=?,lease_token=NULL,lease_until=NULL
                WHERE id=? AND organization_id=? AND state='fallback' AND lease_token=? AND lease_until>?""",
                (
                    json.dumps({
                        "kind": "notice",
                        "notification": {
                            "summary": "Weekly review unavailable. Use /wisdom candidates to review local skills."
                        },
                    }),
                    json.dumps({
                        "title": "Collective Wisdom review",
                        "relevance": "recommend",
                        "explanation": "Hermes could not complete this week's review. Your skills remain private; use /wisdom candidates to review them manually.",
                    }),
                    job["id"],
                    org,
                    job["lease_token"],
                    queue.clock(),
                ),
            )
        return
    selected = []
    report: dict[str, Any] = {
        "considered": list(available),
        "selected": [],
        "excluded": job["reference"]["excluded"],
    }
    if available:
        organization_name = service.organization_display_name()
        result = (reviewer or review_candidates)(
            {
                "window_days": job["reference"]["window"]["days"],
                "window_start": job["reference"]["window"]["start"],
                "window_end": job["reference"]["window"]["end"],
                "min_aggregate_count": policy.min_aggregate_count,
                "organization": {
                    "name": organization_name
                    if isinstance(organization_name, str)
                    else None
                },
                "max_candidates": min(
                    policy.max_candidates, job["reference"]["max_candidates"]
                ),
                "candidates": [item["evidence"] for item in available.values()],
                "memory": {"conversation_excerpts": conversation_context(history)},
            },
            model_call=_session_call(runtime),
        )
        if len(result.recommendations) > min(
            policy.max_candidates, job["reference"]["max_candidates"]
        ):
            raise ValueError("weekly model exceeded the recommendation cap")
        for rec in result.recommendations:
            item = available.get(rec.skill_name)
            if not item or rec.content_hash != item["content_hash"]:
                raise ValueError("weekly model invented or changed a candidate")
            usage = item["evidence"]
            advice = Advice(
                assessment_id=job["id"],
                relevance="recommend",
                title=rec.editorial_name,
                explanation=(
                    rec.one_line_description
                    + " "
                    + f"Used {usage['invocation_count']} times across {usage['days_used']} days in the last {usage['window_days']} days. "
                    + rec.why_coworkers_benefit[:300]
                )[:600],
            ).model_dump()
            advice["usage_evidence"] = {
                key: usage[key]
                for key in (
                    "invocation_count",
                    "days_used",
                    "window_days",
                    "last_used_day",
                )
            }
            advice["how_user_relied_on_it"] = rec.how_user_relied_on_it
            advice["provenance"] = {
                key: str(runtime.get(key) or "")[:128] for key in ("provider", "model")
            }
            selected.append({
                "source": item,
                "editorial_name": rec.editorial_name,
                "editorial_description": rec.one_line_description,
                "advice": advice,
                "portability": [p.model_dump() for p in rec.portability],
            })
        report["nothing_to_recommend_reason"] = result.nothing_to_recommend_reason
    service.require_setup()
    current_choices = []
    for choice in selected:
        local = service.store.local_skill(choice["source"]["local_skill_id"])
        if (
            local
            and not local.get("deleted_at")
            and snapshot_tree(Path(local["canonical_path"]))[0]
            == choice["source"]["content_hash"]
        ):
            current_choices.append(choice)
        else:
            report["excluded"].setdefault("changed_during_review", []).append(
                choice["source"]["local_skill_id"]
            )
    _commit_selection(mediation, org, job, current_choices, report)


def _commit_selection(
    mediation, org: str, job: dict, selected: list[dict], report: dict
) -> None:
    service, queue = mediation.service, mediation.queue
    now = queue.clock()
    with service.store.transaction() as db:
        queue._check_org(db, org)
        if not db.execute(
            "SELECT 1 FROM wisdom_assessment WHERE id=? AND organization_id=? AND state='assessing' AND lease_token=? AND lease_until>?",
            (job["id"], org, job["lease_token"], now),
        ).fetchone():
            return
        for choice in selected:
            source = choice["source"]
            skill_id, content_hash = source["local_skill_id"], source["content_hash"]
            local = db.execute(
                "SELECT current_hash FROM local_skill WHERE id=? AND deleted_at IS NULL",
                (skill_id,),
            ).fetchone()
            if (
                not local
                or local[0] != content_hash
                or _handled(db, org, skill_id, content_hash)
            ):
                continue
            payload = {
                "skill_name": source["evidence"]["display_name"],
                "editorial_name": choice["editorial_name"],
                "editorial_description": choice["editorial_description"],
                "agent_led_weekly": True,
                "qualification": "weekly_usage",
                "local_reasons": source["evidence"],
                "consent_required": True,
                "networked": False,
            }
            event_id = service.store.emit_local_event(
                kind="wisdom.candidate",
                skill_id=skill_id,
                content_hash=content_hash,
                payload=payload,
                session_id=job["owner_session"],
                task_id=None,
                qualification="weekly_usage",
                _db=db,
            )
            if event_id is None:
                continue
            identity = uuid.uuid4().hex
            reference = {
                "kind": "candidate",
                "event_id": event_id,
                "local_skill_id": skill_id,
                "content_hash": content_hash,
                "weekly_review_id": job["id"],
                "weekly_rank": len(report["selected"]) + 1,
            }
            advice = {
                **choice["advice"],
                "assessment_id": identity,
                "portability": choice["portability"],
            }
            db.execute(
                """INSERT INTO wisdom_assessment
                (id,organization_id,event_key,origin_session,reference_json,state,owner_session,available_at,advice_json,created_at,updated_at)
                VALUES(?,?,?,?,?,'ready',?,?,?,?,?)""",
                (
                    identity,
                    org,
                    f"candidate:{event_id}",
                    job["owner_session"],
                    json.dumps(reference),
                    job["owner_session"],
                    now,
                    json.dumps(advice),
                    now,
                    now,
                ),
            )
            report["selected"].append({
                "skill_id": skill_id,
                "content_hash": content_hash,
                "assessment_id": identity,
            })
        db.execute(
            """UPDATE wisdom_assessment SET state='reviewed',advice_json=?,lease_token=NULL,lease_until=NULL,updated_at=?
            WHERE id=? AND organization_id=? AND lease_token=?""",
            (json.dumps(report), now, job["id"], org, job["lease_token"]),
        )
