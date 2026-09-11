"""Native-control-only consent over existing exact-package Wisdom operations."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from .client import WisdomConflict, WisdomNotFound
from .contract import author_description_hash
from .mediation_store import MediationStore, _decode
from .preferences import WisdomPreferences

CONSENT_SECONDS = 24 * 60 * 60
TERMINAL = {"completed", "failed", "stale", "expired", "needs_review"}


@dataclass(frozen=True)
class ConsentActor:
    """Created by the authenticated surface, never decoded from model arguments."""

    session_key: str
    platform: str
    actor_id: str
    chat_id: str = ""
    thread_id: str = ""
    scope_id: str = ""

    @property
    def address(self) -> dict[str, str]:
        return {
            "chat_id": self.chat_id,
            "thread_id": self.thread_id,
            "scope_id": self.scope_id,
        }


def public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    # Receipts, local paths, raw files and credentials never enter the model/UI.
    keys = (
        "skill_id",
        "slug",
        "version",
        "local_version",
        "from_version",
        "update_mode",
        "compatibility",
        "allowed",
        "sensitive_expansion",
        "modified",
        "security_check",
        "professionalism_check",
        "requirements",
        "editorial_name",
        "editorial_description",
        "hashes",
        "content_hash",
        "manifest_hash",
        "sharing_stage",
        "file_names",
        "step",
        "setup_instruction",
        "setup_explanation",
        "setup_requirement",
        "setup_key",
    )
    return {key: plan[key] for key in keys if key in plan}


def _signature(plan: dict[str, Any]) -> dict[str, Any]:
    signature = {
        key: plan.get(key)
        for key in (
            "skill_id",
            "version",
            "from_version",
            "content_hash",
            "manifest_hash",
            "takedown_generation",
            "baseline",
            "previous_baseline",
            "modified",
            "sensitive_expansion",
            "compatibility",
            "allowed",
            "hashes",
            "source_hash",
            "update_mode",
            "setup_key",
        )
    }
    for key in ("security_check", "professionalism_check"):
        review = plan.get(key)
        if isinstance(review, dict):
            # Bind displayed findings, not a fresh scanner timestamp or routing metadata.
            review = {k: v for k, v in review.items() if k not in {
                "assessed_at", "checked_at", "scanned_at", "provenance",
            }}
            if isinstance(review.get("checks"), list):
                review["checks"] = sorted(
                    review["checks"], key=lambda value: json.dumps(value, sort_keys=True)
                )
        signature[key] = hashlib.sha256(
            json.dumps(review, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    # Plans cross a JSON persistence boundary. Compare wire values, not Python
    # tuple/list container types returned by compatibility evaluation.
    return json.loads(json.dumps(signature))


class WisdomConsent:
    def __init__(self, service, *, clock=None):
        self.service = service
        self.queue = MediationStore(
            service.store, **({"clock": clock} if clock else {})
        )

    def request(
        self, org: str, reference: dict[str, Any], actor: ConsentActor,
        *, title: str, explanation: str, queue_delivery: bool = True,
    ) -> dict[str, Any]:
        """Prepare explicit review; passive command replies do not queue a notification."""
        self.service.require_setup()
        now = self.queue.clock()
        reference = {**reference, "user_requested": True}
        setup_plan = self._plan(reference)[1] if reference["kind"] == "setup" else None
        target = (
            ("setup_key", setup_plan["setup_key"], "content_hash", setup_plan["content_hash"])
            if setup_plan else
            ("skill_id", reference["skill_id"], "version", reference["version"])
            if reference["kind"] == "skill"
            else ("event_id", reference["event_id"], "source_hash", reference["content_hash"])
        )
        request_key = "request:" + hashlib.sha256(json.dumps(
            [reference, actor.session_key, actor.platform, actor.actor_id, actor.address],
            sort_keys=True,
        ).encode()).hexdigest()
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            existing = db.execute(
                """SELECT c.*,a.state AS delivery_state,d.deferred_at
                FROM wisdom_consent c JOIN wisdom_assessment a ON a.id=c.assessment_id
                LEFT JOIN wisdom_consent_defer d ON d.interaction_id=c.id AND d.surface=c.platform
                WHERE c.organization_id=? AND c.owner_session=? AND c.actor_id=? AND c.platform=?
                AND json_extract(c.plan_json,?)=? AND json_extract(c.plan_json,?)=?
                AND (?!='skill' OR c.operation IN ('install','update'))
                ORDER BY c.created_at DESC,c.rowid DESC LIMIT 1""",
                (org, actor.session_key, actor.actor_id, actor.platform,
                 f"$.{target[0]}", target[1], f"$.{target[2]}", target[3], reference["kind"]),
            ).fetchone()
            if existing:
                value = _decode(existing)
                if value["plan"].get("origin_address") and value["plan"]["origin_address"] != actor.address:
                    raise WisdomNotFound("Wisdom interaction not found")
                policy_changed = (
                    value["operation"] == "install"
                    and "update_mode" in reference
                    and reference["update_mode"] != value["plan"].get("update_mode")
                )
                if policy_changed and value["delivery_state"] in {"delivering", "delivery_uncertain"}:
                    raise WisdomConflict("Resolve the current review delivery before changing its update policy")
                if (
                    value["state"] in {"applying", "completed"}
                    or value["delivery_state"] in {"delivering", "delivery_uncertain"}
                    or (value["state"] == "pending" and value["expires_at"] > now
                        and value["deferred_at"] is None and not policy_changed)
                ):
                    return self.project(value)
                # Keep the original receipt and suppression history. A successor
                # key makes concurrent requests and interrupted plan rebuilds idempotent.
                request_key = f"request:consent:{value['id']}:{request_key.removeprefix('request:')}"
                if value["state"] == "pending":
                    db.execute(
                        "UPDATE wisdom_consent SET state=?,updated_at=? WHERE id=?",
                        ("expired" if value["expires_at"] <= now else "stale", now, value["id"]),
                    )
                db.execute(
                    """UPDATE wisdom_assessment SET state='retired',lease_token=NULL,lease_until=NULL,updated_at=?
                    WHERE id=? AND state NOT IN ('delivered','delivering','delivery_uncertain')""",
                    (now, value["assessment_id"]),
                )
            identity = self.queue.enqueue(
                org, request_key, reference, origin_session=actor.session_key, _db=db,
            )
            advice = {"assessment_id": identity, "title": title,
                      "explanation": explanation, "relevance": "recommend"}
            db.execute(
                """UPDATE wisdom_assessment SET owner_session=?,advice_json=?,state=?,updated_at=?
                WHERE id=? AND state IN ('pending','passive') AND lease_token IS NULL""",
                (actor.session_key, json.dumps(advice), "ready" if queue_delivery else "passive", now, identity),
            )
        return self.present(org, identity, actor)

    def _plan(self, reference: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if reference["kind"] == "setup":
            from .setup_execution import plan_step

            return "setup", plan_step(self.service.store, reference)
        if reference["kind"] == "candidate":
            event_id = reference["event_id"]
            event, skill_id, source_hash, name = self.service._candidate_event_context(
                event_id
            )
            if event.get("organization_id") != self.service.store.active_org_id():
                raise WisdomConflict("candidate belongs to a different organization")
            if reference.get("content_hash") not in {None, source_hash}:
                raise WisdomConflict("candidate changed after recommendation")
            existing = self.service.store.latest_draft_for_source(skill_id, source_hash)
            local_version = self.service.candidate_local_version(skill_id, source_hash)
            editorial = (
                self.service.store.candidate_editorial_metadata(
                    skill_id, content_hash=source_hash
                )
                or {}
            )
            if existing is None:
                security = self.service.candidate_security_check(
                    skill_id=skill_id, content_hash=source_hash
                )
                review = self.service.store.professionalism_review(
                    skill_id=skill_id,
                    content_hash=source_hash,
                    author_description_hash=author_description_hash(""),
                )
                return "share", {
                    "skill_id": skill_id,
                    "slug": name,
                    "local_version": local_version,
                    "event_id": event_id,
                    "source_hash": source_hash,
                    "allowed": security["upload_allowed"],
                    "sharing_stage": "prepare",
                    "security_check": security,
                    "professionalism_check": (
                        review.get("result") or {"status": review["state"]}
                    )
                    if review
                    else None,
                    **editorial,
                }
            if (
                reference.get("prepared_draft_id")
                and existing["id"] != reference["prepared_draft_id"]
            ):
                raise WisdomConflict("the prepared contribution changed")
            prepared = self.service.prepare_candidate(event_id)
            if prepared["stage"] == "review":
                review = prepared["review"]
                hashes = review["hashes"]
                checks = review.get("draft") or {}
            else:
                checks = prepared["prepared"]
                draft = self.service.store.draft(prepared["prepared"]["local_draft_id"])
                hashes = {
                    "content": draft["content_hash"],
                    "author_description": draft["description_hash"],
                    "package_manifest": draft["manifest_hash"],
                }
            return "publish", {
                "skill_id": skill_id,
                "slug": name,
                "local_version": local_version,
                "event_id": event_id,
                "source_hash": source_hash,
                "sharing_stage": "approve",
                "file_names": [item["path"] for item in checks.get("files", [])],
                "hashes": hashes,
                **(
                    self.service.store.candidate_editorial_metadata(
                        skill_id, content_hash=source_hash
                    )
                    or {}
                ),
                "security_check": checks.get("security_check"),
                **{
                    key: checks[key]
                    for key in ("editorial_name", "editorial_description")
                    if checks.get(key)
                },
                "professionalism_check": checks.get("professionalism_check"),
                "requirements": checks.get("system_specification")
                or checks.get("systemSpec"),
                "allowed": not (
                    (checks.get("security_check") or {}).get("status") == "blocked"
                    or (checks.get("security_check") or {}).get("upload_allowed") is False
                    or ((checks.get("local_scan") or {}).get("guard") or {}).get(
                        "allowed"
                    )
                    is False
                ),
            }
        skill_id, version = reference["skill_id"], reference["version"]
        installation = self.service.store.installation(skill_id)
        if installation and installation["state"] == "active":
            plan = self.service.update_plan(skill_id)
            operation = "update"
        else:
            plan = self.service.install_plan(
                f"{skill_id}@v{version}", update_mode=reference.get("update_mode")
            )
            operation = "install"
        if plan.get("version") != version:
            raise WisdomConflict("the available version changed; inspect it again")
        detail = self.service.version_detail(skill_id, version)
        metadata = detail.get("version") or {}
        for key in (
            "security_check",
            "professionalism_check",
            "editorial_name",
            "editorial_description",
        ):
            plan[key] = metadata.get(key)
        plan["requirements"] = metadata.get("system_spec")
        return operation, plan

    def present(
        self,
        org: str,
        assessment_id: str,
        actor: ConsentActor,
        *,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        self.service.require_setup()
        self.queue._require_org(org)
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            row = db.execute(
                "SELECT * FROM wisdom_assessment WHERE id=? AND organization_id=?",
                (assessment_id, org),
            ).fetchone()
            session = db.execute(
                "SELECT * FROM wisdom_agent_session WHERE organization_id=? AND session_key=?",
                (org, actor.session_key),
            ).fetchone()
            if (
                row is None
                or row["owner_session"] != actor.session_key
                or session is None
                or session["actor_id"] != actor.actor_id
                or session["platform"] != actor.platform
            ):
                raise WisdomNotFound("Wisdom interaction not found")
            if lease_token is not None and (
                row["lease_token"] != lease_token
                or row["lease_until"] <= self.queue.clock()
            ):
                raise WisdomConflict("Wisdom assessment ownership changed")
            existing = db.execute(
                "SELECT * FROM wisdom_consent WHERE assessment_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
                (assessment_id,),
            ).fetchone()
            if (
                existing is not None
                and existing["state"] == "pending"
                and existing["expires_at"] <= self.queue.clock()
            ):
                db.execute(
                    "UPDATE wisdom_consent SET state='expired' WHERE id=?",
                    (existing["id"],),
                )
                existing = None
            if existing is not None and existing["state"] not in {
                "expired",
                "stale",
                "needs_review",
                "failed",
            }:
                return self.project(_decode(existing))
            reference = json.loads(row["reference_json"])
        operation, plan = self._plan(reference)
        if operation == "setup":
            plan["setup_explanation"] = (json.loads(row["advice_json"] or "{}").get("explanation") or "")[:600]
        else:
            plan["not_now_suppression_days"] = WisdomPreferences(
                self.service
            ).review_suppression_days(org)
        address = json.loads(session["address_json"])
        if address and address != actor.address:
            raise WisdomNotFound("Wisdom interaction not found")
        plan["origin_address"] = address
        now = self.queue.clock()
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            # Ownership may have moved while fetching the canonical package.
            owner = db.execute(
                "SELECT owner_session,lease_token,lease_until FROM wisdom_assessment WHERE id=?",
                (assessment_id,),
            ).fetchone()
            if owner is None or owner[0] != actor.session_key:
                raise WisdomConflict("Wisdom assessment ownership changed")
            if lease_token is not None and (owner[1] != lease_token or owner[2] <= now):
                raise WisdomConflict("Wisdom assessment ownership changed")
            db.execute(
                """INSERT INTO wisdom_consent
                (id,organization_id,assessment_id,owner_session,actor_id,platform,
                 operation,plan_json,expires_at,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                (
                    uuid.uuid4().hex,
                    org,
                    assessment_id,
                    actor.session_key,
                    actor.actor_id,
                    actor.platform,
                    operation,
                    json.dumps(plan),
                    now + CONSENT_SECONDS,
                    now,
                    now,
                ),
            )
            value = _decode(
                db.execute(
                    "SELECT * FROM wisdom_consent WHERE assessment_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
                    (assessment_id,),
                ).fetchone()
            )
        return self.project(value)

    def project(self, value: dict[str, Any]) -> dict[str, Any]:
        plan = value["plan"]
        # Read-only surfaces must not offer an approval the resolver will reject.
        state = value["state"]
        if state == "pending" and value["expires_at"] <= self.queue.clock():
            state = "expired"
        blocked = bool(
            plan.get("modified")
            or plan.get("sensitive_expansion")
            or plan.get("allowed") is False
            or (plan.get("compatibility") or {}).get("outcome")
            not in {None, "compatible"}
            or (value["operation"] in {"install", "update"}
                and (plan.get("security_check") or {}).get("status") not in {"pass", "advisory"})
        )
        return {
            "id": value["id"],
            "assessment_id": value["assessment_id"],
            "state": state,
            "operation": value["operation"],
            "expires_at": value["expires_at"],
            "facts": public_plan(plan),
            "actions": (
                ["inspect"]
                if state != "pending"
                else ["defer", "inspect", "confirm"]
                if not blocked
                else ["defer", "inspect"]
            ),
            "result": value.get("result"),
        }

    def _local_contribution(self, org: str, interaction_id: str, actor: ConsentActor) -> dict[str, Any]:
        if actor.platform != "local":
            raise WisdomNotFound("Local publication review not found")
        self._resolve(org, interaction_id, actor, "inspect")
        with self.service.store.transaction() as db:
            row = db.execute("SELECT * FROM wisdom_consent WHERE id=? AND organization_id=?", (interaction_id, org)).fetchone()
            value = _decode(row)
        if value["operation"] not in {"share", "publish"} or value["state"] != "pending" or value["expires_at"] <= self.queue.clock():
            raise WisdomConflict("This contribution needs a fresh review control")
        return value

    def prepare_local_publication(self, org: str, interaction_id: str, actor: ConsentActor) -> dict[str, Any]:
        value = self._local_contribution(org, interaction_id, actor)
        prepared = self.service.prepare_candidate(value["plan"]["event_id"])
        draft_id = prepared["prepared"]["local_draft_id"] if prepared["stage"] == "prepared" else prepared["review"]["draft"]["id"]
        return {"draft_id": draft_id}

    def submit_local_publication(self, org: str, interaction_id: str, actor: ConsentActor, *, draft_id: str, expected_hashes: dict[str, str], publication_mode: str) -> dict[str, Any]:
        value = self._local_contribution(org, interaction_id, actor)
        local = self.service.store.draft(draft_id)
        if local is None or (local["skill_id"], local["source_hash"]) != (value["plan"]["skill_id"], value["plan"]["source_hash"]):
            raise WisdomConflict("This package does not belong to the reviewed contribution")
        # The local modal authorizes this exact revision and submission, not just packaging.
        value["operation"] = "publish"
        value["plan"]["hashes"] = expected_hashes
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            accepted = db.execute(
                "UPDATE wisdom_consent SET state='applying',operation='publish',plan_json=?,updated_at=? WHERE id=? AND state='pending'",
                (json.dumps(value["plan"]), self.queue.clock(), interaction_id),
            )
            if accepted.rowcount != 1:
                raise WisdomConflict("This contribution is already being submitted")
        try:
            result = self.service.submit_reviewed_package(
                draft_id, expected_hashes=expected_hashes, publication_mode=publication_mode,
                _record_intent=lambda review: self._record_publication_intent(value, review),
            )
        except WisdomConflict:
            with self.service.store.transaction() as db:
                self._finish(db, value, "stale", {"reason": "package_or_authority_changed"})
            raise
        with self.service.store.transaction() as db:
            self._finish(db, value, "completed", result)
        return result

    def _record_publication_intent(self, value, review):
        """Bind acceptance to the authoritative reviewed draft before the approval RPC."""
        self.service.require_setup()
        org = value["organization_id"]
        owner = self.service.client.identity.get("owner")
        draft = review["draft"]
        if (not owner or draft["orgId"] != org or draft["ownerUserId"] != owner
                or review["hashes"] != value["plan"]["hashes"]
                or self.queue.clock() >= value["expires_at"]):
            raise WisdomConflict("Publication review or account changed")
        intent = {
            "draft_id": draft["id"], "org_id": org, "owner_user_id": owner,
            "server_revision": draft["updatedAt"], "draft_commit": draft["draftCommit"],
            "hashes": review["hashes"],
        }
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            row = db.execute("SELECT state,plan_json FROM wisdom_consent WHERE id=?", (value["id"],)).fetchone()
            if row is None or row["state"] != "applying":
                raise WisdomConflict("Publication acceptance changed")
            plan = json.loads(row["plan_json"])
            if plan.get("publication_intent") not in (None, intent):
                raise WisdomConflict("A different publication intent is already recorded")
            plan["publication_intent"] = intent
            db.execute("UPDATE wisdom_consent SET plan_json=? WHERE id=?", (json.dumps(plan), value["id"]))
            value["plan"] = plan

    def _recover_publication(self, value):
        intent = value["plan"].get("publication_intent")
        if not intent:
            return None
        self.service.require_setup()
        org = value["organization_id"]
        owner = self.service.client.identity.get("owner")
        if (intent["org_id"], intent["owner_user_id"]) != (org, owner):
            raise WisdomConflict("Publication account changed")
        # Reconstruct bytes and metadata; a matching title or local draft state is not evidence.
        reviewed = self.service.review(intent["draft_id"], acknowledge=False, expected_hashes=intent["hashes"])
        self.service.require_setup()
        if self.service.client.identity.get("owner") != owner:
            raise WisdomConflict("Publication account changed during recovery")
        draft = reviewed["draft"]
        if (draft["id"], draft["orgId"], draft["ownerUserId"], draft["draftCommit"]) != (
            intent["draft_id"], org, owner, intent["draft_commit"],
        ):
            raise WisdomConflict("Publication draft binding changed")
        if draft["state"] not in {"pending_moderation", "published", "changes_requested"}:
            return None
        return {
            "operation": "publish", "draft_id": draft["id"],
            "publication_state": draft["state"], "owner_user_id": owner,
            "portal_url": self.service.portal_review_url(draft["id"]),
            "reason": "gateway_publication_reconciled",
        }

    def resolve(
        self, org: str, interaction_id: str, actor: ConsentActor, action: str
    ) -> dict[str, Any]:
        if action == "setup.status":
            from .setup_continuation import inspect_continuation

            result = self._resolve(org, interaction_id, actor, "inspect")
            return inspect_continuation(self, org, result, actor)
        if action in {"setup.recover", "setup.clear"}:
            return self._recover_setup(org, interaction_id, actor, action)
        if action == "review":
            return self._review_in_portal(org, interaction_id, actor)
        if action == "recheck":
            result = self._resolve(org, interaction_id, actor, "inspect")
            if (result["operation"] == "setup" and result["state"] == "pending"
                    and result["facts"].get("setup_requirement") and result["facts"].get("allowed") is False):
                from .setup_continuation import recheck_prerequisite

                return recheck_prerequisite(self, org, result, actor)
            if result["state"] in {"stale", "expired"} or (
                result["state"] == "pending"
                and result["expires_at"] <= self.queue.clock()
            ) or (
                result["operation"] == "setup" and result["state"] in {"needs_review", "failed"}
                and ((result.get("result") or {}).get("setup") or {}).get("state") in {"abandoned", "blocked", "failed"}
            ):
                return self.present(org, result["assessment_id"], actor)
            return result
        page = re.fullmatch(r"inspect(?:\.([0-9]{1,4}))?", action)
        result = self._resolve(
            org, interaction_id, actor, "inspect" if page else action
        )
        if result["operation"] == "setup" and result["state"] in {"applying", "needs_review"}:
            with self.service.store.transaction() as db:
                self.queue._check_org(db, org)
                value = _decode(db.execute("SELECT * FROM wisdom_consent WHERE id=?", (interaction_id,)).fetchone())
            return self._refresh_setup(value)
        if page and result["operation"] == "share" and result["state"] == "completed":
            next_assessment = (result.get("result") or {}).get("assessment_id")
            if next_assessment:
                with self.service.store.transaction() as db:
                    self.queue._check_org(db, org)
                    child = db.execute(
                        "SELECT id FROM wisdom_consent WHERE organization_id=? "
                        "AND assessment_id=? AND operation='publish' "
                        "ORDER BY created_at DESC LIMIT 1",
                        (org, next_assessment),
                    ).fetchone()
                if child:
                    # Re-authorize the linked interaction; inspecting never approves it.
                    return self.resolve(org, child["id"], actor, action)
        if page and result["operation"] == "publish" and result["state"] == "pending":
            return self._inspect_package(org, interaction_id, result, int(page[1] or 0))
        return result

    def _recover_setup(self, org, interaction_id, actor, action):
        from .setup_execution import recover_step

        result = self._resolve(org, interaction_id, actor, "inspect")
        if result["operation"] != "setup":
            raise WisdomNotFound("Setup interaction not found")
        if result["state"] not in {"applying", "needs_review"}:
            return result
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            value = _decode(db.execute("SELECT * FROM wisdom_consent WHERE id=?", (interaction_id,)).fetchone())

        def finish(db, setup):
            self._finish(db, value, "needs_review", {"operation": "setup", "setup": setup})

        setup = recover_step(self.service.store, value, acknowledged_stopped=action == "setup.clear", finish=finish)
        if setup["state"] in {"passed", "failed", "blocked"}:
            return self._refresh_setup(value)
        return {**result, "state": "needs_review" if setup["state"] == "abandoned" else result["state"],
                "result": {"operation": "setup", "setup": setup}}

    def _refresh_setup(self, value):
        from .setup_execution import interaction_progress

        outcome = {"operation": "setup", "setup": interaction_progress(self.service.store, value)}
        with self.service.store.transaction() as db:
            self.queue._check_org(db, value["organization_id"])
            current = _decode(db.execute("SELECT * FROM wisdom_consent WHERE id=?", (value["id"],)).fetchone())
            if current["state"] not in {"applying", "needs_review"}:
                return self.project(current)
            if current["state"] == "needs_review" and outcome["setup"]["state"] == "unknown":
                # Only persisted uncertain executions get recovery controls;
                # validation failures before journaling must not imply a command ran.
                from .setup_execution import KIND
                exists = db.execute(
                    "SELECT 1 FROM operation_journal WHERE kind=? AND json_extract(payload_json,'$.interaction_id')=?",
                    (KIND, value["id"]),
                ).fetchone()
                if exists is None:
                    return self.project(current)
            if outcome["setup"]["state"] in {"passed", "failed", "blocked"}:
                state = "completed" if outcome["setup"]["state"] == "passed" else "failed"
                self._finish(db, current, state, outcome)
                return {**self.project(current), "state": state, "result": outcome}
        return {**self.project(current), "result": outcome}

    def _review_in_portal(self, org, interaction_id, actor):
        result = self._resolve(org, interaction_id, actor, "inspect")
        if result["operation"] not in {"share", "publish"} or result["state"] != "pending":
            return result
        entity = "portal-review:" + interaction_id
        token = self.service.store.acquire_operation_lock(entity, ttl_seconds=300)
        if not token:
            raise WisdomConflict(
                "the private draft is already being prepared; try again shortly"
            )
        try:
            def guard():
                current = self._resolve(org, interaction_id, actor, "inspect")
                if current["state"] != "pending" or current["expires_at"] <= self.queue.clock():
                    raise WisdomConflict("this review control is no longer current")
                with self.service.store.transaction() as db:
                    lease = db.execute(
                        "SELECT owner FROM operation_lock WHERE entity_id=?", (entity,)
                    ).fetchone()
                    if lease is None or lease[0] != token:
                        raise WisdomConflict("private review ownership changed")
                return current

            guard()
            with self.service.store.transaction() as db:
                plan_json = db.execute(
                    "SELECT plan_json FROM wisdom_consent WHERE id=? AND organization_id=?",
                    (interaction_id, org),
                ).fetchone()[0]
            plan = json.loads(plan_json)
            reference = {
                "kind": "candidate", "event_id": plan["event_id"],
                "content_hash": plan["source_hash"],
            }
            # Validate the qualified source before authorizing any private upload.
            self._plan(reference)
            draft = self.service.draft_candidate(
                plan["event_id"], expected_hashes=plan.get("hashes"),
                _pre_upload_guard=guard,
            )
            operation, refreshed = self._plan(reference)
            refreshed["origin_address"] = plan.get("origin_address")
            outcome = {
                "draft_id": draft["draft_id"], "portal_url": draft["portal_url"],
                "publication_state": draft["state"],
                "owner_user_id": self.service.client.identity.get("owner"),
            }
            guard()
            with self.service.store.transaction() as db:
                self.queue._check_org(db, org)
                changed = db.execute(
                    "UPDATE wisdom_consent SET operation=?,plan_json=?,result_json=?,updated_at=? "
                    "WHERE id=? AND organization_id=? AND state='pending' AND plan_json=?",
                    (operation, json.dumps(refreshed), json.dumps(outcome), self.queue.clock(),
                     interaction_id, org, plan_json),
                ).rowcount
                if not changed:
                    raise WisdomConflict("the interaction changed while preparing review")
            return self._resolve(org, interaction_id, actor, "inspect")
        finally:
            self.service.store.release_operation_lock(entity, token)

    def _inspect_package(self, org, interaction_id, result, page):
        from .contract import author_description_hash, sha256_address
        from .package import verify_content_files

        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            plan = json.loads(
                db.execute(
                    "SELECT plan_json FROM wisdom_consent WHERE id=? AND organization_id=?",
                    (interaction_id, org),
                ).fetchone()[0]
            )
        prepared = self.service.prepare_candidate(plan["event_id"])
        if prepared["stage"] == "prepared":
            review = prepared["prepared"]
            description = review["drafted_description"]
        else:
            review = prepared["review"]
            description = review["draft"]["authorDescription"]
        files = [
            (item["path"], "file", item["content_utf8"].encode("utf-8"))
            for item in review["files"]
        ]
        _, content_hash = verify_content_files(files)
        hashes = {
            "content": content_hash,
            "author_description": author_description_hash(description),
            "package_manifest": sha256_address(
                next(body for name, _, body in files if name == "skill.manifest.json")
            ),
        }
        if hashes != plan["hashes"]:
            raise WisdomConflict(
                "the review package changed; request a fresh consent control"
            )
        # Small sequential pages preserve every byte without overflowing native
        # message limits. This private action never inserts file text into a model.
        pages = []
        for name, _, body in [
            ("Author description", "file", description.encode("utf-8")),
            *files,
        ]:
            text = re.sub(
                r"[\x00-\x08\x0b-\x1f\x7f-\x9f]",
                lambda match: f"\\u{ord(match[0]):04x}",
                body.decode("utf-8"),
            )
            for start in range(0, max(1, len(text)), 1000):
                pages.append({
                    "path": name,
                    "content": text[start : start + 1000],
                    "hash": sha256_address(body),
                })
        if page >= len(pages):
            raise WisdomNotFound("review page not found")
        self.queue._require_org(org)
        return {
            **result,
            "inspection": {
                **pages[page],
                "page": page,
                "page_count": len(pages),
                "description": description,
            },
        }

    def _resolve(
        self, org: str, interaction_id: str, actor: ConsentActor, action: str
    ) -> dict[str, Any]:
        """Called only from authenticated button/CLI handlers, never a model tool."""
        self.service.require_setup()
        now = self.queue.clock()
        preferences = WisdomPreferences(self.service, clock=self.queue.clock)
        preference_user = None
        if action == "defer":
            with self.service.store.transaction() as db:
                operation = db.execute(
                    "SELECT operation FROM wisdom_consent WHERE id=? AND organization_id=?",
                    (interaction_id, org),
                ).fetchone()
            # Setup deferral is entirely local, not an organization preference.
            if operation and operation["operation"] != "setup":
                preference_user = preferences.identity(org)
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            row = db.execute(
                "SELECT * FROM wisdom_consent WHERE id=? AND organization_id=?",
                (interaction_id, org),
            ).fetchone()
            if row is None or (
                row["owner_session"],
                row["platform"],
                row["actor_id"],
            ) != (actor.session_key, actor.platform, actor.actor_id):
                raise WisdomNotFound("Wisdom interaction not found")
            value = _decode(row)
            if (
                value["plan"].get("origin_address")
                and value["plan"]["origin_address"] != actor.address
            ):
                raise WisdomNotFound("Wisdom interaction not found")
            if action == "inspect" or value["state"] in TERMINAL:
                return self.project(value)
            if action == "defer":
                if value["state"] != "pending" or value["expires_at"] <= now:
                    return self.project(value)
                assessment = db.execute(
                    "SELECT reference_json FROM wisdom_assessment WHERE id=? AND organization_id=?",
                    (value["assessment_id"], org),
                ).fetchone()
                if assessment is None:
                    raise WisdomNotFound("Wisdom assessment not found")
                preference = {} if value["operation"] == "setup" else preferences.stage_suppression(
                    db,
                    org=org,
                    user=preference_user,
                    reference=json.loads(assessment["reference_json"]),
                    days=value["plan"].get("not_now_suppression_days", 30),
                )
                db.execute(
                    "INSERT OR REPLACE INTO wisdom_consent_defer VALUES(?,?,?)",
                    (interaction_id, actor.platform, now),
                )
                return {**self.project(value), "deferred": True, **preference}
            if action != "confirm":
                raise ValueError("unsupported Wisdom consent action")
            if value["state"] != "pending":
                return self.project(value)
            if value["expires_at"] <= now:
                self._finish(db, value, "expired", {"reason": "consent_expired"})
                return {**self.project(value), "state": "expired"}
            if "confirm" not in self.project(value)["actions"]:
                return {**self.project(value), "state": "needs_review"}
            if value["operation"] == "share":
                # Consent authorizes local packaging only. Queue creation and
                # acceptance commit together, so repeated clicks cannot fork work.
                plan = value["plan"]
                identity = self.queue.enqueue(
                    org,
                    f"share-package:{interaction_id}",
                    {
                        "kind": "share_package",
                        "event_id": plan["event_id"],
                        "content_hash": plan["source_hash"],
                        "consent_id": interaction_id,
                        "user_requested": True,
                    },
                    origin_session=actor.session_key,
                    _db=db,
                )
                outcome = {
                    "operation": "share",
                    "packaging_state": "queued",
                    "assessment_id": identity,
                    "published": False,
                }
                db.execute(
                    "UPDATE wisdom_consent SET state='completed',result_json=?,updated_at=? WHERE id=?",
                    (json.dumps(outcome), now, interaction_id),
                )
                from .operation_outbox import stage

                stage(db, value, "completed", outcome, now)
                return {**self.project(value), "state": "completed", "result": outcome}
            db.execute(
                "UPDATE wisdom_consent SET state='applying',updated_at=? WHERE id=?",
                (now, interaction_id),
            )
        # The service journal owns recovery. Never blindly repeat an ambiguous
        # apply after a crash; the pending record stays visible for reconciliation.
        try:
            plan = value["plan"]
            if value["operation"] == "setup":
                from .setup_execution import execute_step

                self.queue._require_org(org)
                if self.queue.clock() >= value["expires_at"]:
                    raise WisdomConflict("setup consent expired")
                setup = execute_step(self.service, value, actor)
                outcome = {"operation": "setup", "setup": setup}
                state = {"blocked": "failed", "passed": "completed"}.get(setup["state"], "applying")
                with self.service.store.transaction() as db:
                    self.queue._check_org(db, org)
                    current = _decode(db.execute("SELECT * FROM wisdom_consent WHERE id=?", (interaction_id,)).fetchone())
                    if current["state"] != "applying":
                        return self.project(current)
                    if state == "applying":
                        db.execute("UPDATE wisdom_consent SET result_json=? WHERE id=?", (json.dumps(outcome), interaction_id))
                    else:
                        self._finish(db, current, state, outcome)
                return {**self.project(value), "state": state, "result": outcome}
            elif value["operation"] == "publish":

                def check_authority():
                    self.service.require_setup()
                    self.queue._require_org(org)
                    if self.queue.clock() >= value["expires_at"]:
                        raise WisdomConflict("consent expired before publication")

                _, refreshed = self._plan({
                    "kind": "candidate",
                    "event_id": plan["event_id"],
                    "content_hash": plan["source_hash"],
                })
                if _signature(refreshed) != _signature(plan):
                    raise WisdomConflict("the prepared package changed; review again")
                result = self.service.approve_candidate(
                    plan["event_id"],
                    expected_hashes=plan["hashes"],
                    _pre_upload_guard=check_authority,
                    _record_intent=lambda review: self._record_publication_intent(value, review),
                )
            else:
                ref = {
                    "kind": "skill",
                    "skill_id": plan["skill_id"],
                    "version": plan["version"],
                    "update_mode": plan.get("update_mode"),
                }
                operation, refreshed = self._plan(ref)
                if (
                    "confirm" not in self.project({**value, "plan": refreshed})["actions"]
                    or operation != value["operation"]
                    or _signature(refreshed) != _signature(plan)
                ):
                    raise WisdomConflict("the exact plan changed; review again")
                self.queue._require_org(org)
                if self.queue.clock() >= value["expires_at"]:
                    raise WisdomConflict("consent expired while validating the package")
                # Persist the service receipt before apply. Recovery matches this
                # exact journal entry, never merely the current installed version.
                value["plan"]["receipt"] = refreshed["receipt"]
                with self.service.store.transaction() as db:
                    self.queue._check_org(db, org)
                    db.execute(
                        "UPDATE wisdom_consent SET plan_json=? WHERE id=?",
                        (json.dumps(value["plan"]), interaction_id),
                    )
                apply = (
                    self.service.install_apply
                    if operation == "install"
                    else self.service.update_apply
                )
                result = apply(refreshed["receipt"])
            state = "completed"
            # Result continuation is bounded and typed, not arbitrary provider text.
            outcome = {
                "operation": value["operation"],
                "skill_id": plan["skill_id"],
                "version": plan.get("version"),
                "publication_state": result.get("publication_state")
                if isinstance(result, dict)
                else None,
                "portal_url": result.get("portal_url")
                if isinstance(result, dict)
                else None,
                "draft_id": result.get("draft_id")
                if isinstance(result, dict) and value["operation"] == "publish"
                else None,
                "owner_user_id": self.service.client.identity.get("owner")
                if value["operation"] == "publish" else None,
            }
        except WisdomConflict:
            state, outcome = "stale", {"reason": "package_or_authority_changed"}
        except Exception as exc:
            state, outcome = "needs_review", {"reason": type(exc).__name__}
            if value["operation"] == "publish" and value["plan"].get("publication_intent"):
                # A lost response is not failure evidence. Leave the exact accepted
                # operation for read-only reconciliation rather than report failure.
                return {**self.project(value), "state": "applying", "result": {"reason": "publication_reconciliation_pending"}}
        with self.service.store.transaction() as db:
            current = _decode(db.execute("SELECT * FROM wisdom_consent WHERE id=?", (interaction_id,)).fetchone())
            if value["operation"] == "setup" and current["state"] != "applying":
                return self.project(current)
            self._finish(db, value, state, outcome)
        return {**self.project(value), "state": state, "result": outcome}

    def _finish(self, db, value, state, result):
        now = self.queue.clock()
        payload = json.dumps({"state": state, **result})
        db.execute(
            "UPDATE wisdom_consent SET state=?,result_json=?,updated_at=? WHERE id=?",
            (state, payload, now, value["id"]),
        )
        db.execute(
            "INSERT OR IGNORE INTO wisdom_consent_outcome VALUES(?,?,?,?,NULL)",
            (value["id"], value["organization_id"], value["owner_session"], payload),
        )
        from .operation_outbox import stage

        stage(db, value, state, result, now)
        if state == "completed" and value["operation"] in {"install", "update", "setup"}:
            self.queue.enqueue(
                value["organization_id"], f"setup-handoff:{value['id']}",
                {"kind": "setup_handoff", "consent_id": value["id"], "user_requested": True},
                origin_session=value["owner_session"], _db=db,
            )

    def pending(self, org: str) -> list[dict[str, Any]]:
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            rows = db.execute(
                "SELECT * FROM wisdom_consent WHERE organization_id=? ORDER BY created_at DESC LIMIT 100",
                (org,),
            ).fetchall()
            result = []
            for row in rows:
                item = self.project(_decode(row))
                item["deferred_surfaces"] = [
                    r[0]
                    for r in db.execute(
                        "SELECT surface FROM wisdom_consent_defer WHERE interaction_id=?",
                        (row["id"],),
                    ).fetchall()
                ]
                result.append(item)
            return result

    def recover(self, org: str) -> None:
        """Reconcile interrupted consent without creating new authorization.

        Existing setup/check recovery owns unfinished service journals. A missing
        or unfinished exact journal stays visible for manual review, not replay.
        """
        setup, publications = [], []
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            rows = db.execute(
                "SELECT * FROM wisdom_consent WHERE organization_id=? AND state='applying' "
                "AND (operation='setup' OR updated_at<?)",
                (org, self.queue.clock() - 900),
            ).fetchall()
            for row in rows:
                value = _decode(row)
                if value["operation"] == "setup":
                    setup.append(value)
                    continue
                if value["operation"] == "publish":
                    publications.append(value)
                    continue
                plan = value["plan"]
                completed = False
                if value["operation"] in {"install", "update"}:
                    journals = db.execute(
                        "SELECT payload_json,state FROM operation_journal WHERE kind=? AND entity_id=?",
                        (value["operation"], plan["skill_id"]),
                    ).fetchall()
                    completed = any(
                        journal["state"] == "done"
                        and plan.get("receipt")
                        and json.loads(journal["payload_json"]).get("receipt")
                        == plan["receipt"]
                        for journal in journals
                    )
                self._finish(
                    db,
                    value,
                    "completed" if completed else "needs_review",
                    {
                        "operation": value["operation"],
                        "skill_id": plan["skill_id"],
                        "version": plan.get("version"),
                        "reason": "journal_reconciled"
                        if completed
                        else "interrupted_operation_requires_review",
                    },
                )
        for value in setup:
            self._refresh_setup(value)
        for value in publications:
            try:
                outcome = self._recover_publication(value)
            except WisdomConflict:
                outcome = None
            except Exception:
                continue  # Offline or unavailable Gateway: retain the exact pending intent.
            with self.service.store.transaction() as db:
                self.queue._check_org(db, org)
                current = _decode(db.execute("SELECT * FROM wisdom_consent WHERE id=?", (value["id"],)).fetchone())
                if current["state"] != "applying" or current["plan"] != value["plan"]:
                    continue
                if outcome:
                    self.service.store.complete_contribution(outcome["draft_id"], outcome["publication_state"], _db=db)
                    self.service.store.consume_receipt(outcome["draft_id"], _db=db)
                self._finish(db, current, "completed" if outcome else "needs_review",
                             outcome or {"reason": "interrupted_publication_requires_review"})
