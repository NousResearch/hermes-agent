"""Native-control-only consent over existing exact-package Wisdom operations."""

from __future__ import annotations

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
        "from_version",
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
    )
    return {key: plan[key] for key in keys if key in plan}


def _signature(plan: dict[str, Any]) -> dict[str, Any]:
    return {
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
        )
    }


class WisdomConsent:
    def __init__(self, service, *, clock=None):
        self.service = service
        self.queue = MediationStore(
            service.store, **({"clock": clock} if clock else {})
        )

    def _plan(self, reference: dict[str, Any]) -> tuple[str, dict[str, Any]]:
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
            plan = self.service.install_plan(f"{skill_id}@v{version}")
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

    @staticmethod
    def project(value: dict[str, Any]) -> dict[str, Any]:
        plan = value["plan"]
        blocked = bool(
            plan.get("modified")
            or plan.get("sensitive_expansion")
            or plan.get("allowed") is False
            or (plan.get("compatibility") or {}).get("outcome")
            not in {None, "compatible"}
        )
        return {
            "id": value["id"],
            "assessment_id": value["assessment_id"],
            "state": value["state"],
            "operation": value["operation"],
            "expires_at": value["expires_at"],
            "facts": public_plan(plan),
            "actions": (
                ["inspect"]
                if value["state"] != "pending"
                else ["defer", "inspect", "confirm"]
                if not blocked
                else ["defer", "inspect"]
            ),
            "result": value.get("result"),
        }

    def resolve(
        self, org: str, interaction_id: str, actor: ConsentActor, action: str
    ) -> dict[str, Any]:
        page = re.fullmatch(r"inspect(?:\.([0-9]{1,4}))?", action)
        result = self._resolve(
            org, interaction_id, actor, "inspect" if page else action
        )
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
        preference_user = preferences.identity(org) if action == "defer" else None
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
                preference = preferences.stage_suppression(
                    db,
                    org=org,
                    user=preference_user,
                    reference=json.loads(assessment["reference_json"]),
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
            if value["operation"] == "publish":

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
                )
            else:
                ref = {
                    "kind": "skill",
                    "skill_id": plan["skill_id"],
                    "version": plan["version"],
                }
                operation, refreshed = self._plan(ref)
                if operation != value["operation"] or _signature(
                    refreshed
                ) != _signature(plan):
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
        with self.service.store.transaction() as db:
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
        with self.service.store.transaction() as db:
            self.queue._check_org(db, org)
            rows = db.execute(
                "SELECT * FROM wisdom_consent WHERE organization_id=? AND state='applying' AND updated_at<?",
                (org, self.queue.clock() - 900),
            ).fetchall()
            for row in rows:
                value = _decode(row)
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
