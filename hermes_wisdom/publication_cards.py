"""Refresh an existing consent card from authoritative owner draft decisions."""

import json
import time

from .consent import WisdomConsent
from .mediation_store import MediationStore
from .mediation_view import interaction_view

FINAL = {"published", "changes_requested", "declined", "invalidated"}


def has_card(store, org, draft_id):
    with store.transaction() as db:
        return (
            db.execute(
                """SELECT 1 FROM wisdom_consent c JOIN wisdom_delivery_receipt r
            ON r.assessment_id=c.assessment_id AND r.organization_id=c.organization_id
            AND r.owner_session=c.owner_session WHERE c.organization_id=?
            AND c.operation='publish' AND c.state='completed'
            AND json_extract(c.result_json,'$.draft_id')=?
            AND c.platform IN ('telegram','slack') LIMIT 1""",
                (org, draft_id),
            ).fetchone()
            is not None
        )


class PublicationCards:
    def __init__(self, service):
        self.service, self.store = service, service.store

    def claim(self, platform):
        org = self.store.active_org_id()
        if not org:
            return []
        with self.store.transaction() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    """SELECT c.*,r.receipt_json FROM wisdom_consent c
                JOIN wisdom_delivery_receipt r ON r.assessment_id=c.assessment_id
                  AND r.organization_id=c.organization_id AND r.owner_session=c.owner_session
                WHERE c.organization_id=? AND c.platform=? AND c.operation='publish'
                  AND c.state='completed' AND json_extract(c.result_json,'$.draft_id') IS NOT NULL
                  AND (json_extract(c.result_json,'$.publication_state')='pending_moderation'
                    OR COALESCE(json_extract(c.result_json,'$.card_displayed_state'),'')
                       != json_extract(c.result_json,'$.publication_state'))
                ORDER BY c.created_at""",
                    (org, platform),
                )
            ]
        if not rows:
            return []
        self.service.require_setup()
        owner = self.service.client.identity.get("owner")
        rows = [
            r
            for r in rows
            if json.loads(r["result_json"]).get("owner_user_id") == owner
        ]
        if not rows:
            return []
        drafts = {d.id: d for d in self.service.client.list_drafts()}
        jobs = []
        for row in rows:
            if len(jobs) >= 4:
                break
            result, plan = json.loads(row["result_json"]), json.loads(row["plan_json"])
            draft = drafts.get(result["draft_id"])
            if not draft or draft.orgId != org or draft.ownerUserId != owner:
                continue
            if plan.get("hashes") != {
                "content": draft.contentHash,
                "author_description": draft.authorDescriptionHash,
                "package_manifest": draft.packageManifestHash,
            }:
                continue
            if draft.state not in FINAL | {"pending_moderation"}:
                continue
            result["publication_state"] = draft.state
            if (
                result.get("card_displayed_state") == draft.state
                or result.get("card_retry_after", 0) > time.time()
            ):
                continue
            receipt = json.loads(row["receipt_json"])
            address = plan.get("origin_address") or {}
            if (
                receipt.get("platform") != platform
                or receipt.get("destination") != address.get("chat_id")
                or str(receipt.get("thread_id") or "")
                != str(address.get("thread_id") or "")
                or str(receipt.get("scope_id") or "")
                != str(address.get("scope_id") or "")
                or not receipt.get("message_id")
            ):
                continue
            entity = "publication-card:" + row["id"]
            token = self.store.acquire_operation_lock(entity, ttl_seconds=180)
            if not token:
                continue
            try:
                with self.store.transaction() as db:
                    MediationStore._check_org(db, org)
                    changed = db.execute(
                        "UPDATE wisdom_consent SET result_json=? WHERE id=? AND result_json=? AND state='completed'",
                        (json.dumps(result), row["id"], row["result_json"]),
                    ).rowcount
                if not changed:
                    self.store.release_operation_lock(entity, token)
                    continue
                view = interaction_view(
                    WisdomConsent.project({**row, "plan": plan, "result": result})
                )
                jobs.append({
                    "id": row["id"],
                    "org": org,
                    "state": draft.state,
                    "receipt": receipt,
                    "view": view,
                    "entity": entity,
                    "token": token,
                })
            except BaseException:
                self.store.release_operation_lock(entity, token)
                raise
        return jobs

    def finish(self, job, *, success):
        try:
            with self.store.transaction() as db:
                MediationStore._check_org(db, job["org"])
                lease = db.execute(
                    "SELECT owner FROM operation_lock WHERE entity_id=?",
                    (job["entity"],),
                ).fetchone()
                if lease is None or lease[0] != job["token"]:
                    return
                row = db.execute(
                    "SELECT result_json FROM wisdom_consent WHERE id=?", (job["id"],)
                ).fetchone()
                result = json.loads(row[0])
                if result.get("publication_state") != job["state"]:
                    return
                if success:
                    result["card_displayed_state"] = job["state"]
                    result.pop("card_retry_after", None)
                    result.pop("card_edit_failures", None)
                else:
                    failures = result.get("card_edit_failures", 0) + 1
                    result["card_edit_failures"] = failures
                    result["card_retry_after"] = time.time() + min(
                        3600, 60 * 2 ** min(failures, 6)
                    )
                db.execute(
                    "UPDATE wisdom_consent SET result_json=? WHERE id=?",
                    (json.dumps(result), job["id"]),
                )
        finally:
            self.store.release_operation_lock(job["entity"], job["token"])
