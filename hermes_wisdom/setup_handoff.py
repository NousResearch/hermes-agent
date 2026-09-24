"""Setup continuation authority from native consent or a recorded automatic update."""

import json

from .client import WisdomConflict
from .mediation_store import MediationStore, _decode
from .store import utc_now


def finish_automatic_update(store, operation_id):
    """Commit the update receipt and its continuation together, without inventing consent."""
    queue = MediationStore(store)
    with store.transaction() as db:
        operation = db.execute("SELECT * FROM operation_journal WHERE id=?", (operation_id,)).fetchone()
        if (not operation or operation["kind"] != "update"
                or operation["phase"] not in {"local_ledger_committed", "gateway_recorded"}):
            raise WisdomConflict("automatic update receipt is unavailable")
        plan = json.loads(operation["payload_json"])
        if plan.get("automatic") is not True or plan["update_mode"] not in {"AUTO_WITH_NOTICE", "REQUIRED"}:
            raise WisdomConflict("automatic update authority is unavailable")
        org = plan["previous_installation"]["org_id"]
        queue._check_org(db, org)
        installed = db.execute("SELECT * FROM managed_install WHERE skill_id=?", (plan["skill_id"],)).fetchone()
        if not installed or (installed["org_id"], installed["state"], installed["version"], installed["content_hash"]) != (
            org, "active", plan["version"], plan["content_hash"],
        ):
            raise WisdomConflict("automatic update installation changed")
        queue.enqueue(
            org, f"setup-update:{operation_id}",
            {"kind": "setup_handoff", "update_operation_id": operation_id, "user_requested": True},
            _db=db,
        )
        db.execute("UPDATE operation_journal SET phase='gateway_recorded',state='done',updated_at=? WHERE id=?",
                   (utc_now(), operation_id))


def setup_source(db, queue, org, job):
    """Revalidate the source and elected session before proposing or persisting a step."""
    queue.check_claim(db, org, job["id"], job["lease_token"])
    reference = job["reference"]
    session = db.execute(
        "SELECT actor_id,platform,address_json FROM wisdom_agent_session WHERE organization_id=? AND session_key=?",
        (org, job["owner_session"]),
    ).fetchone()
    if session is None:
        raise WisdomConflict("setup session authority changed")
    if reference.get("update_operation_id"):
        row = db.execute("SELECT * FROM operation_journal WHERE id=?", (reference["update_operation_id"],)).fetchone()
        if not row or (row["kind"], row["phase"], row["state"]) != ("update", "gateway_recorded", "done"):
            raise WisdomConflict("setup has no recorded automatic update")
        payload = json.loads(row["payload_json"])
        if (payload.get("automatic") is not True or payload["previous_installation"]["org_id"] != org
                or payload["update_mode"] not in {"AUTO_WITH_NOTICE", "REQUIRED"}):
            raise WisdomConflict("automatic update authority changed")
        plan = {key: payload[key] for key in ("skill_id", "slug", "version", "content_hash")}
        plan["origin_address"] = json.loads(session["address_json"])
        parent = {"id": None, "plan": plan, "owner_session": job["owner_session"],
                  "actor_id": session["actor_id"], "platform": session["platform"]}
        # A step on any surface already owns setup for this automatic update.
        latest = db.execute(
            """SELECT id FROM wisdom_consent WHERE organization_id=?
            AND (operation='setup' OR (operation IN ('install','update') AND state='completed'))
            AND json_extract(plan_json,'$.skill_id')=? AND json_extract(plan_json,'$.version')=?
            AND json_extract(plan_json,'$.content_hash')=? ORDER BY created_at DESC,rowid DESC LIMIT 1""",
            (org, plan["skill_id"], plan["version"], plan["content_hash"]),
        ).fetchone()
        return None if latest else parent
    row = db.execute("SELECT * FROM wisdom_consent WHERE id=? AND organization_id=?",
                     (reference["consent_id"], org)).fetchone()
    if (row is None or row["state"] != "completed" or row["owner_session"] != job["owner_session"]
            or row["operation"] not in {"install", "update", "setup"}):
        raise WisdomConflict("setup has no matching completed native operation")
    parent = _decode(row)
    plan = parent["plan"]
    if (session["actor_id"], session["platform"], json.loads(session["address_json"])) != (
        parent["actor_id"], parent["platform"], plan.get("origin_address", {}),
    ):
        raise WisdomConflict("setup session authority changed")
    latest = db.execute(
        """SELECT id FROM wisdom_consent WHERE organization_id=? AND owner_session=?
        AND operation IN ('install','update','setup') AND json_extract(plan_json,'$.skill_id')=?
        AND json_extract(plan_json,'$.version')=? ORDER BY created_at DESC,rowid DESC LIMIT 1""",
        (org, parent["owner_session"], plan["skill_id"], plan["version"]),
    ).fetchone()
    return parent if latest and latest["id"] == parent["id"] else None
