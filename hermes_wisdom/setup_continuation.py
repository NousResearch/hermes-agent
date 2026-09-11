"""Read-only recovery entry from a completed native installation or setup step."""

import json

from .client import WisdomNotFound
from .installed_setup import inspect_installed_setup
from .package import PackagePolicyError


def recheck_prerequisite(consent, org, result, actor):
    """Replace a blocked review with current facts, without acknowledging or running it."""
    with consent.service.store.transaction() as db:
        consent.queue._check_org(db, org)
        db.execute(
            """UPDATE wisdom_consent SET state='stale',updated_at=?
            WHERE id=? AND organization_id=? AND state='pending'
            AND operation='setup' AND json_extract(plan_json,'$.allowed')=0""",
            (consent.queue.clock(), result["id"], org),
        )
    return consent.present(org, result["assessment_id"], actor)


def inspect_continuation(consent, org, result, actor):
    if result["operation"] not in {"install", "update", "setup"}:
        raise WisdomNotFound("Setup interaction not found")
    if result["state"] != "completed":
        return consent.resolve(org, result["id"], actor, "inspect")
    facts = result["facts"]

    def status(state, message):
        return {**result, "setup_continuation": {"state": state, "message": message}}

    try:
        info = inspect_installed_setup(consent.service.store, facts["skill_id"], version=facts["version"])
    except PackagePolicyError:
        return status("needs_review", "This installation changed or is unavailable. Review the current installed version before continuing setup.")
    if info["content_hash"] != facts["content_hash"]:
        return status("needs_review", "The installed package changed. Review its current version before continuing setup.")
    with consent.service.store.transaction() as db:
        consent.queue._check_org(db, org)
        latest = db.execute(
            """SELECT id,state,operation FROM wisdom_consent WHERE organization_id=?
            AND owner_session=? AND actor_id=? AND platform=?
            AND operation IN ('install','update','setup')
            AND json_extract(plan_json,'$.skill_id')=? AND json_extract(plan_json,'$.version')=?
            AND json_extract(plan_json,'$.content_hash')=? ORDER BY created_at DESC,rowid DESC LIMIT 1""",
            (org, actor.session_key, actor.actor_id, actor.platform,
             facts["skill_id"], facts["version"], facts["content_hash"]),
        ).fetchone()
        handoff = db.execute(
            "SELECT state,last_error,reference_json,advice_json FROM wisdom_assessment WHERE organization_id=? AND event_key=?",
            (org, f"setup-handoff:{latest['id'] if latest else result['id']}"),
        ).fetchone()
    if latest and latest["operation"] == "setup" and latest["state"] != "completed":
        # Re-authorize the exact linked control; this never creates or confirms one.
        current = consent.resolve(org, latest["id"], actor, "inspect")
        if current["state"] == "pending" and current["expires_at"] <= consent.queue.clock():
            current = {**current, "state": "expired"}
        return current
    if info["ready_to_use"] is True:
        return status("ready", "Declared prerequisites and the approved verification command passed.")
    if not info["guidance"]:
        return status("needs_review", "This version has no valid setup guide. Ask the publisher for setup and verification instructions.")
    if handoff and handoff["last_error"] == "session_model_unavailable" and handoff["state"] == "pending":
        return status("waiting_for_model", "Setup is waiting for this conversation's model. Select or reconnect the model, then send a message here to resume. No setup command is started while waiting.")
    if handoff and handoff["advice_json"]:
        advice = json.loads(handoff["advice_json"])
        if advice.get("assessment_kind") == "setup_status":
            if (json.loads(handoff["reference_json"]).get("setup_status") or {}).get("ready"):
                return status("needs_review", "Setup verification is no longer current. Inspect prerequisites and request a fresh review.")
            return status("needs_review", advice["explanation"])
    return status("queued", "Setup is queued in this conversation. Send a message here to resume with its active model. Each command still requires separate approval.")
