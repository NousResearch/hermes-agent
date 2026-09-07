"""Consent-requested packaging on the existing private-session work queue."""

import json
from pathlib import Path

from .agent_led.agent import package_for_share, session_model_call
from .agent_led.schemas import SharePackage
from .agent_led.share_flow import normalize_generated_package, prepass
from .client import WisdomConflict


def process_share_package(mediation, org: str, job: dict, *, runtime) -> dict:
    service, queue = mediation.service, mediation.queue
    service.require_setup()
    reference = dict(job["reference"])

    def guard(db):
        queue.check_claim(db, org, job["id"], job["lease_token"])

    if job["state"] == "fallback":
        with service.store.transaction() as db:
            queue._check_org(db, org)
            changed = db.execute(
                """UPDATE wisdom_assessment SET reference_json=?,advice_json=?
                WHERE id=? AND organization_id=? AND state='fallback'
                AND lease_token=? AND lease_until>?""",
                (
                    json.dumps({"kind": "notice", "user_requested": True}),
                    json.dumps({
                        "title": "Share preparation needs attention",
                        "relevance": "recommend",
                        "explanation": "Hermes could not prepare this package. Nothing was uploaded or published. Use /wisdom candidates to review it manually.",
                    }),
                    job["id"],
                    org,
                    job["lease_token"],
                    queue.clock(),
                ),
            ).rowcount
            if not changed:
                raise WisdomConflict("Wisdom assessment ownership changed")
            db.execute(
                "UPDATE wisdom_consent SET result_json=json_set(result_json,'$.packaging_state','failed') WHERE id=? AND organization_id=?",
                (reference["consent_id"], org),
            )
        return _read_job(service, job["id"])

    event, skill_id, source_hash, name = service._candidate_event_context(
        reference["event_id"]
    )
    if event.get("organization_id") != org or source_hash != reference["content_hash"]:
        raise WisdomConflict("the candidate changed before packaging")
    source = Path(service.store.local_skill(skill_id)["canonical_path"])
    with service.store.transaction() as db:
        guard(db)
        consent = db.execute(
            "SELECT state,operation,owner_session FROM wisdom_consent WHERE id=? AND organization_id=?",
            (reference["consent_id"], org),
        ).fetchone()
        if not consent or tuple(consent) != (
            "completed",
            "share",
            job["owner_session"],
        ):
            raise WisdomConflict("packaging has no matching native Share request")

    if reference.get("package") is None:
        evidence = prepass(source)
        if evidence["source_content_hash"] != source_hash:
            raise WisdomConflict("the source changed during the packaging pre-pass")
        package = normalize_generated_package(
            package_for_share(
                {
                    "skill_name": name,
                    "source_content_hash": source_hash,
                    "files": evidence["files"],
                    "prepass": {
                        key: value
                        for key, value in evidence.items()
                        if key not in {"files", "existing_security_scan"}
                    },
                },
                model_call=session_model_call(
                    runtime, name="wisdom_share_packaging", max_tokens=16000
                ),
            )
        )
        if package.skill_name != name or package.source_content_hash != source_hash:
            raise WisdomConflict("the generated package changed candidate identity")
        reference["package"] = package.model_dump(mode="json")
        # Persist model output before materializing a draft. A crash after this
        # point resumes the same proposal instead of producing competing bytes.
        with service.store.transaction() as db:
            guard(db)
            db.execute(
                "UPDATE wisdom_assessment SET reference_json=? WHERE id=?",
                (json.dumps(reference), job["id"]),
            )
    package = SharePackage.model_validate(reference["package"])
    if not queue.renew(org, job["id"], job["lease_token"]):
        raise WisdomConflict("Wisdom assessment ownership changed")
    prepared = service.prepare_share_package(
        package,
        source_path=source,
        staging_root=service.store.root / "share-staging",
        _lease_guard=guard,
    )
    service.require_setup()
    with service.store.transaction() as db:
        guard(db)
        next_reference = {
            "kind": "candidate",
            "event_id": reference["event_id"],
            "content_hash": source_hash,
            "local_skill_id": skill_id,
            "prepared_draft_id": prepared["local_draft_id"],
            "user_requested": True,
        }
        advice = {
            "title": package.editorial_name,
            "relevance": "recommend",
            "explanation": "Your proposed team package is ready to review. Nothing has been uploaded or published. Review its files and portability notes before approving sharing.",
        }
        db.execute(
            "UPDATE wisdom_assessment SET reference_json=?,advice_json=?,state='ready',updated_at=? WHERE id=?",
            (json.dumps(next_reference), json.dumps(advice), queue.clock(), job["id"]),
        )
        db.execute(
            "UPDATE wisdom_consent SET result_json=json_set(result_json,'$.packaging_state','ready') WHERE id=? AND organization_id=?",
            (reference["consent_id"], org),
        )
    return _read_job(service, job["id"])


def _read_job(service, identity):
    from .mediation_store import _decode

    with service.store.transaction() as db:
        return _decode(
            db.execute(
                "SELECT * FROM wisdom_assessment WHERE id=?", (identity,)
            ).fetchone()
        )
