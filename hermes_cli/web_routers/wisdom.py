"""Profile-scoped Collective Wisdom API for Dashboard and Desktop.

Gateway credentials and endpoints remain inside this process.
"""

import asyncio
import hashlib
import re
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_models import (
    WisdomSuggestRequest,
    WisdomReviewRequest,
    WisdomPublicationRequest,
    WisdomPreparedSaveRequest,
    WisdomCandidateDismissRequest,
    WisdomCandidateEventRequest,
    WisdomReviseRequest,
    WisdomDecisionRequest,
    WisdomSetupRequest,
    WisdomScanRequest,
    WisdomInstallPlanRequest,
    WisdomInstallApplyRequest,
    WisdomCheckRequest,
    WisdomUpdatePlanRequest,
    WisdomUpdateApplyRequest,
    WisdomUninstallRequest,
    WisdomNotificationRequest,
    WisdomConsentRequest,
    WisdomMutePrepareRequest,
    WisdomSyncRetryRequest,
    WisdomMuteChooseRequest,
)
from hermes_cli.web_routers._common import (
    _profile_scope,
    _spawn_hermes_action,
    log as _log,
)
from hermes_cli.web_server_profiles import _profile_cli_args
from hermes_cli.web_server_gateway import _ACTION_LOG_FILES, _ACTION_PROCS

router = APIRouter()


def _wisdom_http_error(exc: Exception) -> HTTPException:
    from hermes_wisdom.client import WisdomError
    from hermes_wisdom.package import PackagePolicyError

    if isinstance(exc, WisdomError):
        status = {
            3: 403,
            4: 404,
            5: 409,
            6: 422,
            7: 409,
            8: 503,
        }.get(exc.exit_code, 500)
        return HTTPException(status_code=status, detail=str(exc))
    if isinstance(exc, PackagePolicyError):
        return HTTPException(status_code=422, detail=str(exc))
    _log.exception("Collective Wisdom request failed")
    return HTTPException(status_code=500, detail="Collective Wisdom request failed")


async def _run_wisdom(profile: Optional[str], fn, *, require_setup: bool = True):
    def run():
        with _profile_scope(profile):
            from hermes_wisdom.service import WisdomService
            from hermes_wisdom.entitlement import require_entitlement

            require_entitlement()
            service = WisdomService()
            if require_setup:
                service.require_setup()
            return fn(service)

    try:
        return await asyncio.to_thread(run)
    except Exception as exc:
        raise _wisdom_http_error(exc) from exc


def _schedule_wisdom_professionalism_reviews(profile: Optional[str]) -> None:
    """Wake the local review worker after Dashboard/Desktop creates new work."""

    async def run() -> None:
        try:
            await _run_wisdom(
                profile,
                lambda service: service.process_professionalism_reviews(max_jobs=4),
            )
        except Exception:
            _log.debug("Collective Wisdom professionalism worker failed", exc_info=True)

    asyncio.create_task(run())




@router.get("/api/wisdom/entitlement")
async def get_wisdom_entitlement(profile: Optional[str] = None):
    """Expose refresh-free local JWT entitlement metadata for UI gating."""
    def run():
        with _profile_scope(profile):
            from hermes_wisdom.entitlement import current_entitlement

            # Derive the boolean and metadata from one auth/token snapshot so a
            # concurrent token replacement cannot produce a mismatched response.
            metadata = current_entitlement()
            scopes = tuple(metadata.get("scopes", ()))
            return {
                "entitled": "wisdom:read" in scopes,
                "org_id": metadata.get("org_id"),
                "scopes": list(scopes),
                "expires_at": metadata.get("expires_at"),
            }

    try:
        return await asyncio.to_thread(run)
    except Exception:
        # This presentation gate must fail closed and must never trigger auth refresh.
        _log.debug("Collective Wisdom entitlement probe failed", exc_info=True)
        return {"entitled": False, "org_id": None, "scopes": [], "expires_at": None}


@router.get("/api/wisdom/status")
async def get_wisdom_status(profile: Optional[str] = None):
    return await _run_wisdom(
        profile, lambda service: service.status(), require_setup=False
    )


@router.get("/api/wisdom/sync")
async def get_wisdom_sync(profile: Optional[str] = None):
    return await _run_wisdom(profile, lambda service: service.sync_status())


@router.post("/api/wisdom/sync/retry")
async def post_wisdom_sync_retry(body: WisdomSyncRetryRequest):
    return await _run_wisdom(body.profile, lambda service: service.retry_sync())


def _wisdom_action_name(verb: str, profile: Optional[str]) -> str:
    scope = (profile or "current").strip().lower() or "current"
    slug = re.sub(r"[^a-z0-9]+", "-", scope).strip("-")[:32] or "current"
    digest = hashlib.sha1(scope.encode()).hexdigest()[:8]
    name = f"wisdom-{verb}-{slug}-{digest}"
    _ACTION_LOG_FILES.setdefault(name, f"action-{name}.log")
    return name


def _spawn_wisdom_action(
    verb: str, profile: Optional[str], command: List[str]
) -> dict[str, Any]:
    name = _wisdom_action_name(verb, profile)
    existing = _ACTION_PROCS.get(name)
    if existing is not None and existing.poll() is None:
        return {"ok": True, "pid": existing.pid, "name": name, "reused": True}
    args = [*_profile_cli_args(profile), "wisdom", *command, "--json"]
    proc = _spawn_hermes_action(args, name)
    return {"ok": True, "pid": proc.pid, "name": name, "reused": False}


@router.post("/api/wisdom/setup")
async def post_wisdom_setup(body: WisdomSetupRequest):
    if not body.accept_disclosure:
        raise HTTPException(
            status_code=422,
            detail="Collective Wisdom setup requires explicit disclosure acceptance",
        )
    try:
        return _spawn_wisdom_action(
            "setup", body.profile, ["setup", "--accept-disclosure"]
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to start Collective Wisdom setup")
        raise HTTPException(status_code=500, detail=f"Failed to start setup: {exc}")


@router.post("/api/wisdom/scan")
async def post_wisdom_scan(body: WisdomScanRequest):
    command = ["scan", *([body.skill] if body.skill else [])]
    try:
        return _spawn_wisdom_action("scan", body.profile, command)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to start Collective Wisdom scan")
        raise HTTPException(status_code=500, detail=f"Failed to start scan: {exc}")


@router.get("/api/wisdom/candidates")
async def get_wisdom_candidates(profile: Optional[str] = None):
    return await _run_wisdom(
        profile, lambda service: {"candidates": service.scan_candidates()}
    )


@router.get("/api/wisdom/events")
async def get_wisdom_events(
    profile: Optional[str] = None, session_id: Optional[str] = None
):
    def read(service):
        from hermes_wisdom.mediation import delivery_mode

        if delivery_mode() == "agent":
            return {"events": [], "delivery_mode": "agent"}
        return {
            "events": service.pending_candidate_events(
                session_id=session_id or "", surface="desktop"
            )
        }

    return await _run_wisdom(profile, read)


@router.get("/api/wisdom/drafts")
async def get_wisdom_drafts(profile: Optional[str] = None):
    return await _run_wisdom(
        profile,
        lambda service: {
            "drafts": [
                draft.model_dump(mode="json") for draft in service.client.list_drafts()
            ]
        },
    )


@router.get("/api/wisdom/discovery")
async def get_wisdom_discovery(profile: Optional[str] = None):
    return await _run_wisdom(profile, lambda service: service.list_skills())


@router.get("/api/wisdom/skills/{skill_id}")
async def get_wisdom_skill(skill_id: str, profile: Optional[str] = None):
    # Command views intentionally show human-readable slugs. Accept the same
    # opaque-ID-or-exact-slug references as `/wisdom show` so Desktop preview
    # actions can consume command output without leaking registry IDs into it.
    return await _run_wisdom(profile, lambda service: service.resolve_skill(skill_id))


@router.get("/api/wisdom/skills/{skill_id}/versions/{version}")
async def get_wisdom_version(
    skill_id: str, version: int, profile: Optional[str] = None
):
    return await _run_wisdom(
        profile, lambda service: service.version_detail(skill_id, version)
    )


@router.get("/api/wisdom/skills/{skill_id}/versions/{version}/content")
async def get_wisdom_version_content(
    skill_id: str, version: int, profile: Optional[str] = None
):
    return await _run_wisdom(
        profile, lambda service: service.version_content(skill_id, version)
    )


@router.post("/api/wisdom/suggest")
async def post_wisdom_suggest(body: WisdomSuggestRequest):
    result = await _run_wisdom(
        body.profile,
        lambda service: service.suggest(
            body.skill,
            description=body.description,
            system_specification=body.system_specification,
            allow_private_secret_review=body.send_for_owner_only_server_review,
            local_skill_id=body.local_skill_id,
        ),
    )
    if result.get("network_submission") is False and result.get("local_draft_id"):
        _schedule_wisdom_professionalism_reviews(body.profile)
    return result


@router.post("/api/wisdom/review")
async def post_wisdom_review(body: WisdomReviewRequest):
    return await _run_wisdom(
        body.profile,
        lambda service: service.review(
            body.draft_id, acknowledge=body.acknowledge, portal=False,
            expected_hashes=body.expected_hashes,
        ),
    )


@router.post("/api/wisdom/publication/review")
async def post_wisdom_publication_review(body: WisdomDecisionRequest):
    return await _run_wisdom(body.profile, lambda service: service.publication_review(body.draft_id))


@router.post("/api/wisdom/publication/submit")
async def post_wisdom_publication_submit(body: WisdomPublicationRequest):
    if set(body.expected_hashes) != {"content", "author_description", "package_manifest"}:
        raise HTTPException(status_code=422, detail="Confirmation requires all three displayed package hashes")
    if body.interaction_id or body.session_id:
        if not body.interaction_id or not body.session_id:
            raise HTTPException(status_code=422, detail="Both interaction and session are required")
        def submit(service):
            from hermes_wisdom.consent import ConsentActor, WisdomConsent
            actor = ConsentActor(body.session_id, "local", "local-user", f"local:{body.session_id}")
            return WisdomConsent(service).submit_local_publication(
                service.store.active_org_id(), body.interaction_id, actor,
                draft_id=body.draft_id, expected_hashes=body.expected_hashes, publication_mode=body.publication_mode,
            )
        return await _run_wisdom(body.profile, submit)
    return await _run_wisdom(
        body.profile,
        lambda service: service.submit_reviewed_package(
            body.draft_id, expected_hashes=body.expected_hashes,
            publication_mode=body.publication_mode,
        ),
    )


@router.post("/api/wisdom/prepared/save")
async def post_wisdom_prepared_save(body: WisdomPreparedSaveRequest):
    result = await _run_wisdom(
        body.profile,
        lambda service: service.save_prepared(
            body.draft_id,
            author_description=body.author_description,
            files=[item.model_dump() for item in body.files],
        ),
    )
    _schedule_wisdom_professionalism_reviews(body.profile)
    return result


@router.post("/api/wisdom/candidates/dismiss")
async def post_wisdom_candidate_dismiss(body: WisdomCandidateDismissRequest):
    return await _run_wisdom(
        body.profile,
        lambda service: service.dismiss_local_candidate(
            body.local_skill_id, body.content_hash
        ),
    )


@router.post("/api/wisdom/candidates/defer")
async def post_wisdom_candidate_defer(body: WisdomCandidateEventRequest):
    return await _run_wisdom(
        body.profile,
        lambda service: service.defer_candidate_prompt(
            body.event_id, surface="desktop"
        ),
    )


@router.post("/api/wisdom/candidates/prepare")
async def post_wisdom_candidate_prepare(body: WisdomCandidateEventRequest):
    result = await _run_wisdom(
        body.profile, lambda service: service.prepare_candidate(body.event_id)
    )
    if result.get("stage") == "prepared":
        _schedule_wisdom_professionalism_reviews(body.profile)
    return result


@router.post("/api/wisdom/candidates/approve")
async def post_wisdom_candidate_approve(body: WisdomCandidateEventRequest):
    return await _run_wisdom(
        body.profile, lambda service: service.approve_candidate(body.event_id)
    )


@router.post("/api/wisdom/revise")
async def post_wisdom_revise(body: WisdomReviseRequest):
    return await _run_wisdom(
        body.profile,
        lambda service: service.revise(
            body.draft_id,
            author_description=body.author_description,
            files=[item.model_dump() for item in body.files],
            expected_content_hash=body.expected_content_hash,
            expected_description_hash=body.expected_author_description_hash,
            expected_manifest_hash=body.expected_package_manifest_hash,
            allow_private_secret_review=body.send_for_owner_only_server_review,
        ),
    )


@router.post("/api/wisdom/approve")
async def post_wisdom_approve(body: WisdomDecisionRequest):
    return await _run_wisdom(
        body.profile, lambda service: service.approve(body.draft_id)
    )


@router.post("/api/wisdom/decline")
async def post_wisdom_decline(body: WisdomDecisionRequest):
    return await _run_wisdom(
        body.profile, lambda service: service.decline(body.draft_id)
    )


@router.post("/api/wisdom/install/plan")
async def post_wisdom_install_plan(body: WisdomInstallPlanRequest):
    return await _run_wisdom(
        body.profile,
        lambda service: service.install_plan(
            body.reference, update_mode=body.update_mode
        ),
    )


@router.post("/api/wisdom/install/apply")
async def post_wisdom_install_apply(body: WisdomInstallApplyRequest):
    return await _run_wisdom(
        body.profile,
        lambda service: service.install_apply(
            body.receipt, accept_partial=body.accept_partial
        ),
    )


@router.get("/api/wisdom/installations")
async def get_wisdom_installations(profile: Optional[str] = None):
    def read(service):
        from hermes_wisdom.mediation import delivery_mode

        return {
            "delivery_mode": delivery_mode(),
            "installations": service.store.installations(),
            "notifications": service.notifications(mark_seen=False)["events"],
        }

    return await _run_wisdom(profile, read)


@router.post("/api/wisdom/check")
async def post_wisdom_check(body: WisdomCheckRequest):
    return await _run_wisdom(
        body.profile,
        lambda service: service.check(apply_automatic=body.apply_automatic),
    )


@router.post("/api/wisdom/update/plan")
async def post_wisdom_update_plan(body: WisdomUpdatePlanRequest):
    return await _run_wisdom(
        body.profile, lambda service: service.update_plan(body.skill_id)
    )


@router.post("/api/wisdom/update/apply")
async def post_wisdom_update_apply(body: WisdomUpdateApplyRequest):
    return await _run_wisdom(
        body.profile,
        lambda service: service.update_apply(
            body.receipt,
            accept_sensitive=body.accept_sensitive,
            accept_partial=body.accept_partial,
            preserve_modified=body.preserve_modified,
        ),
    )


@router.post("/api/wisdom/uninstall")
async def post_wisdom_uninstall(body: WisdomUninstallRequest):
    return await _run_wisdom(
        body.profile, lambda service: service.uninstall(body.skill_id)
    )


@router.post("/api/wisdom/notifications")
async def post_wisdom_notifications(body: WisdomNotificationRequest):
    return await _run_wisdom(
        body.profile,
        lambda service: service.notifications(mark_seen=body.mark_seen),
    )


@router.get("/api/wisdom/mediation")
async def get_wisdom_mediation(profile: Optional[str] = None):
    from hermes_wisdom.mediation import WisdomMediation
    from hermes_wisdom.mediation_view import desktop_interaction

    def activity(service):
        value = WisdomMediation(service).activity()
        return {**value, "interactions": [desktop_interaction(item) for item in value["interactions"]]}

    return await _run_wisdom(profile, activity)


@router.get("/api/wisdom/mute")
async def get_wisdom_mute(profile: Optional[str] = None):
    from hermes_wisdom.preferences import WisdomPreferences

    return await _run_wisdom(profile, lambda service: WisdomPreferences(service).native_mute_command())


@router.post("/api/wisdom/mute/prepare")
async def post_wisdom_mute_prepare(body: WisdomMutePrepareRequest):
    def prepare(service):
        from hermes_wisdom.preferences import WisdomPreferences

        preferences = WisdomPreferences(service)
        org = service.store.active_org_id()
        control = preferences.prepare_mute_control(org)
        return {**control, "organization_id": org, "sync": preferences.mute_status(org)}

    return await _run_wisdom(body.profile, prepare)


@router.post("/api/wisdom/mute/choose")
async def post_wisdom_mute_choose(body: WisdomMuteChooseRequest):
    def choose(service):
        from hermes_wisdom.preferences import WisdomPreferences

        preferences = WisdomPreferences(service)
        preferences.choose_mute_control(service.store.active_org_id(), body.control_id, body.duration)
        return preferences.native_mute_command()

    return await _run_wisdom(body.profile, choose)


@router.post("/api/wisdom/consent")
async def post_wisdom_consent(body: WisdomConsentRequest):
    def resolve(service):
        from hermes_wisdom.consent import ConsentActor, WisdomConsent
        from hermes_wisdom.mediation_view import desktop_interaction

        actor = ConsentActor(body.session_id, "local", "local-user", f"local:{body.session_id}")
        return desktop_interaction(WisdomConsent(service).resolve(
            service.store.active_org_id(), body.interaction_id, actor, body.action
        ))
    return await _run_wisdom(body.profile, resolve)


@router.post("/api/wisdom/consent/publication-review")
async def post_wisdom_consent_publication_review(body: WisdomConsentRequest):
    def prepare(service):
        from hermes_wisdom.consent import ConsentActor, WisdomConsent
        actor = ConsentActor(body.session_id, "local", "local-user", f"local:{body.session_id}")
        return WisdomConsent(service).prepare_local_publication(service.store.active_org_id(), body.interaction_id, actor)
    return await _run_wisdom(body.profile, prepare)
