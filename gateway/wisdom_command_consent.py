"""Bind command reviews to the existing session and saved approval state."""

from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def surface_context(adapter, *, user_id, chat_id, profile, organization_id,
                    is_group=False, thread_id="", scope_id=""):
    from gateway.wisdom_command import WisdomCommandContext

    source = SessionSource(
        platform=adapter.platform, user_id=user_id, chat_id=chat_id,
        profile=profile, chat_type="group" if is_group else "dm",
        thread_id=thread_id or None, scope_id=scope_id or None,
    )
    return WisdomCommandContext(
        user_id=user_id, chat_id=chat_id, profile=profile,
        organization_id=organization_id, is_group=is_group,
        thread_id=thread_id, scope_id=scope_id,
        platform=adapter.platform.value,
        session_key=adapter._event_session_key(MessageEvent(text="", source=source)),
    )


def command_review(service, plan, context):
    from hermes_wisdom.consent import ConsentActor, WisdomConsent
    from hermes_wisdom.mediation_view import interaction_view

    local = context.chat_id.startswith("local:") and not context.platform
    platform = "local" if local else context.platform
    session_key = context.chat_id.removeprefix("local:") if local else context.session_key
    if (context.is_group or not context.user_id or not session_key
            or platform not in {"local", "telegram", "slack"}
            or (local and context.user_id != "local-user")):
        raise PermissionError("Open a private session to review this installation.")
    org = service.store.active_org_id()
    if org != context.organization_id:
        raise PermissionError("The active organization changed. Reopen the review.")
    consent = WisdomConsent(service)
    actor = ConsentActor(session_key, platform, context.user_id, context.chat_id,
                         context.thread_id, context.scope_id)
    # Commands establish ownership, not eligibility for background notifications.
    with service.store.transaction() as db:
        consent.queue._check_org(db, org)
        existing = db.execute(
            "SELECT 1 FROM wisdom_agent_session WHERE organization_id=? AND session_key=?",
            (org, session_key),
        ).fetchone()
    if not existing:
        consent.queue.register_session(
            org, session_key=session_key, session_id=session_key,
            platform=platform, actor_id=actor.actor_id, private=True,
            available=False, address=actor.address,
        )
    result = consent.request(
        org, {"kind": "skill", "skill_id": plan["skill_id"],
              "version": plan["version"], "update_mode": plan.get("update_mode")},
        actor, title=str(plan.get("slug") or plan["skill_id"]),
        explanation="You requested this exact package review.", queue_delivery=False,
    )
    return interaction_view(result)
