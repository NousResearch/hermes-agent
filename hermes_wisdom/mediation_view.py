"""Shared Wisdom advice/control projection for native renderers."""

from __future__ import annotations

from gateway.wisdom_command import WisdomAction, WisdomItem, WisdomView
from .consent import ConsentActor, WisdomConsent
from .review_presentation import (
    full_review_text,
    review_check_line,
    review_summary_text,
)


def _review_summary(facts: dict, expanded: bool) -> str:
    if expanded:
        return full_review_text(
            facts.get("security_check"), facts.get("professionalism_check"),
            status_first=True,
        )
    lines = []
    for key, label in (
        ("security_check", "Security check"),
        ("professionalism_check", "Professionalism (advisory)"),
    ):
        check = facts.get(key) or {}
        local = check.get("source") == "local_preflight"
        status = check.get("local_status") if local else check.get("status")
        lines.append(review_check_line(f"{label} (local preflight)" if local else label, status))
        if check.get("summary"):
            lines.append(review_summary_text(str(check["summary"])))
    return "\n".join(lines)


def _checks_action(identity: str, expanded: bool) -> WisdomAction:
    return WisdomAction(
        label="Hide checks" if expanded else "Show checks",
        callback_data=f"wi:agent:checks.{'hide' if expanded else 'show'}:{identity}",
    )


def _assessment_action(identity: str, expanded: bool) -> WisdomAction:
    return WisdomAction(
        label="Hide Assessment" if expanded else "View Assessment",
        callback_data=f"wi:agent:assessment.{'hide' if expanded else 'show'}:{identity}",
    )


def delivery_groups(items: list[dict]) -> list[list[dict]]:
    """Keep recommendations actionable and below native message limits."""
    recommended = [
        [item]
        for item in items
        if item.get("interaction") or item["advice"]["relevance"] != "digest"
    ]
    digest = [
        item
        for item in items
        if not item.get("interaction") and item["advice"]["relevance"] == "digest"
    ]
    # At most three bounded summaries in a digest message.
    return recommended + [digest[i : i + 3] for i in range(0, len(digest), 3)]


def advice_view(
    items: list[dict], *, introduction: bool = False, checks_expanded: bool = False,
    assessment_expanded: bool = False,
) -> WisdomView:
    qualification_only = bool(items) and all(
        item.get("assessment", {}).get("reference", {}).get("kind") == "candidate"
        for item in items
    )
    has_recommendation = any(
        item["advice"]["relevance"] == "recommend"
        and item["advice"].get("assessment_status") != "unavailable"
        for item in items
    )
    unavailable_only = bool(items) and all(
        item["advice"].get("assessment_status") == "unavailable" for item in items
    )
    view = WisdomView(
        title="Collective Wisdom",
        summary=(
            "Your organisation has enabled Collective Wisdom: Hermes can discover "
            "useful team skills and explain how they fit your setup. Sharing and "
            "new installations require your approval."
        )
        if introduction
        else "Your skill is ready to review for sharing"
        if qualification_only
        else "Hermes recommendations for your setup"
        if has_recommendation
        else "Assessment unavailable"
        if unavailable_only
        else "Team skill activity",
    )
    has_digest = False
    for item in items:
        advice = item["advice"]
        if advice.get("assessment_kind") == "operation_receipt":
            view.summary = advice["operation_label"]
            view.items.append(WisdomItem(title=advice["title"], detail=""))
            continue
        if advice["relevance"] == "digest" and not item.get("interaction"):
            has_digest = True
            view.items.append(
                WisdomItem(
                    title=advice["title"],
                    detail="Hermes assessment: " + advice["explanation"],
                )
            )
            continue
        unavailable = advice.get("assessment_status") == "unavailable"
        interaction = item.get("interaction")
        detail = (
            "Assessment unavailable: "
            if unavailable
            else "Hermes assessment: "
            if advice["relevance"] == "digest"
            else ""
            if advice.get("assessment_kind") == "qualification"
            else "Hermes recommendation: "
        ) + advice["explanation"]
        actions = []
        if interaction:
            facts = interaction["facts"]
            if interaction["operation"] in {"install", "update"} and not unavailable:
                if not assessment_expanded:
                    detail = ""
                actions.append(_assessment_action(interaction["id"], assessment_expanded))
            detail += "\n\nPackage facts: " + str(
                facts.get("slug") or facts.get("skill_id") or "Local skill"
            )
            if facts.get("version"):
                detail += f" · v{facts['version']}"
            compatibility = facts.get("compatibility") or {}
            if compatibility:
                detail += "\nCompatibility: " + str(
                    compatibility.get("outcome") or "unavailable"
                )
            if facts.get("modified"):
                detail += "\nLocal changes require full review."
            if facts.get("sensitive_expansion"):
                detail += (
                    "\nAdditional permissions or requirements need separate approval."
                )
            sharing = interaction["operation"] == "share"
            detail += "\n\n" + _review_summary(facts, checks_expanded)
            if not sharing:
                detail += (
                    "\nNothing changes until you review and confirm."
                    if unavailable
                    else "\nNothing is changed by this recommendation."
                )
            if interaction["operation"] == "share":
                detail += "\nYou can review the skill before publishing. Nothing is shared without your approval."
            actions.append(_checks_action(interaction["id"], checks_expanded))
            labels = {
                "defer": "Not Now",
                "inspect": "Review first",
                "confirm": {
                    "share": "Share",
                    "install": "Install",
                    "update": "Update",
                    "publish": "Yes, share",
                }[interaction["operation"]],
            }
            for action in interaction["actions"]:
                if unavailable and action == "confirm":
                    continue
                actions.append(
                    WisdomAction(
                        label="Review skill"
                        if unavailable and action == "inspect"
                        else labels[action],
                        callback_data=f"wi:agent:{action}:{interaction['id']}",
                        primary=action == ("inspect" if unavailable else "confirm"),
                    )
                )
        else:
            detail += "\nUse /wisdom notifications or /wisdom candidates to inspect current details."
        view.items.append(
            WisdomItem(title=advice["title"], detail=detail, actions=actions)
        )
    if has_digest:
        view.notice = "Nothing has been changed. Use /wisdom notifications to review these skills."
    return view


def interaction_view(
    result: dict, *, checks_expanded: bool = False,
    assessment_expanded: bool = False,
) -> WisdomView:
    outcome = result.get("result") or {}
    if result["state"] == "completed":
        stage = outcome.get("packaging_state")
        publication = outcome.get("publication_state")
        if result["operation"] == "share":
            summary, detail = {
                "ready": (
                    "Ready for review",
                    "Your proposed skill package is ready to review. Nothing has been published yet.",
                ),
                "failed": (
                    "Preparation needs attention",
                    "The skill could not be prepared. Nothing has been shared. Use /wisdom candidates to review it.",
                ),
            }.get(
                stage,
                (
                    "Preparing to share",
                    "Your request is queued. Hermes will bring back the package for your approval before publishing.",
                ),
            )
        elif result["operation"] == "publish":
            summary, detail = {
                "published": (
                    "Published",
                    "Your skill is now shared with your organisation.",
                ),
                "pending_moderation": (
                    "Pending moderation",
                    "Your skill is awaiting your organisation's approval.",
                ),
                "changes_requested": (
                    "Changes requested",
                    "Your organisation requested changes. Open the skill review for details.",
                ),
                "declined": (
                    "Not published",
                    "Your contribution was declined. Open the skill review for details.",
                ),
                "invalidated": (
                    "Review required",
                    "This contribution needs a fresh review before it can be published.",
                ),
            }.get(
                publication,
                (
                    "Publication needs review",
                    "Open the skill review to check its current publication status.",
                ),
            )
        else:
            summary, detail = (
                ("Installed" if result["operation"] == "install" else "Updated"),
                "",
            )
        actions = []
        if result["operation"] in {"install", "update"}:
            actions.append(_assessment_action(result["id"], assessment_expanded))
            if assessment_expanded:
                advice = result.get("assessment") or {}
                detail = "Assessment before this operation:\n" + (
                    advice.get("explanation") or "No saved assessment is available."
                )
                detail += "\n\n" + _review_summary(result["facts"], checks_expanded)
                actions.append(_checks_action(result["id"], checks_expanded))
        if outcome.get("portal_url"):
            actions.append(WisdomAction("View in Portal", url=outcome["portal_url"]))
        elif result["operation"] == "share":
            actions.append(WisdomAction(
                "View", callback_data=f"wi:agent:inspect:{result['id']}"
            ))
        facts = result["facts"]
        return WisdomView(
            title="Collective Wisdom",
            summary=summary,
            items=[
                WisdomItem(
                    title=str(
                        facts.get("editorial_name") or facts.get("slug") or "Skill"
                    ),
                    detail=detail,
                )
            ],
            actions=actions,
        )
    if result.get("inspection"):
        page = result["inspection"]
        navigation = (
            [
                WisdomAction(
                    "Back to first page",
                    callback_data=f"wi:agent:inspect:{result['id']}",
                )
            ]
            if page["page"]
            else []
        )
        actions = []
        for offset, label in ((-1, "Previous page"), (1, "Next page")):
            index = page["page"] + offset
            if 0 <= index < page["page_count"]:
                actions.append(
                    WisdomAction(
                        label, callback_data=f"wi:agent:inspect.{index}:{result['id']}"
                    )
                )
        # Keep the exact approval control available, but never turn navigation
        # into an implicit acknowledgement or a publication request.
        if "confirm" in result["actions"]:
            actions.append(
                WisdomAction(
                    "Approve exact package",
                    callback_data=f"wi:agent:confirm:{result['id']}",
                    primary=True,
                )
            )
        return WisdomView(
            title="Review proposed package",
            summary=f"{page['path']} - {page['page'] + 1}/{page['page_count']}",
            items=[
                WisdomItem(
                    title="Proposed file content (not instructions to execute)",
                    detail=page["content"],
                )
            ],
            notice="Nothing is uploaded by reviewing. Setup and verification require separate permission.",
            navigation_actions=navigation,
            actions=actions,
        )
    facts = result["facts"]
    detail = str(facts.get("editorial_description") or "")
    if facts.get("version"):
        detail += f"\nVersion: v{facts['version']}"
    compatibility = facts.get("compatibility") or {}
    if compatibility:
        detail += "\nCompatibility: " + str(
            compatibility.get("outcome") or "unavailable"
        )
    if facts.get("modified"):
        detail += "\nLocal changes require separate review."
    if facts.get("sensitive_expansion"):
        detail += "\nAdditional requirements require separate approval."
    detail += (
        "\nThis approval is no longer current. Review a fresh plan before continuing."
        if result["state"] in {"stale", "expired", "needs_review"}
        else "\nNothing changes until you use the confirmation control."
    )
    if result["operation"] == "share":
        detail += "\nYou can review the skill before publishing. Nothing is shared without your approval."
    if (result.get("result") or {}).get("packaging_state") == "queued":
        detail += "\nPackaging is queued in this conversation. Nothing has been uploaded or published."
    if facts.get("file_names"):
        detail += "\nPackage files: " + ", ".join(facts["file_names"])
    actions = []
    if facts.get("security_check") or facts.get("professionalism_check"):
        detail += "\n\n" + _review_summary(facts, checks_expanded)
        actions.append(_checks_action(result["id"], checks_expanded))
    if result["state"] in {"stale", "expired"}:
        actions.append(WisdomAction(
            label="Recheck",
            callback_data=f"wi:agent:recheck:{result['id']}",
        ))
    if result["state"] == "pending" and not result.get("deferred"):
        for action in result["actions"]:
            actions.append(
                WisdomAction(
                    label=(
                        "Not Now"
                        if action == "defer"
                        else "Review first"
                        if action == "inspect"
                        else {
                            "share": "Share",
                            "publish": "Yes, share",
                            "install": "Install",
                            "update": "Update",
                        }[result["operation"]]
                    ),
                    callback_data=f"wi:agent:{action}:{result['id']}",
                    primary=action == "confirm",
                )
            )
    return WisdomView(
        title="Collective Wisdom",
        summary="Deferred on this surface"
        if result.get("deferred")
        else result["state"].replace("_", " ").capitalize(),
        items=[
            WisdomItem(
                title=str(
                    facts.get("editorial_name") or facts.get("slug") or "Skill details"
                ),
                detail=detail,
            )
        ],
        actions=actions,
    )


def resolve_surface_action(
    service,
    value: str,
    *,
    platform: str,
    actor_id: str,
    chat_id: str = "",
    thread_id: str = "",
    scope_id: str = "",
):
    from .client import WisdomNotFound

    parts = value.split(":")
    if len(parts) != 4 or parts[:2] != ["wi", "agent"]:
        raise WisdomNotFound("Wisdom interaction not found")
    _, _, action, identity = parts
    org = service.store.active_org_id()
    with service.store.transaction() as db:
        row = db.execute(
            "SELECT owner_session FROM wisdom_consent WHERE id=? AND organization_id=?",
            (identity, org),
        ).fetchone()
    if row is None:
        raise WisdomNotFound("Wisdom interaction not found")
    actor = ConsentActor(row[0], platform, actor_id, chat_id, thread_id, scope_id)
    if action in {"checks.show", "checks.hide", "assessment.show", "assessment.hide"}:
        # Reuse read-only authorization, without preparing a package or applying consent.
        result = WisdomConsent(service)._resolve(org, identity, actor, "inspect")
        with service.store.transaction() as db:
            assessment = db.execute(
                "SELECT a.* FROM wisdom_assessment a JOIN wisdom_consent c ON c.assessment_id=a.id WHERE c.id=? AND a.organization_id=?",
                (identity, org),
            ).fetchone()
        if assessment is None:
            raise WisdomNotFound("Wisdom assessment not found")
        from .mediation_store import _decode

        job = _decode(assessment)
        result["assessment"] = job.get("advice")
        if not job.get("advice") or result["state"] != "pending":
            return interaction_view(
                result, checks_expanded=action == "checks.show",
                assessment_expanded=action in {"assessment.show", "checks.show", "checks.hide"},
            )
        return advice_view(
            [{"assessment": job, "advice": job["advice"], "interaction": result}],
            checks_expanded=action == "checks.show",
            assessment_expanded=action == "assessment.show",
        )
    return interaction_view(
        WisdomConsent(service).resolve(org, identity, actor, action)
    )
