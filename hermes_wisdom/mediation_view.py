"""Shared Wisdom advice/control projection for native renderers."""

from __future__ import annotations

from gateway.wisdom_command import WisdomAction, WisdomItem, WisdomView
from .consent import ConsentActor, WisdomConsent
from .review_presentation import full_review_text


def delivery_groups(items: list[dict]) -> list[list[dict]]:
    """Keep recommendations actionable and below native message limits."""
    recommended = [[item] for item in items if item["advice"]["relevance"] != "digest"]
    digest = [item for item in items if item["advice"]["relevance"] == "digest"]
    # At most three bounded summaries in a digest message.
    return recommended + [digest[i : i + 3] for i in range(0, len(digest), 3)]


def advice_view(items: list[dict], *, introduction: bool = False) -> WisdomView:
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
        else "Hermes recommendations for your setup"
        if has_recommendation
        else "Assessment unavailable"
        if unavailable_only
        else "Team skill activity",
    )
    has_digest = False
    for item in items:
        advice = item["advice"]
        if advice["relevance"] == "digest":
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
            "Assessment unavailable: " if unavailable else "Hermes recommendation: "
        ) + advice["explanation"]
        actions = []
        if interaction:
            facts = interaction["facts"]
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
            for key, label in (
                ("security_check", "Security"),
                ("professionalism_check", "Professionalism (advisory)"),
            ):
                check = facts.get(key) or {}
                detail += f"\n{label}: {check.get('status') or 'unavailable'}"
            detail += (
                "\nNothing changes until you review and confirm."
                if unavailable
                else "\nNothing is changed by this recommendation."
            )
            if interaction["operation"] == "share":
                detail += "\nShare prepares a local handoff package. You will review it and approve separately before anything is uploaded or published."
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


def interaction_view(result: dict) -> WisdomView:
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
    detail += "\nNothing changes until you use the confirmation control."
    if result["operation"] == "share":
        detail += "\nShare prepares a local handoff package only. Publishing requires a separate approval after review."
    if (result.get("result") or {}).get("packaging_state") == "queued":
        detail += "\nPackaging is queued in this conversation. Nothing has been uploaded or published."
    if facts.get("file_names"):
        detail += "\nPackage files: " + ", ".join(facts["file_names"])
    if facts.get("security_check") or facts.get("professionalism_check"):
        detail += "\n\n" + full_review_text(
            facts.get("security_check"), facts.get("professionalism_check")
        )
    actions = []
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
    return interaction_view(
        WisdomConsent(service).resolve(org, identity, actor, action)
    )
