"""Shared Wisdom advice/control projection for native renderers."""

from __future__ import annotations

from gateway.wisdom_command import WisdomAction, WisdomItem, WisdomView, _NavigationTarget
from .consent import ConsentActor, WisdomConsent
from .review_presentation import review_card_text

_SETUP_COMMAND_LABEL = "Proposed command (local terminal)"


def _update_policy_line(interaction: dict) -> str:
    facts = interaction["facts"]
    if interaction["operation"] != "install" or "update_mode" not in facts:
        return ""
    mode = {
        None: "Organization default",
        "MANUAL": "Manual",
        "AUTO_WITH_NOTICE": "Automatic with notice",
        "REQUIRED": "Required",
    }.get(facts["update_mode"], "Unavailable")
    return f"\nFuture updates: {mode}"


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
    if len(items) == 1 and (items[0].get("interaction") or {}).get("operation") == "setup":
        return interaction_view(items[0]["interaction"])
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
        title="Hermes Collective Wisdom",
        summary=(
            "Your organization has enabled Collective Wisdom, a feature designed to "
            "automatically detect and share useful skills across all team members."
        )
        if introduction
        else "Your skill is ready to review for sharing"
        if qualification_only
        else "Hermes recommendations for your setup"
        if has_recommendation
        else "Assessment unavailable"
        if unavailable_only
        else "Team skill activity",
        actions=[WisdomAction(
            "Notification settings", "mute", local_command="/wisdom mute",
        )],
        _navigation_target=_NavigationTarget("command", {"raw_args": "inbox"}),
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
            detail += _update_policy_line(interaction)
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
            detail += "\n\n" + review_card_text(facts, checks_expanded)
            if not sharing:
                detail += (
                    "\nNothing changes until you review and confirm."
                    if unavailable
                    else "\nNothing is changed by this recommendation."
                )
            if interaction["operation"] == "share":
                detail += "\n\nWould you like to share it?"
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
                if action == "inspect" and (interaction.get("result") or {}).get("portal_url"):
                    actions.append(WisdomAction("View Skill", url=interaction["result"]["portal_url"]))
                    continue
                native_action = "review" if action == "inspect" and interaction["operation"] in {"share", "publish"} else action
                actions.append(
                    WisdomAction(
                        label="Review skill"
                        if unavailable and action == "inspect"
                        else labels[action],
                        callback_data=f"wi:agent:{native_action}:{interaction['id']}",
                        primary=action == ("inspect" if unavailable else "confirm"),
                    )
                )
        else:
            detail += "\nUse /wisdom notifications or /wisdom candidates to inspect current details."
        title = advice["title"]
        if interaction and interaction["facts"].get("local_version"):
            title += f" · local v{interaction['facts']['local_version']}"
        view.items.append(WisdomItem(title=title, detail=detail, actions=actions))
    if has_digest:
        view.notice = "Nothing has been changed. Use /wisdom notifications to review these skills."
    return view


def _setup_view(result: dict, *, include_command: bool = True) -> WisdomView:
    facts = result["facts"]
    step = facts["step"]
    state = ((result.get("result") or {}).get("setup") or {}).get("state", result["state"])
    requirement = facts.get("setup_requirement") or {}
    missing_requirement = bool(requirement) and facts.get("allowed") is False
    summary = {
        "pending": "Review setup step", "running": "Setup step running",
        "applying": "Checking setup progress", "unknown": "Setup outcome unknown",
        "passed": "Prerequisite acknowledged" if step["phase"] == "prerequisite" else "Command completed",
        "failed": "Setup step failed", "stale": "Setup needs a fresh review",
        "expired": "Setup approval expired", "needs_review": "Setup needs attention",
        "blocked": "Terminal permission required",
        "abandoned": "Interrupted step cleared",
    }.get(state, "Setup progress")
    if state == "pending" and missing_requirement:
        summary = "Prerequisite missing"
    elif state == "passed" and requirement.get("status") == "present":
        summary = "Prerequisite checked"
    detail = facts["setup_instruction"]
    if facts.get("setup_explanation"):
        detail = facts["setup_explanation"] + "\n\n" + detail
    if step["command"] and include_command:
        detail += f"\n\n{_SETUP_COMMAND_LABEL}:\n" + step["command"]
    if state == "pending" and not missing_requirement:
        detail += "\n\nOnly this step is authorized by confirming. Do not enter credentials in chat."
    elif state == "unknown":
        detail += "\n\nThe command may have run. It will not be repeated automatically."
        if ((result.get("result") or {}).get("setup") or {}).get("recovery_review"):
            detail += (
                "\n\nBefore clearing this record, check that the command and its child processes "
                "have stopped and inspect any changes they made. Clearing does not undo changes, "
                "mark setup complete, or authorize another command."
            )
    elif state == "abandoned":
        detail += "\n\nThe interrupted attempt remains unverified. Review a fresh step before continuing."
    elif state == "blocked":
        detail += "\n\nThis command did not run. Resolve terminal permissions, then request a fresh setup review."
    elif state == "passed":
        detail += "\n\nContinue setup inspection to check remaining prerequisites and verification."
    actions = []
    if result["state"] == "pending" and missing_requirement and not result.get("deferred"):
        actions = [
            WisdomAction("Not Now", callback_data=f"wi:agent:defer:{result['id']}"),
            WisdomAction("Recheck", callback_data=f"wi:agent:recheck:{result['id']}", primary=True),
        ]
    elif result["state"] == "pending" and not result.get("deferred"):
        actions = [
            WisdomAction("Not Now", callback_data=f"wi:agent:defer:{result['id']}"),
            WisdomAction("Confirm prerequisite" if step["phase"] == "prerequisite" else "Run this step",
                         callback_data=f"wi:agent:confirm:{result['id']}", primary=True),
        ]
    elif state == "unknown":
        if ((result.get("result") or {}).get("setup") or {}).get("recovery_review"):
            actions = [
                WisdomAction("Back", callback_data=f"wi:agent:inspect:{result['id']}"),
                WisdomAction("Confirmed stopped; clear record", callback_data=f"wi:agent:setup.clear:{result['id']}", primary=True),
            ]
        else:
            actions = [
                WisdomAction("Check progress", callback_data=f"wi:agent:inspect:{result['id']}"),
                WisdomAction("Review interruption", callback_data=f"wi:agent:setup.recover:{result['id']}"),
            ]
    elif result["state"] == "applying":
        actions = [WisdomAction("Check progress", callback_data=f"wi:agent:inspect:{result['id']}")]
    elif result["state"] in {"stale", "expired"} or state in {"abandoned", "blocked", "failed"}:
        actions = [WisdomAction("Recheck", callback_data=f"wi:agent:recheck:{result['id']}")]
    elif result["state"] == "completed":
        actions = [WisdomAction("Check setup", callback_data=f"wi:agent:setup.status:{result['id']}")]
    return WisdomView(
        title="Hermes Collective Wisdom", summary=summary,
        items=[WisdomItem(title=f"{facts['slug']} · v{facts['version']}", detail=detail)],
        actions=actions,
    )


def desktop_interaction(result: dict) -> dict:
    """Project existing setup controls without moving their state machine into JS."""
    import time

    is_step = result["operation"] == "setup"
    if not (is_step or result.get("setup_continuation") or (
        result["operation"] in {"install", "update"} and result["state"] == "completed"
    )):
        return result
    if result["state"] == "pending" and result["expires_at"] <= time.time():
        result = {**result, "state": "expired"}
    view = (
        _setup_view(result, include_command=False)
        if is_step and not result.get("setup_continuation") else interaction_view(result)
    )
    prefix = "wi:agent:"
    suffix = ":" + result["id"]
    actions = []
    for action in view.actions:
        callback = action.callback_data or ""
        if not callback.startswith(prefix) or not callback.endswith(suffix):
            continue
        code = callback[len(prefix):-len(suffix)]
        if code not in {"defer", "inspect", "confirm", "recheck", "setup.status", "setup.recover", "setup.clear"}:
            continue
        actions.append({"action": code, "label": action.label, "primary": action.primary})
    return {
        **result,
        "setup_review": {
            "summary": view.summary,
            "detail": "\n\n".join(item.detail for item in view.items),
            "command": result["facts"]["step"]["command"] if is_step and not result.get("setup_continuation") else "",
            "command_label": _SETUP_COMMAND_LABEL,
            "actions": actions,
        },
    }


def interaction_view(
    result: dict, *, checks_expanded: bool = False,
    assessment_expanded: bool = False,
) -> WisdomView:
    if result.get("setup_continuation"):
        continuation = result["setup_continuation"]
        return WisdomView(
            title="Hermes Collective Wisdom",
            summary={"queued": "Setup queued", "waiting_for_model": "Setup waiting for model",
                     "needs_review": "Setup needs attention", "ready": "Ready"}[continuation["state"]],
            items=[WisdomItem(title=f"{result['facts']['slug']} · v{result['facts']['version']}", detail=continuation["message"])],
            actions=[WisdomAction("Check setup", callback_data=f"wi:agent:setup.status:{result['id']}")],
        )
    if result["operation"] == "setup":
        return _setup_view(result)
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
                    "Your skill is now shared with your organization.",
                ),
                "pending_moderation": (
                    "Pending moderation",
                    "Your skill is awaiting your organization's approval.",
                ),
                "changes_requested": (
                    "Changes requested",
                    "Your organization requested changes. Open the skill review for details.",
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
                ("Files installed" if result["operation"] == "install" else "Files updated"),
                "Setup is queued. Prerequisites and verification must pass before the skill is ready.",
            )
        actions = []
        if result["operation"] in {"install", "update"}:
            actions.append(WisdomAction("Check setup", callback_data=f"wi:agent:setup.status:{result['id']}"))
            actions.append(_assessment_action(result["id"], assessment_expanded))
            if assessment_expanded:
                advice = result.get("assessment") or {}
                detail = "Assessment before this operation:\n" + (
                    advice.get("explanation") or "No saved assessment is available."
                )
                detail += "\n\n" + review_card_text(result["facts"], checks_expanded)
                actions.append(_checks_action(result["id"], checks_expanded))
        if outcome.get("portal_url"):
            actions.append(WisdomAction("View in Portal", url=outcome["portal_url"]))
        elif result["operation"] == "share":
            actions.append(WisdomAction(
                "View", callback_data=f"wi:agent:inspect:{result['id']}"
            ))
        facts = result["facts"]
        return WisdomView(
            title="Hermes Collective Wisdom",
            summary=summary,
            items=[
                WisdomItem(
                    title=str(
                        facts.get("editorial_name") or facts.get("slug") or "Skill"
                    ) + (f" · local v{facts['local_version']}" if facts.get("local_version") else ""),
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
                    callback_data=f"wi:agent:inspect.0:{result['id']}",
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
        if "defer" in result["actions"]:
            actions.append(
                WisdomAction(
                    "Not Now",
                    callback_data=f"wi:agent:defer:{result['id']}",
                )
            )
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
    detail += _update_policy_line(result)
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
    if (result.get("result") or {}).get("portal_url") and result["state"] == "pending":
        detail += "\nYour private draft is ready in the Portal. You can review and edit it before publishing."
    if (result.get("result") or {}).get("packaging_state") == "queued":
        detail += "\nPackaging is queued in this conversation. Nothing has been uploaded or published."
    if facts.get("file_names"):
        detail += "\nPackage files: " + ", ".join(facts["file_names"])
    actions = []
    if facts.get("security_check") or facts.get("professionalism_check"):
        detail += "\n\n" + review_card_text(facts, checks_expanded)
        actions.append(_checks_action(result["id"], checks_expanded))
    if (
        result["operation"] == "share"
        and result["state"] == "pending"
        and not result.get("deferred")
    ):
        detail += "\n\nWould you like to share it?"
    if result["state"] in {"stale", "expired"}:
        actions.append(WisdomAction(
            label="Recheck",
            callback_data=f"wi:agent:recheck:{result['id']}",
        ))
    if result["state"] == "pending" and not result.get("deferred"):
        for action in result["actions"]:
            if action == "inspect" and (result.get("result") or {}).get("portal_url"):
                actions.append(WisdomAction("View Skill", url=result["result"]["portal_url"]))
                continue
            native_action = "review" if action == "inspect" and result["operation"] in {"share", "publish"} else action
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
                    callback_data=f"wi:agent:{native_action}:{result['id']}",
                    primary=action == "confirm",
                )
            )
    return WisdomView(
        title="Hermes Collective Wisdom",
        summary="Deferred on this surface"
        if result.get("deferred")
        else result["state"].replace("_", " ").capitalize(),
        items=[
            WisdomItem(
                title=str(
                    facts.get("editorial_name") or facts.get("slug") or "Skill details"
                ) + (f" · local v{facts['local_version']}" if facts.get("local_version") else ""),
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
    if action == "inspect":
        current = WisdomConsent(service)._resolve(org, identity, actor, "inspect")
        if current["operation"] in {"share", "publish"} and current["state"] == "pending":
            # Older native Review first buttons used inspect. Keep CLI/tool reads read-only.
            action = "review"
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
        if not job.get("advice") or job["state"] == "passive" or result["state"] != "pending":
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
