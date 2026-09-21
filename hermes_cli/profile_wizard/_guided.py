from __future__ import annotations

from typing import Any

from rich import box

from ._data import (
    OnboardingAnswers,
    ProfileIdea,
    AgentProfileSpec,
    INTEREST_OPTIONS,
    USER_TYPE_OPTIONS,
    USE_CASE_OPTIONS,
    HERMES_SETUP_OPTIONS,
    TARGET_PLATFORM_OPTIONS,
    APP_TOOL_OPTIONS,
    AUTOMATION_COMFORT_OPTIONS,
    SENSITIVE_AREA_OPTIONS,
    profile_idea_by_id,
    profile_ideas,
)
from ._ui import (
    console,
    Panel,
    ACCENT,
    WARNING,
    BACK_SENTINEL,
    _single,
    _multi,
    _text,
    _confirm,
    _advance_step_on_back,
    _select_provider,
)
from ._scoring import _score_profile_ideas, _spec_from_onboarding, _boundaries_from_onboarding, _recommended_bundle_keys, _choose_skills
from ._builder import browse_profile_ideas


def guided_onboarding(initial_name: str | None = None) -> list[AgentProfileSpec] | str | None:
    """Plain-language onboarding interview with real back navigation."""
    console.print(Panel(
        "Answer a few plain-language questions. Hermes will recommend profile blueprints and skill bundles.\n\n"
        "Nothing is created until you approve the final preview.",
        title="Guided Setup",
        border_style=ACCENT,
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    user_type = "individual"
    working_context = "organise"
    user_context = "I want Hermes to help me organise work and build useful workflows."
    interests: list[str] = []
    use_cases: list[str] = []
    ideas = ""
    setup_style = "recommend"
    platforms: list[str] = ["cli"]
    apps: list[str] = []
    other_apps = ""
    automation_comfort = "draft_prepare"
    sensitive_areas: list[str] = ["credentials", "external_messages"]

    step = 0
    total = 12
    while step < total:
        step += 1
        if step == 1:
            r = _single("About you", "Which best describes you?", USER_TYPE_OPTIONS, default=user_type)
            if r == BACK_SENTINEL:
                return BACK_SENTINEL
            user_type = str(r)
        elif step == 2:
            r = _single("Working context", "Where will Hermes fit first?", USE_CASE_OPTIONS[:8] + [("mixed", "Mixed use")], default=working_context)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            working_context = str(r)
        elif step == 3:
            r = _text("Context", "Tell Hermes anything useful about you", default=user_context)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            user_context = str(r)
        elif step == 4:
            r = _multi("Interests", "What are you interested in?", INTEREST_OPTIONS, default=interests)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            interests = list(r) if isinstance(r, list) else interests
        elif step == 5:
            r = _multi("Use cases", "What do you want Hermes to help you achieve?", USE_CASE_OPTIONS, default=use_cases)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            use_cases = list(r) if isinstance(r, list) else use_cases
        elif step == 6:
            r = _text("Ideas", "Any specific ideas you already have?", default=ideas)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            ideas = str(r)
        elif step == 7:
            r = _single("Setup style", "What kind of Hermes setup do you want first?", HERMES_SETUP_OPTIONS, default=setup_style)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            setup_style = str(r)
        elif step == 8:
            r = _multi("Surfaces", "Where do you want to use Hermes?", TARGET_PLATFORM_OPTIONS + [("scheduled_jobs", "Scheduled/background jobs"), ("not_sure", "Not sure yet")], default=platforms)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            platforms = [p for p in (list(r) if isinstance(r, list) else platforms) if p != "not_sure"]
        elif step == 9:
            r = _multi("Apps and tools", "Which apps/tools matter to you?", APP_TOOL_OPTIONS, default=apps)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            apps = list(r) if isinstance(r, list) else apps
        elif step == 10:
            r = _text("Other apps", "Any other daily tools Hermes should know about?", default=other_apps)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            other_apps = str(r)
        elif step == 11:
            r = _single("Automation comfort", "How much should Hermes do without asking?", AUTOMATION_COMFORT_OPTIONS, default=automation_comfort)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            automation_comfort = str(r)
        elif step == 12:
            r = _multi("Sensitive areas", "Where should Hermes be careful?", SENSITIVE_AREA_OPTIONS, default=sensitive_areas)
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            sensitive_areas = list(r) if isinstance(r, list) else sensitive_areas

    answers = OnboardingAnswers(
        user_type=user_type,
        working_context=working_context,
        user_context=user_context,
        interests=interests,
        use_cases=use_cases,
        ideas=ideas,
        setup_style=setup_style,
        platforms=platforms,
        apps=apps,
        other_apps=other_apps,
        automation_comfort=automation_comfort,
        sensitive_areas=sensitive_areas,
    )
    recommendations = _score_profile_ideas(answers)
    body = "Based on your answers, Hermes recommends:\n\n" + "\n".join(
        f"{idx + 1}. {idea.avatar_emoji} {idea.name}\n   {idea.description}" for idx, idea in enumerate(recommendations[:5])
    )
    body += "\n\nSkill bundles suggested: " + ", ".join(_recommended_bundle_keys(answers))
    console.print(Panel(body, title="Recommended Setup", border_style=ACCENT, box=box.ROUNDED, padding=(1, 2)))
    # Recommend profiles, let user select which to configure
    values = [(idea.id, f"{idea.avatar_emoji} {idea.name} — {idea.description}") for idea in recommendations[:5]]
    values.insert(0, ("__all__", "Create all recommended profiles"))
    values.append(("__first__", "Create only the first recommended profile"))
    values.append(("__choose__", "Choose which profiles to configure"))
    values.append(("browse", "Browse all profile ideas instead"))
    values.append(("back", "Back to profile workspace"))
    values.append(("exit", "Exit"))
    selected = _single(
        "Choose starting profile",
        "Pick one or more profiles to build. You can add more later.",
        values,
        default="__all__",
    )
    if selected in {BACK_SENTINEL, "back"}:
        return BACK_SENTINEL
    if selected == "exit":
        return None
    if selected == "browse":
        return browse_profile_ideas(initial_name=initial_name)
    # Build list of selected profile specs
    selected_ideas: list[ProfileIdea] = []
    if selected == "__all__":
        selected_ideas = list(recommendations[:5])
    elif selected == "__first__":
        selected_ideas = [recommendations[0]]
    elif selected == "__choose__":
        # Multi-select from recommendations
        picked = _multi(
            "Select profiles to configure",
            "Choose one or more profiles.",
            [(idea.id, f"{idea.avatar_emoji} {idea.name} — {idea.description}") for idea in recommendations[:5]],
            default=[recommendations[0].id],
        )
        if picked == BACK_SENTINEL:
            return BACK_SENTINEL
        if isinstance(picked, list):
            for pid in picked:
                idea = profile_idea_by_id(str(pid))
                if idea:
                    selected_ideas.append(idea)
    else:
        idea = profile_idea_by_id(str(selected))
        if idea:
            selected_ideas.append(idea)
    if not selected_ideas:
        return _spec_from_onboarding(answers, recommendations[0], initial_name=initial_name)
    return [
        _spec_from_onboarding(answers, idea, initial_name=initial_name)
        for idea in selected_ideas
    ]
