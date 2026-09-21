from pathlib import Path
import inspect
import json

import hermes_cli.profile_wizard as profile_wizard
from hermes_cli.profile_wizard import (
    AgentProfileSpec,
    export_profile,
    generate_soul_md,
    generate_user_md,
    render_preview,
)
import hermes_cli.profile_wizard._guided as _guided_mod
import hermes_cli.profile_wizard._builder as _builder_mod
import hermes_cli.profile_wizard._scoring as _scoring_mod
import hermes_cli.profile_wizard.wizard as _wizard_mod


def _spec() -> AgentProfileSpec:
    return AgentProfileSpec(
        profile_name="demo-agent",
        display_name="Demo Agent",
        tagline="Do useful work.",
        avatar_emoji="⚕",
        primary_role="utility_bot",
        secondary_roles=["knowledge_base"],
        personality="craftsman",
        tones=["professional", "technical"],
        skill_sets={"Utility & Tools": ["automation", "reminders"]},
        custom_skills=["Answer product questions"],
        boundaries=["no_sensitive_data_disclosure"],
        blocked_topics=["secrets"],
        user_context="A small team running a Discord community.",
        intended_outcomes="Answer questions and automate routine support.",
        operating_context="Discord and Hermes CLI.",
        target_platforms=["discord", "cli"],
    )


def test_render_preview_includes_interview_model_and_skill_count():
    preview = render_preview(_spec())

    assert "Demo Agent" in preview
    assert "Interview summary:" in preview
    assert "A small team running a Discord community." in preview
    assert "gpt-5.5 via openai-codex" in preview
    assert "Total skills: 3" in preview


def test_generated_profile_files_include_model_and_interview_context():
    spec = _spec()

    soul = generate_soul_md(spec)
    user = generate_user_md(spec)

    assert "Default backend reference: `gpt-5.5` via `openai-codex`" in soul
    assert "Target user/context: A small team running a Discord community." in soul
    assert "Primary role: `utility_bot`" in soul
    assert "after a full interview" in user
    assert "Intended outcomes: Answer questions and automate routine support." in user


def test_export_json_and_js(tmp_path: Path):
    spec = _spec()

    json_path = export_profile(spec, "json", tmp_path)
    js_path = export_profile(spec, "js", tmp_path)

    assert json_path is not None
    assert js_path is not None
    data = json.loads(json_path.read_text())
    assert data["model"] == "gpt-5.5"
    assert data["user_context"] == "A small team running a Discord community."
    assert js_path.read_text().startswith("export default ")


def test_provider_rows_match_model_setup_order_and_active_focus_markers():
    entries = profile_wizard._provider_entries()

    assert entries[0].tui_desc == "Nous Portal (Everything your agent needs, 300+ models with bundled tool use)"
    # KENSEI CUSTOM: upstream added a `fireworks` wizard entry at index 1 without
    # bumping this test's fixed indices — resolve dynamically instead of snapshotting.
    by_slug = {e.slug: e for e in entries}
    assert by_slug["openrouter"].tui_desc == "OpenRouter (Pay-per-use API aggregator)"
    assert by_slug["novita"].tui_desc == "NovitaAI (Cloud: Model API, Agent Sandbox, GPU Cloud)"
    codex = by_slug["openai-codex"]
    assert entries[-1].tui_desc == "Ollama Cloud (Cloud-hosted open models, ollama.com)"

    row = profile_wizard._format_provider_row(codex, focused=True, active=True)
    unfocused = profile_wizard._format_provider_row(by_slug["novita"], focused=False, active=False)

    # The row is composed from the entry's real display copy (tui_desc), which
    # is product copy and may change. Assert on the row's *structure* — the
    # focus marker, the active marker, and that the provider's own tui_desc is
    # embedded — rather than freezing a specific marketing label.
    assert row.startswith(" → (●) ")
    assert row.endswith("← currently active")
    assert codex.tui_desc in row
    assert unfocused.startswith("   (○) ")
    assert by_slug['novita'].tui_desc in unfocused


def test_profile_ideas_include_ootb_examples_leads_and_workers():
    ids = {idea.id for idea in profile_wizard.profile_ideas()}

    assert {"general_assistant", "research_assistant", "coding_assistant", "admin_assistant_starter", "content_assistant", "knowledge_base", "automation_assistant", "community_assistant"}.issubset(ids)
    assert {"nous_girl", "senter", "chizul", "klerik"}.issubset(ids)
    assert {"kensei_orchestrator", "research_lead", "coding_lead", "qa_lead"}.issubset(ids)
    assert {"code_worker", "research_worker", "qa_worker", "content_worker"}.issubset(ids)


def test_profile_idea_to_spec_prefills_defaults_without_writing():
    idea = profile_wizard.profile_idea_by_id("senter")
    assert idea is not None

    spec = profile_wizard.spec_from_profile_idea(idea)

    assert spec.source_idea == "senter"
    assert spec.source_idea_name == "Senter-style Triage Orchestrator"
    assert spec.display_name == "Senter-style Triage Orchestrator"
    assert spec.model_provider == "nous"
    assert spec.model == "qwen/qwen3.6-plus"
    assert spec.primary_role == "devops_admin"
    assert "automation" in spec.skill_sets["Utility & Tools"]
    assert "Started from: Senter-style Triage Orchestrator" in render_preview(spec)


def test_chizul_and_nous_girl_examples_reflect_reviewed_profiles():
    chizul = profile_wizard.spec_from_profile_idea(profile_wizard.profile_idea_by_id("chizul"))
    nous_girl = profile_wizard.spec_from_profile_idea(profile_wizard.profile_idea_by_id("nous_girl"))

    assert chizul.tagline == "Do, verify, report."
    assert chizul.personality == "craftsman"
    assert nous_girl.tagline == "Yes, and — then capture the idea."
    assert nous_girl.model == "qwen/qwen3.6-flash"


def test_choice_rows_use_required_selectable_markers():
    focused = profile_wizard._format_choice_row("Browse recommended profile ideas", focused=True)
    active = profile_wizard._format_choice_row("Create a custom profile", focused=False, active=True)

    assert focused.startswith(" → (○) Browse recommended profile ideas")
    assert active.startswith("   (●) Create a custom profile")
    assert active.endswith("← currently active")


def test_text_input_preserves_backspace_for_deleting_text():
    source = inspect.getsource(profile_wizard._text)

    assert '@kb.add("c-h")' not in source
    assert '@kb.add("c-b")' in source
    assert "Backspace deletes" in source


def test_prompt_label_uses_question_sections_instead_of_question_mark():
    source = inspect.getsource(profile_wizard._prompt_fragments) + inspect.getsource(profile_wizard._prompt_label)

    assert "Question" in source
    assert '"? "' not in source
    assert "Instructions" in source
    assert "Options" in source


def test_profile_wizard_defaults_to_gpt55_codex_not_glm():
    spec = AgentProfileSpec(profile_name="x", display_name="X", tagline="", avatar_emoji="⚕", primary_role="utility_bot")

    assert spec.model_provider == "openai-codex"
    assert spec.model == "gpt-5.5"
    import hermes_cli.profile_wizard._data as _data_mod
    assert "glm-5.1" not in inspect.getsource(_data_mod)


def test_action_loop_does_not_create_profile_until_create_action(monkeypatch):
    spec = _spec()
    calls = {"writes": 0}

    monkeypatch.setattr(_wizard_mod, "_single", lambda *args, **kwargs: "exit")

    def fail_write(*args, **kwargs):
        calls["writes"] += 1
        raise AssertionError("profile write should not happen unless create is selected")

    monkeypatch.setattr(_wizard_mod, "_write_profile_files", fail_write)

    result = profile_wizard._action_loop(spec)

    assert result is None
    assert calls["writes"] == 0


def test_skill_bundles_and_docs_skill_catalogue_are_available():
    assert "research" in profile_wizard.SKILL_BUNDLES
    assert "research_assistant" in profile_wizard.SKILL_BUNDLES["research"]["AI & Intelligence"]

    docs_names = {name for opts in profile_wizard.DOCS_SKILL_CATEGORIES.values() for name, _label in opts}
    assert {"hermes-agent", "codex", "github-pr-workflow", "google-workspace", "social-content"}.issubset(docs_names)


def test_guided_onboarding_returns_list_of_specs_for_multi_agent_setup(monkeypatch):
    answers = profile_wizard.OnboardingAnswers(
        user_type="developer",
        working_context="coding",
        user_context="I build apps and want help shipping faster.",
        interests=["software", "ai_automation"],
        use_cases=["coding", "automation"],
        ideas="PR review and debugging",
        setup_style="specialists",
        platforms=["cli"],
        apps=["github", "linear"],
        other_apps="VS Code",
        automation_comfort="safe_local",
        sensitive_areas=["credentials"],
    )
    # We patch _single to simulate choosing "__all__" from the recommendation screen
    def fake_single(title, prompt, values, default=None):
        if title == "Choose starting profile":
            return "__all__"
        return default or values[0][0]
    def fake_multi(*args, **kwargs):
        return kwargs.get("default", [])
    monkeypatch.setattr(_guided_mod, "_score_profile_ideas", lambda a: [profile_wizard.profile_idea_by_id("coding_assistant"), profile_wizard.profile_idea_by_id("research_assistant"), profile_wizard.profile_idea_by_id("admin_assistant_starter")])
    monkeypatch.setattr(_guided_mod, "_single", fake_single)
    monkeypatch.setattr(_guided_mod, "_multi", fake_multi)
    monkeypatch.setattr(_guided_mod, "_text", lambda *args, **kwargs: "test")
    monkeypatch.setattr(_guided_mod, "_select_provider", lambda ap: ap)
    monkeypatch.setattr(_guided_mod, "profile_idea_by_id", profile_wizard.profile_idea_by_id)
    result = profile_wizard.guided_onboarding()
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(s, profile_wizard.AgentProfileSpec) for s in result)
    assert result[0].display_name == "Coding Assistant"
    assert result[1].display_name == "Research Assistant"

def test_onboarding_recommends_profile_and_skill_bundles_from_answers():
    answers = profile_wizard.OnboardingAnswers(
        user_type="developer",
        working_context="coding",
        user_context="I build apps and want help shipping faster.",
        interests=["software", "ai_automation"],
        use_cases=["coding", "automation"],
        ideas="PR review and debugging",
        setup_style="specialists",
        platforms=["cli"],
        apps=["github", "linear"],
        other_apps="VS Code",
        automation_comfort="safe_local",
        sensitive_areas=["credentials"],
    )

    recs = profile_wizard._score_profile_ideas(answers)
    assert recs[0].id in {"coding_lead", "code_worker", "qa_lead"}
    bundles = profile_wizard._recommended_bundle_keys(answers)
    assert "developer" in bundles
    assert "operations" in bundles

    spec = profile_wizard._spec_from_onboarding(answers, recs[0])
    assert "github" in spec.skill_sets.get("Integrations & APIs", [])
    assert "automation" in spec.skill_sets.get("Utility & Tools", [])
    assert "Use cases: coding, automation" in spec.intended_outcomes


def test_local_skill_options_reads_skill_frontmatter(tmp_path, monkeypatch):
    skill_dir = tmp_path / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text('---\nname: demo-skill\ndescription: "Demo skill"\n---\n', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    options = profile_wizard._local_skill_options(limit=5)

    assert ("demo-skill", "demo-skill — Demo skill") in options


def test_browse_category_back_returns_navigation_sentinel(monkeypatch):
    monkeypatch.setattr(_builder_mod, "_single", lambda *args, **kwargs: "back")

    assert profile_wizard.browse_profile_ideas() == profile_wizard.BACK_SENTINEL


def test_guided_first_back_returns_workspace_sentinel(monkeypatch):
    monkeypatch.setattr(_guided_mod, "_single", lambda *args, **kwargs: profile_wizard.BACK_SENTINEL)

    assert profile_wizard.guided_onboarding() == profile_wizard.BACK_SENTINEL


def test_run_profile_wizard_back_from_browse_returns_to_workspace_then_exits(monkeypatch):
    choices = iter(["browse", "exit"])
    calls = {"workspace": 0}

    def fake_single(title, *args, **kwargs):
        if title == "Profile workspace":
            calls["workspace"] += 1
            return next(choices)
        return "exit"

    monkeypatch.setattr(_wizard_mod, "_single", fake_single)
    monkeypatch.setattr(_wizard_mod, "browse_profile_ideas", lambda *args, **kwargs: profile_wizard.BACK_SENTINEL)

    assert profile_wizard.run_profile_wizard() is None
    assert calls["workspace"] == 2


def test_idea_category_labels_are_user_facing_not_jargon():
    labels = dict(profile_wizard.IDEA_CATEGORY_OPTIONS)

    assert labels["ootb"].startswith("Starter profiles")
    assert labels["examples"].startswith("Advanced examples")
    assert labels["leads"].startswith("Multi-agent team roles")
    assert "Profile Distribution" not in " ".join(labels.values())
    assert "Hermes specialist lead" not in " ".join(labels.values())


def test_collect_profile_ctrl_b_rewinds_to_immediate_prior_question(monkeypatch):
    text_calls = []
    text_answers = {
        "Who is this for?": ["User context"],
        "Goals": ["Goals"],
        "Environment": ["Environment"],
        "Display name": ["Demo Agent"],
        "CLI name": ["demo-agent", "demo-agent-2"],
        "Tagline": [profile_wizard.BACK_SENTINEL, "Useful profile"],
        "Avatar emoji": ["⚕"],
        "Skill": [""],
        "Blocked topics": [""],
    }

    def fake_text(title, *args, **kwargs):
        text_calls.append(title)
        answers = text_answers.get(title, [kwargs.get("default", "")])
        value = answers.pop(0) if answers else kwargs.get("default", "")
        text_answers[title] = answers
        return value

    def fake_multi(title, prompt, values, default=None):
        return list(default or [])

    def fake_single(title, prompt, values, default=None):
        if title == "Add more skills?":
            return "done"
        if title == "Add a custom skill?":
            return "no"
        return default or values[0][0]

    monkeypatch.setattr(_builder_mod, "_text", fake_text)
    monkeypatch.setattr(_builder_mod, "_multi", fake_multi)
    monkeypatch.setattr(_builder_mod, "_single", fake_single)
    monkeypatch.setattr(_builder_mod, "_select_provider", lambda active_provider: active_provider)
    monkeypatch.setattr(_scoring_mod, "_single", fake_single)
    monkeypatch.setattr(_scoring_mod, "_multi", fake_multi)

    spec = profile_wizard.collect_profile()

    assert isinstance(spec, profile_wizard.AgentProfileSpec)
    assert spec.profile_name == "demo-agent-2"
    assert text_calls.count("Display name") == 1
    assert text_calls.count("CLI name") == 2
    assert text_calls.count("Tagline") == 2
    assert text_calls.index("Display name") < text_calls.index("CLI name") < text_calls.index("Tagline")


def test_profile_idea_defaults_preserve_docs_and_custom_skills():
    idea = profile_wizard.profile_idea_by_id("klerik")
    spec = profile_wizard.spec_from_profile_idea(idea)

    all_grouped = {skill for skills in spec.skill_sets.values() for skill in skills}
    assert "hermes-agent-skill-authoring" in all_grouped or "hermes-agent-skill-authoring" in spec.custom_skills
    assert "systematic-debugging" in all_grouped or "systematic-debugging" in spec.custom_skills
    assert spec.profile_name
    assert spec.display_name
    assert spec.model_provider
    assert spec.model
    assert spec.target_platforms
    assert spec.boundaries
    assert spec.user_context
    assert spec.intended_outcomes
    assert spec.operating_context


def test_review_and_approve_route_skips_manual_profile_interview(monkeypatch):
    seed = profile_wizard.spec_from_profile_idea(profile_wizard.profile_idea_by_id("general_assistant"))
    choices = iter(["browse", "review", "exit"])
    calls = {"collect": 0, "preview": 0, "action": 0}

    def fake_single(title, *args, **kwargs):
        if title == "Profile workspace":
            return next(choices)
        if title == "Use pre-configured defaults?":
            return next(choices)
        return kwargs.get("default") or "exit"

    def fake_collect(*args, **kwargs):
        calls["collect"] += 1
        raise AssertionError("review route should not run manual collect_profile")

    def fake_preview(spec):
        calls["preview"] += 1
        assert spec.profile_name
        assert spec.skill_sets
        assert spec.target_platforms
        assert spec.boundaries

    def fake_action(spec, **kwargs):
        calls["action"] += 1
        return None

    monkeypatch.setattr(_wizard_mod, "_single", fake_single)
    monkeypatch.setattr(_wizard_mod, "browse_profile_ideas", lambda *args, **kwargs: seed)
    monkeypatch.setattr(_wizard_mod, "collect_profile", fake_collect)
    monkeypatch.setattr(_wizard_mod, "print_preview", fake_preview)
    monkeypatch.setattr(_wizard_mod, "_action_loop", fake_action)

    assert profile_wizard.run_profile_wizard() is None
    assert calls == {"collect": 0, "preview": 1, "action": 1}


def test_customise_preconfigured_route_still_prefills_manual_interview(monkeypatch):
    seed = profile_wizard.spec_from_profile_idea(profile_wizard.profile_idea_by_id("research_assistant"))
    choices = iter(["browse", "customise"])
    seen_seed = {"value": None}

    def fake_single(title, *args, **kwargs):
        if title == "Profile workspace":
            return next(choices)
        if title == "Use pre-configured defaults?":
            return next(choices)
        return kwargs.get("default") or "exit"

    def fake_collect(initial_name=None, seed=None):
        seen_seed["value"] = seed
        return seed

    monkeypatch.setattr(_wizard_mod, "_single", fake_single)
    monkeypatch.setattr(_wizard_mod, "browse_profile_ideas", lambda *args, **kwargs: seed)
    monkeypatch.setattr(_wizard_mod, "collect_profile", fake_collect)
    monkeypatch.setattr(_wizard_mod, "print_preview", lambda spec: None)
    monkeypatch.setattr(_wizard_mod, "_action_loop", lambda spec, **kwargs: None)

    assert profile_wizard.run_profile_wizard() is None
    assert seen_seed["value"] is seed
    assert seen_seed["value"].skill_sets


def test_recommendation_scoring_scenarios_return_real_profile_ids():
    valid_ids = {idea.id for idea in profile_wizard.profile_ideas()}
    scenarios = [
        ("researcher", ["research", "analysis"], ["notion", "obsidian"]),
        ("community", ["community"], ["discord", "telegram"]),
        ("founder", ["automation", "multi_agent"], ["github", "gmail"]),
        ("creator", ["content"], ["x_twitter", "linkedin", "youtube"]),
    ]
    for user_type, use_cases, apps in scenarios:
        answers = profile_wizard.OnboardingAnswers(
            user_type=user_type,
            working_context="mixed",
            user_context="test",
            interests=[],
            use_cases=use_cases,
            ideas="",
            setup_style="recommend",
            platforms=["cli"],
            apps=apps,
            other_apps="",
            automation_comfort="draft_prepare",
            sensitive_areas=[],
        )
        recs = profile_wizard._score_profile_ideas(answers)
        assert recs
        assert all(idea.id in valid_ids for idea in recs)
