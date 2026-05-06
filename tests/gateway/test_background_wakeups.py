"""Tests for the compact background wake-up registry."""

import json
from pathlib import Path

import pytest

import gateway.background_wakeups as background_wakeups

_MODULE_PATH = Path(background_wakeups.__file__).resolve()
_REPO_ROOT = Path(__file__).resolve().parents[2]
assert _MODULE_PATH.is_relative_to(_REPO_ROOT), (
    f"gateway.background_wakeups resolved outside clean worktree: {_MODULE_PATH}"
)

from gateway.background_wakeups import (
    build_background_ephemeral_prompt,
    build_feishu_capability_gap_hint,
    build_feishu_director_hint,
    clear_background_wake_manifest_cache,
    forced_routes_for_command,
    get_background_route_catalog,
    resolve_background_wakeup,
    resolve_feishu_capability_gap,
    resolve_owner_work_dispatch,
    resolve_runtime_receipt_contract,
    resolve_specialist_receipt_binding,
    suggested_commands_for_routes,
)
from hermes_cli.config import save_config


@pytest.fixture(autouse=True)
def wake_skills_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills_dir = tmp_path / "skills" / "productivity"

    def write_skill(name: str, route: str, *, aliases: tuple[str, ...] | None = None, keywords: tuple[str, ...] | None = None):
        target = skills_dir / name
        target.mkdir(parents=True, exist_ok=True)
        alias_values = aliases or (name,)
        alias_text = ", ".join(f'"{value}"' for value in alias_values)
        keyword_block = ""
        if keywords:
            keyword_text = ", ".join(f'"{value}"' for value in keywords)
            keyword_block = f"      keywords: [{keyword_text}]\n"
        (target / "SKILL.md").write_text(
            "---\n"
            f"name: {name}\n"
            "description: demo\n"
            "metadata:\n"
            "  hermes:\n"
            "    wake:\n"
            f"      route: {route}\n"
            f"      aliases: [{alias_text}]\n"
            f"{keyword_block}"
            "---\n"
            "# Demo\n",
            encoding="utf-8",
        )

    write_skill("feishu-cli-first", "doc_feishu")
    write_skill("feishu-cloud-doc-delivery", "doc_feishu")
    write_skill(
        "pdf-production-playbook",
        "doc_pdf",
        aliases=("pdf-playbook",),
        keywords=("html to pdf", "convert to pdf"),
    )
    write_skill(
        "any2pdf",
        "doc_pdf",
        aliases=("local-doc-to-pdf",),
        keywords=("markdown to pdf",),
    )
    write_skill("gog-google-workspace", "doc_google")
    write_skill("hank-ppt-cluster-routing", "ppt")
    write_skill("hank-ppt-storyline", "ppt")
    write_skill("powerpoint", "ppt")
    write_skill("github-auth", "repo")
    write_skill("github-code-review", "repo")
    write_skill(
        "scrapling",
        "difficult_web_extract",
        aliases=("difficult-web-extract", "selector-web-extract"),
        keywords=(
            "web_extract failed",
            "selector extraction",
            "css selector",
            "batch homogeneous pages",
            "light anti-bot fallback",
        ),
    )

    clear_background_wake_manifest_cache()
    yield
    clear_background_wake_manifest_cache()


def test_non_feishu_uses_default_toolsets_only():
    plan = resolve_background_wakeup(
        "research the market and draft a memo",
        platform="telegram",
        default_toolsets=["terminal", "file"],
    )

    assert plan.route_names == ("default",)
    assert set(plan.enabled_toolsets) == {"terminal", "file"}
    assert plan.skill_names == ()


def test_feishu_scan_route_adds_web():
    plan = resolve_background_wakeup(
        "去搜集公开资料和来源清单，不做结论",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    assert "work" in plan.route_names
    assert "scan" in plan.route_names
    assert set(plan.enabled_toolsets) == {
        "clarify",
        "file",
        "memory",
        "session_search",
        "skills",
        "terminal",
        "todo",
        "web",
    }
    assert plan.skill_names == ()


def test_difficult_web_extract_route_is_task_named_not_library_named():
    catalog = get_background_route_catalog("feishu")

    assert "difficult_web_extract" in catalog
    assert "scrapling" not in catalog
    assert catalog["difficult_web_extract"]["display_command"] == "/bg"
    assert "scrapling" in catalog["difficult_web_extract"]["skills"]


def test_difficult_web_extract_routes_selector_batch_fallback_without_browser():
    plan = resolve_background_wakeup(
        "web_extract failed on these 50 announcement pages; use CSS selector .article-title and batch homogeneous pages, browser is too heavy",
        platform="feishu",
        default_toolsets=["feishu_doc"],
    )

    assert "difficult_web_extract" in plan.route_names
    assert "scrapling" in plan.skill_names
    assert "/bg" in plan.wrapper_commands
    assert "web" in plan.enabled_toolsets
    assert "terminal" in plan.enabled_toolsets
    assert "file" in plan.enabled_toolsets
    assert "browser" not in plan.enabled_toolsets
    assert any("difficult_web_extract" in detail for detail in plan.match_details)


def test_difficult_web_extract_does_not_replace_ordinary_web_extract_or_browser():
    ordinary = resolve_background_wakeup(
        "请用 web_extract 总结 https://example.com 这篇普通文章",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )
    browser_task = resolve_background_wakeup(
        "打开网页截图并检查 console，需要登录交互",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    assert "difficult_web_extract" not in ordinary.route_names
    assert "difficult_web_extract" not in browser_task.route_names


def test_feishu_research_multi_agent_upgrades_lane():
    plan = resolve_background_wakeup(
        "并行做一版行业研究，拆给多个 agent 分头处理",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    assert "research" in plan.route_names
    assert "multi_agent" in plan.route_names
    assert "delegation" in plan.enabled_toolsets
    assert "web" in plan.enabled_toolsets
    assert "terminal" in plan.enabled_toolsets


def test_repo_route_adds_local_inspection_toolsets_for_thin_feishu_parent():
    plan = resolve_background_wakeup(
        "/repo 只做只读 smoke test：检查 Hermes repo 当前版本和 routing command 注册状态，不修改文件，返回 receipt。",
        platform="feishu",
        default_toolsets=["feishu_doc", "feishu_drive"],
    )

    assert "repo" in plan.route_names
    assert "/repo" in plan.wrapper_commands
    assert "terminal" in plan.enabled_toolsets
    assert "file" in plan.enabled_toolsets


def test_route_eval_dataset_core_cases():
    fixture_path = Path(__file__).parent / "fixtures" / "route_eval_dataset.jsonl"
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert rows
    for row in rows:
        plan = resolve_background_wakeup(
            row["prompt"],
            platform="feishu",
            default_toolsets=["hermes-feishu-work"],
        )
        meaningful_routes = tuple(
            route for route in plan.route_names if route not in {"default", "work"}
        )
        for route in row["expected_routes"]:
            assert route in meaningful_routes, row["id"]
        for wrapper in row["expected_wrappers"]:
            assert wrapper in plan.wrapper_commands, row["id"]
        if not row["expected_routes"]:
            assert meaningful_routes == (), row["id"]


def test_feishu_route_governance_audit_routes_repo_research_and_orchestration():
    plan = resolve_background_wakeup(
        "请体系化审查和制定route机制提升计划，阅读开源社区先进案例、Hermes本身机制",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    assert "repo" in plan.route_names
    assert "research" in plan.route_names
    assert "multi_agent" in plan.route_names
    assert "/repo" in plan.wrapper_commands
    assert "/research" in plan.wrapper_commands
    assert "/bg" in plan.wrapper_commands

    suggestion = resolve_feishu_capability_gap(
        "请体系化审查和制定route机制提升计划，阅读开源社区先进案例、Hermes本身机制",
        active_toolsets=("terminal", "file", "skills", "session_search", "memory", "todo", "clarify"),
    )
    assert suggestion is not None
    assert "requires_orchestration" in suggestion.work_shape_reasons
    assert plan.execution_plan is not None
    assert plan.execution_plan.dispatch_policy == "parallel"
    assert tuple((unit.route_name, unit.parallelizable, unit.depends_on) for unit in plan.execution_plan.units) == (
        ("research", True, ()),
        ("repo", True, ()),
    )


def test_feishu_multi_work_dispatch_upgrades_to_delegation_lane():
    plan = resolve_background_wakeup(
        "先 research 一下，再整理成飞书文档",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    assert plan.owner_work_plan is not None
    assert tuple((unit.owner, unit.work_class, unit.route_name) for unit in plan.owner_work_plan.work_units) == (
        ("bran", "research", "research"),
        ("claire", "feishu_doc", "doc_feishu"),
    )
    assert plan.execution_plan is not None
    assert plan.execution_plan.dispatch_policy == "serial"
    assert tuple((unit.unit_index, unit.owner, unit.route_name, unit.depends_on) for unit in plan.execution_plan.units) == (
        (1, "bran", "research", ()),
        (2, "claire", "doc_feishu", (1,)),
    )
    assert plan.route_names == ("work", "research", "doc_feishu", "multi_agent")
    assert "delegation" in plan.enabled_toolsets


def test_feishu_doc_and_ppt_skill_bundles_are_small_and_targeted():
    plan = resolve_background_wakeup(
        "基于这些材料写一版飞书云文档，并给一版可编辑 .pptx 的 storyline",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    assert "doc_feishu" in plan.route_names
    assert "ppt" in plan.route_names
    assert plan.skill_names == (
        "feishu-cli-first",
        "feishu-cloud-doc-delivery",
        "hank-ppt-cluster-routing",
        "hank-ppt-storyline",
        "powerpoint",
    )


def test_google_doc_bundle_routes_to_workspace_skill():
    plan = resolve_background_wakeup(
        "起草一版 Google Doc 初稿",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    assert "doc_google" in plan.route_names
    assert plan.skill_names == ("gog-google-workspace",)


def test_pdf_doc_bundle_routes_to_pdf_skills_without_feishu_contamination():
    plan = resolve_background_wakeup(
        "markdown to pdf",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    assert "doc_pdf" in plan.route_names
    assert "doc_feishu" not in plan.route_names
    assert plan.skill_names == ("any2pdf", "pdf-production-playbook")


def test_background_wakeup_tracks_route_families_and_wrapper_commands():
    plan = resolve_background_wakeup(
        "起草一版 Google Doc 初稿",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    assert plan.route_names == ("work", "doc_google")
    assert plan.route_families == ("work", "doc")
    assert plan.wrapper_commands == ("/doc",)



def test_forced_route_command_can_upgrade_plain_prompt():
    plan = resolve_background_wakeup(
        "整理一下这些材料",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
        forced_routes=forced_routes_for_command("doc"),
    )

    assert "doc_feishu" in plan.route_names
    assert "doc_feishu<=forced:doc" in plan.match_details


def test_forced_doc_command_prefers_pdf_lane_for_pdf_prompts():
    plan = resolve_background_wakeup(
        "markdown to pdf",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
        forced_routes=forced_routes_for_command("doc"),
    )

    assert "doc_pdf" in plan.route_names
    assert "doc_feishu" not in plan.route_names
    assert "doc_pdf<=forced:doc" in plan.match_details


def test_metadata_alias_can_trigger_route_match(tmp_path):
    extra_skill = tmp_path / "skills" / "custom" / "doc-canvas-helper"
    extra_skill.mkdir(parents=True, exist_ok=True)
    (extra_skill / "SKILL.md").write_text(
        "---\n"
        "name: doc-canvas-helper\n"
        "description: demo\n"
        "metadata:\n"
        "  hermes:\n"
        "    wake:\n"
        "      route: doc_google\n"
        "      aliases: [canvas-doc]\n"
        "      keywords: [协作文档画布]\n"
        "---\n"
        "# Demo\n",
        encoding="utf-8",
    )
    clear_background_wake_manifest_cache()

    plan = resolve_background_wakeup(
        "帮我起一版 canvas doc 初稿",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )

    clear_background_wake_manifest_cache()
    assert "doc_google" in plan.route_names
    assert "doc-canvas-helper" in plan.skill_names
    assert "doc_google<=metadata_alias:canvas doc" in plan.match_details


def test_metadata_keyword_can_drive_capability_gap_hint(tmp_path):
    extra_skill = tmp_path / "skills" / "custom" / "route-keyword-helper"
    extra_skill.mkdir(parents=True, exist_ok=True)
    (extra_skill / "SKILL.md").write_text(
        "---\n"
        "name: route-keyword-helper\n"
        "description: demo\n"
        "metadata:\n"
        "  hermes:\n"
        "    wake:\n"
        "      route: automation\n"
        "      aliases: [cadence-check]\n"
        "      keywords: [节奏巡检]\n"
        "---\n"
        "# Demo\n",
        encoding="utf-8",
    )
    clear_background_wake_manifest_cache()

    hint = build_feishu_capability_gap_hint("把这个事情做成 cadence check")

    clear_background_wake_manifest_cache()
    assert "/bg" in hint


def test_feishu_routes_merge_with_custom_default_toolsets():
    plan = resolve_background_wakeup(
        "并行搜集资料，再自动化每周跟进",
        platform="feishu",
        default_toolsets=["terminal", "file", "todo", "vision"],
    )

    assert "scan" in plan.route_names
    assert "automation" in plan.route_names
    assert "multi_agent" in plan.route_names
    assert set(plan.enabled_toolsets) == {
        "terminal",
        "file",
        "todo",
        "vision",
        "web",
        "cronjob",
        "delegation",
    }


def test_route_catalog_exposes_background_wrapper_commands_and_manifest_skills():
    catalog = get_background_route_catalog()

    assert catalog["research"]["display_command"] == "/research"
    assert catalog["scan"]["display_command"] == "/research"
    assert catalog["doc_feishu"]["display_command"] == "/doc"
    assert catalog["doc_pdf"]["display_command"] == "/doc"
    assert catalog["multi_agent"]["display_command"] == "/bg"
    assert catalog["repo"]["display_command"] == "/repo"
    assert catalog["automation"]["display_command"] == "/bg"
    assert catalog["ppt"]["display_command"] == "/ppt"
    assert catalog["doc_feishu"]["skills"] == (
        "feishu-cli-first",
        "feishu-cloud-doc-delivery",
    )
    assert catalog["doc_pdf"]["skills"] == (
        "any2pdf",
        "pdf-production-playbook",
    )
    assert catalog["repo"]["skills"] == (
        "github-auth",
        "github-code-review",
    )
    assert catalog["ppt"]["skills"] == (
        "hank-ppt-cluster-routing",
        "hank-ppt-storyline",
        "powerpoint",
    )


def test_route_catalog_picks_up_config_wake_override_without_manual_cache_clear():
    initial = get_background_route_catalog()
    assert initial["ppt"]["skills"] == (
        "hank-ppt-cluster-routing",
        "hank-ppt-storyline",
        "powerpoint",
    )

    save_config(
        {
            "skills": {
                "wake_overrides": {
                    "github-code-review": {
                        "route": "ppt",
                    }
                }
            }
        }
    )

    updated = get_background_route_catalog()
    assert updated["ppt"]["skills"] == (
        "github-code-review",
        "hank-ppt-cluster-routing",
        "hank-ppt-storyline",
        "powerpoint",
    )


def test_resolve_owner_work_dispatch_prefers_explicit_owner_and_work_class():
    plan = resolve_owner_work_dispatch("让 Claire 起个飞书文档初稿")

    assert plan is not None
    assert plan.explicit_owner == "claire"
    assert plan.conflict is None
    assert tuple((unit.owner, unit.work_class, unit.route_name) for unit in plan.work_units) == (
        ("claire", "feishu_doc", "doc_feishu"),
    )


def test_resolve_owner_work_dispatch_defaults_owner_when_no_work_class_found():
    plan = resolve_owner_work_dispatch("让 Bran 先处理一下")

    assert plan is not None
    assert plan.explicit_owner == "bran"
    assert plan.conflict is None
    assert tuple((unit.owner, unit.work_class, unit.route_name) for unit in plan.work_units) == (
        ("bran", "research", "research"),
    )


def test_resolve_owner_work_dispatch_detects_owner_work_conflict():
    plan = resolve_owner_work_dispatch("让 Frank 做 research")

    assert plan is not None
    assert plan.explicit_owner == "frank"
    assert plan.work_units == ()
    assert plan.conflict is not None
    assert plan.conflict.owner == "frank"
    assert plan.conflict.work_class == "research"


def test_resolve_owner_work_dispatch_marks_known_but_unroutable_owner():
    plan = resolve_owner_work_dispatch("让 Sam 先整理知识库")

    assert plan is not None
    assert plan.explicit_owner == "sam"
    assert plan.work_units == ()
    assert plan.conflict is None
    assert plan.unsupported_owner is not None
    assert plan.unsupported_owner.owner == "sam"


def test_resolve_owner_work_dispatch_routes_work_only_requests_to_fixed_owner():
    plan = resolve_owner_work_dispatch("帮我写个文档初稿")

    assert plan is not None
    assert plan.explicit_owner is None
    assert tuple((unit.owner, unit.work_class, unit.route_name) for unit in plan.work_units) == (
        ("claire", "document", "doc_feishu"),
    )


def test_resolve_owner_work_dispatch_routes_html_work_to_ppt_lane():
    plan = resolve_owner_work_dispatch("帮我做个 HTML slides")

    assert plan is not None
    assert plan.explicit_owner is None
    assert tuple((unit.owner, unit.work_class, unit.route_name) for unit in plan.work_units) == (
        ("claire", "html", "ppt"),
    )


def test_resolve_owner_work_dispatch_routes_repo_work_to_frank():
    plan = resolve_owner_work_dispatch("帮我看下这个 PR")

    assert plan is not None
    assert plan.explicit_owner is None
    assert tuple((unit.owner, unit.work_class, unit.route_name) for unit in plan.work_units) == (
        ("frank", "repo", "repo"),
    )


def test_resolve_owner_work_dispatch_preserves_multi_work_order():
    plan = resolve_owner_work_dispatch("先 research 一下，再整理成飞书文档")

    assert plan is not None
    assert tuple((unit.owner, unit.work_class, unit.route_name) for unit in plan.work_units) == (
        ("bran", "research", "research"),
        ("claire", "feishu_doc", "doc_feishu"),
    )


def test_resolve_owner_work_dispatch_supports_multi_owner_sequence():
    plan = resolve_owner_work_dispatch("让 Bran 研究一下，再让 Claire 起个飞书文档")

    assert plan is not None
    assert plan.conflict is None
    assert plan.unsupported_owner is None
    assert tuple((unit.owner, unit.work_class, unit.route_name, unit.explicit_owner) for unit in plan.work_units) == (
        ("bran", "research", "research", True),
        ("claire", "feishu_doc", "doc_feishu", True),
    )


def test_resolve_owner_work_dispatch_supports_multi_owner_repo_and_ppt_sequence():
    plan = resolve_owner_work_dispatch("Frank 看 repo，Claire 整理成 deck")

    assert plan is not None
    assert plan.conflict is None
    assert plan.unsupported_owner is None
    assert tuple((unit.owner, unit.work_class, unit.route_name, unit.explicit_owner) for unit in plan.work_units) == (
        ("frank", "repo", "repo", True),
        ("claire", "ppt", "ppt", True),
    )


def test_suggested_commands_dedupe_route_aliases():
    assert suggested_commands_for_routes(("research", "scan", "doc_feishu", "doc_pdf", "ppt", "repo", "automation", "multi_agent")) == (
        "/research",
        "/doc",
        "/ppt",
        "/repo",
        "/bg",
    )


def test_resolve_specialist_receipt_binding_for_single_alias_family():
    binding = resolve_specialist_receipt_binding(("research", "scan"))

    assert binding is not None
    assert binding.route_role == "research-specialist"
    assert binding.target_agent_id == "bran"
    assert binding.route_names == ("research", "scan")


def test_resolve_specialist_receipt_binding_returns_none_for_mixed_aliases():
    assert resolve_specialist_receipt_binding(("research", "repo")) is None
    assert resolve_specialist_receipt_binding(("multi_agent",)) is None


def test_resolve_runtime_receipt_contract_returns_route_contract_for_mixed_routes():
    contract = resolve_runtime_receipt_contract(("research", "repo"))

    assert contract is not None
    assert contract.binding_kind == "route"
    assert contract.route_role == "mixed-route"
    assert contract.target_agent_id == "route:mixed"
    assert contract.route_names == ("research", "repo")


def test_resolve_runtime_receipt_contract_returns_route_contract_for_unbound_single_route():
    contract = resolve_runtime_receipt_contract(("multi_agent",))

    assert contract is not None
    assert contract.binding_kind == "route"
    assert contract.route_role == "multi_agent-route"
    assert contract.target_agent_id == "route:multi_agent"


def test_resolve_runtime_receipt_contract_uses_concrete_doc_variants_only():
    concrete_feishu = resolve_runtime_receipt_contract(("doc_feishu",))
    concrete_pdf = resolve_runtime_receipt_contract(("doc_pdf",))
    family_only = resolve_runtime_receipt_contract(("doc",))

    assert concrete_feishu is not None
    assert concrete_feishu.binding_kind == "entity"
    assert concrete_feishu.target_agent_id == "claire"
    assert concrete_pdf is not None
    assert concrete_pdf.binding_kind == "entity"
    assert concrete_pdf.target_agent_id == "claire"
    assert family_only is None


def test_background_ephemeral_prompt_mentions_director_role():
    plan = resolve_background_wakeup(
        "并行做一版行业研究",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )
    prompt = build_background_ephemeral_prompt(plan)

    assert "background worker session" in prompt
    assert "director" in prompt.lower()


def test_background_ephemeral_prompt_includes_multi_work_delegation_plan():
    plan = resolve_background_wakeup(
        "先 research 一下，再整理成飞书文档",
        platform="feishu",
        default_toolsets=["hermes-feishu-work"],
    )
    prompt = build_background_ephemeral_prompt(plan)

    assert "Execution plan" in prompt
    assert '"dispatch_policy": "serial"' in prompt
    assert '"unit_index": 1' in prompt
    assert '"depends_on": [1]' in prompt
    assert "delegate_task" in prompt
    assert "receipt_binding.owner" in prompt
    assert "bran" in prompt.lower()
    assert "claire" in prompt.lower()
    assert "preserve this order" in prompt.lower()


def test_feishu_director_hint_mentions_bg_and_review():
    hint = build_feishu_director_hint()

    assert "/bg" in hint
    assert "/research" in hint
    assert "/doc" in hint
    assert "/ppt" in hint
    assert "/repo" in hint
    assert "director" in hint.lower()
    assert "final reviewer" in hint.lower()
    assert "receipt" in hint.lower()


def test_feishu_capability_gap_hint_suggests_research_wrapper():
    hint = build_feishu_capability_gap_hint("请帮我搜集行业资料和公开来源")

    assert "/research" in hint
    assert "web" in hint
    assert "live wrapper map" in hint.lower()
    assert "receipt" in hint.lower()
    assert "bran/claire/frank" in hint.lower()


def test_feishu_capability_gap_hint_surfaces_owner_work_conflict_instead_of_silent_reroute():
    hint = build_feishu_capability_gap_hint("让 Frank 做 research")

    assert "conflicts with the fixed hank dispatch map" in hint.lower()
    assert "do not silently reroute" in hint.lower()
    assert "frank" in hint.lower()
    assert "research" in hint.lower()


def test_feishu_capability_gap_hint_surfaces_known_but_unroutable_owner():
    hint = build_feishu_capability_gap_hint("让 Sam 先整理知识库")

    assert "sam" in hint.lower()
    assert "known governance alias" in hint.lower()
    assert "no live wrapper or route" in hint.lower()
    assert "do not say a worker already picked it up" in hint.lower()


def test_resolve_feishu_capability_gap_exposes_missing_toolsets():
    suggestion = resolve_feishu_capability_gap(
        "请帮我搜集公开资料和来源",
        active_toolsets=("terminal", "file", "skills", "session_search", "memory", "todo", "clarify"),
    )

    assert suggestion is not None
    assert suggestion.route_names == ("scan",)
    assert suggestion.suggested_commands == ("/research",)
    assert suggestion.missing_toolsets == ("web",)
    assert suggestion.skill_names == ()


def test_feishu_capability_gap_stays_empty_when_foreground_already_has_needed_toolset():
    hint = build_feishu_capability_gap_hint(
        "请帮我搜集公开资料和来源",
        active_toolsets=(
            "terminal",
            "file",
            "skills",
            "session_search",
            "memory",
            "todo",
            "clarify",
            "web",
        ),
    )

    assert hint == ""


def test_feishu_capability_gap_can_route_heavy_research_even_without_tool_gap():
    suggestion = resolve_feishu_capability_gap(
        "做一版完整行业研究并给出综合结论",
        active_toolsets=(
            "terminal",
            "file",
            "skills",
            "session_search",
            "memory",
            "todo",
            "clarify",
            "web",
        ),
    )

    assert suggestion is not None
    assert suggestion.route_names == ("research",)
    assert suggestion.missing_toolsets == ()
    assert suggestion.skill_names == ()
    assert "likely_long_running" in suggestion.work_shape_reasons


def test_feishu_capability_gap_hint_mentions_work_shape_routing():
    hint = build_feishu_capability_gap_hint(
        "做一版完整行业研究并给出综合结论",
        active_toolsets=(
            "terminal",
            "file",
            "skills",
            "session_search",
            "memory",
            "todo",
            "clarify",
            "web",
        ),
    )

    assert "/research" in hint
    assert "work-shape routing" in hint.lower()
    assert "likely_long_running" in hint


def test_feishu_capability_gap_marks_orchestration_heavy_multi_work():
    suggestion = resolve_feishu_capability_gap(
        "先 research 一下，再整理成飞书文档",
        active_toolsets=(
            "terminal",
            "file",
            "skills",
            "session_search",
            "memory",
            "todo",
            "clarify",
            "web",
        ),
    )

    assert suggestion is not None
    assert "requires_orchestration" in suggestion.work_shape_reasons


def test_feishu_capability_gap_still_surfaces_route_bound_skills_without_tool_gap():
    suggestion = resolve_feishu_capability_gap(
        "起草一版 Google Doc 初稿",
        active_toolsets=("terminal", "file", "skills", "session_search", "memory", "todo", "clarify"),
    )

    assert suggestion is not None
    assert suggestion.route_names == ("doc_google",)
    assert suggestion.suggested_commands == ("/doc",)
    assert suggestion.missing_toolsets == ()
    assert suggestion.skill_names == ("gog-google-workspace",)


def test_feishu_capability_gap_suggests_ppt_wrapper_for_deck_work():
    suggestion = resolve_feishu_capability_gap(
        "基于这些要点起一版可编辑 deck 初稿",
        active_toolsets=("terminal", "file", "skills", "session_search", "memory", "todo", "clarify"),
    )

    assert suggestion is not None
    assert suggestion.route_names == ("ppt",)
    assert suggestion.suggested_commands == ("/ppt",)
    assert suggestion.missing_toolsets == ()
    assert suggestion.skill_names == (
        "hank-ppt-cluster-routing",
        "hank-ppt-storyline",
        "powerpoint",
    )


def test_feishu_capability_gap_suggests_repo_wrapper_for_github_work():
    suggestion = resolve_feishu_capability_gap(
        "Review this GitHub PR and check whether auth setup is correct",
        active_toolsets=("terminal", "file", "skills", "session_search", "memory", "todo", "clarify"),
    )

    assert suggestion is not None
    assert suggestion.route_names == ("repo",)
    assert suggestion.suggested_commands == ("/repo",)
    assert suggestion.missing_toolsets == ()
    assert suggestion.skill_names == ("github-auth", "github-code-review")


def test_feishu_capability_gap_hint_stays_empty_for_plain_work():
    hint = build_feishu_capability_gap_hint("把这段话润一下")

    assert hint == ""


def test_resolve_feishu_capability_gap_returns_none_for_owner_work_conflict():
    suggestion = resolve_feishu_capability_gap(
        "让 Frank 做 research",
        active_toolsets=("terminal", "file", "skills", "session_search", "memory", "todo", "clarify"),
    )

    assert suggestion is None


def test_resolve_feishu_capability_gap_returns_none_for_known_but_unroutable_owner():
    suggestion = resolve_feishu_capability_gap(
        "让 Sam 先整理知识库",
        active_toolsets=("terminal", "file", "skills", "session_search", "memory", "todo", "clarify"),
    )

    assert suggestion is None
