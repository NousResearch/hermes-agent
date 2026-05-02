"""Review Templates — static quality check definitions for Hermes 2.8.

Hardcoded per-category review checklists. No Memory dependency, no database reads.
Templates are keyed by task_category; "universal" applies to ALL tasks.

Each check has:
  - id: stable identifier (used in Review Gate result)
  - question: the review question the reviewer must answer
  - type: "rule" (deterministic, auto-checkable) or "llm" (semantic, needs model)
"""

REVIEW_TEMPLATES = {
    "brand_strategy": {
        "name": "品牌方案审查",
        "checks": [
            {
                "id": "brand_core",
                "question": '是否有母品牌心智（“要强=蒙牛”）并转译到具体场景？',
                "type": "llm",
            },
            {
                "id": "brand_not_sales",
                "question": "是否避免把品牌方案写成销售方案？",
                "type": "llm",
            },
            {
                "id": "brand_attribution",
                "question": "是否有品牌归因机制——能证明品牌动作带来了什么？",
                "type": "llm",
            },
            {
                "id": "why_mengniu",
                "question": '是否能回答"为什么必须由蒙牛做"？',
                "type": "llm",
            },
            {
                "id": "competitor_moat",
                "question": '是否能回答"竞品为什么复制不了"？',
                "type": "llm",
            },
        ],
    },
    "architecture_review": {
        "name": "架构设计审查",
        "checks": [
            {
                "id": "serves_secretary_goal",
                "question": "是否服务'主 Agent 秘书化'目标？",
                "type": "llm",
            },
            {
                "id": "infra_vs_experience",
                "question": "是否区分基础设施升级和体验升级？",
                "type": "llm",
            },
            {
                "id": "clear_priority",
                "question": "是否有明确的优先级排序？",
                "type": "llm",
            },
            {
                "id": "no_overengineering",
                "question": "是否避免过度工程化？",
                "type": "llm",
            },
            {
                "id": "implementable_steps",
                "question": "是否能被分步实现？",
                "type": "llm",
            },
        ],
    },
    "universal": {
        "name": "通用质量审查",
        "checks": [
            {
                "id": "has_core_judgment",
                "question": "是否有核心判断，而非空泛建议？",
                "type": "llm",
            },
            {
                "id": "has_actionable_changes",
                "question": "是否有可执行改法，而非罗列选项？",
                "type": "llm",
            },
            {
                "id": "matches_gu_style",
                "question": "是否符合 Gu 的表达偏好（直接、不废话）？",
                "type": "llm",
            },
            {
                "id": "next_step_ready",
                "question": "是否能直接进入下一步工作？",
                "type": "llm",
            },
            {
                "id": "matches_user_preferences",
                "question": "是否符合用户（Gu）的已知偏好？",
                "type": "llm",
            },
            {
                "id": "matches_project_context",
                "question": "是否符合当前项目（Hermes）的上下文和规则？",
                "type": "llm",
            },
            {
                "id": "has_result_summary",
                "question": "是否有 result_summary？",
                "type": "rule",
            },
            {
                "id": "has_task_card",
                "question": "是否有 Task Card？",
                "type": "rule",
            },
            {
                "id": "has_compiled_intent",
                "question": "是否有 compiled_intent？",
                "type": "rule",
            },
            {
                "id": "success_criteria_addressed",
                "question": "是否回应了所有 success_criteria？",
                "type": "rule",
            },
            {
                "id": "agent_result_accepted",
                "question": "子 Agent 的结果是否被主 Agent 审查并采纳/修改？",
                "type": "rule",
            },
        ],
    },
}


def get_checks_for_category(task_category: str) -> list:
    """Return combined check list: category-specific + universal."""
    category_checks = REVIEW_TEMPLATES.get(task_category, {}).get("checks", [])
    universal_checks = REVIEW_TEMPLATES.get("universal", {}).get("checks", [])
    return category_checks + universal_checks


def get_template_name(task_category: str) -> str:
    tmpl = REVIEW_TEMPLATES.get(task_category, {})
    return tmpl.get("name", "通用审查")
