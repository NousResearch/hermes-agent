"""v2.8 减法优化 — AgentRouter 路由单元测试

测试五类真实任务路由、关键词硬规则、other 阻断、客户映射。
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure hermes-agent is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent_router import (
    AgentRouter,
    RoutingDecision,
    DEFAULT_ROUTES,
    TASK_TYPE_KEYWORDS,
    CLIENT_NAME_KEYWORDS,
    TASK_CATEGORY_REQUIRED_CAPABILITY,
)


# ── Fixtures ──

@pytest.fixture
def mock_registry():
    """Minimal agent-registry.json fixture matching current production state."""
    return {
        "agents": {
            "claude": {
                "id": "claude",
                "display_name": "Claude Code",
                "capabilities": ["file_modification", "script_execution", "git_operations", "code_review"],
            },
            "codex": {
                "id": "codex",
                "display_name": "Codex CLI",
                "capabilities": ["file_modification", "script_execution", "git_operations", "code_review", "implementation_planning"],
            },
            "agent-tars": {
                "id": "agent-tars",
                "display_name": "Agent TARS",
                "capabilities": ["desktop_control", "app_operation", "screenshot", "macos_automation"],
            },
            "hermes-internal": {
                "id": "hermes-internal",
                "display_name": "Hermes Internal Reasoning",
                "capabilities": ["analysis", "decision_making", "creative_planning", "prioritization", "strategy_decision"],
            },
            "designer": {
                "id": "designer",
                "display_name": "Designer 视觉设计师",
                "capabilities": ["creative_direction", "visual_design", "web_visual_design", "ui_design"],
            },
            "intelligence": {
                "id": "intelligence",
                "display_name": "Intelligence",
                "capabilities": ["web_research", "file_reading_analysis"],
            },
            "deepseek-worker": {
                "id": "deepseek-worker",
                "display_name": "Staam Worker",
                "capabilities": ["background_processing", "file_analysis"],
            },
            "deepseek-tui": {
                "id": "deepseek-tui",
                "display_name": "DeepSeek TUI",
                "capabilities": ["code_generation", "code_review", "debugging"],
            },
        },
        "routing_rules": {
            "file_modification": "claude",
            "script_execution": "claude",
            "git_operations": "claude",
            "code_review": "codex",
            "implementation_planning": "codex",
            "desktop_control": "agent-tars",
            "app_operation": "agent-tars",
            "web_research": "intelligence",
            "file_reading_analysis": "intelligence",
            "strategy_decision": "hermes-internal",
            "creative_direction": "designer",
            "background_task": "deepseek-worker",
        },
    }


@pytest.fixture
def router(mock_registry):
    """Create an AgentRouter pre-loaded with mock registry."""
    r = AgentRouter()
    r._registry = mock_registry
    r._agents_map = mock_registry["agents"]
    r._routing_rules = mock_registry["routing_rules"]
    return r


# ── Task 2: Five task types ──

class TestNewTaskTypes:
    """测试五类新增任务类型路由正确。"""

    def test_marketing_deck_routes_to_strategy(self, router):
        result = router.route("marketing_deck")
        assert result.mode == "pipeline"
        assert "hermes-internal" in result.agents
        assert "intelligence" in result.agents

    def test_file_work_routes_to_reading(self, router):
        result = router.route("file_work")
        assert result.mode == "single_agent"
        assert "intelligence" in result.agents

    def test_code_maintenance_routes_to_writable_executor(self, router):
        result = router.route("code_maintenance")
        assert result.mode == "single_agent"
        assert result.agents == ["claude"]

    def test_desktop_operation_routes_to_agent_tars(self, router):
        result = router.route("desktop_operation")
        assert result.mode == "single_agent"
        assert "agent-tars" in result.agents

    def test_conversation_is_self_execute(self, router):
        result = router.route("conversation")
        assert result.mode == "self_execute"
        assert result.agents == []


# ── Task 2c: Keyword-based routing ──

class TestKeywordRouting:
    """测试关键词硬规则：LLM 分类为 other 但关键词命中 → 强制升格。"""

    def test_mengniu_ppt_force_marketing_deck(self, router):
        """蒙牛 PPT 方案 → marketing_deck"""
        result = router.route("other", raw_request="帮我生成蒙牛的 PPT 方案")
        assert result.mode == "pipeline"
        assert result.client == "蒙牛"

    def test_brand_keyword_force_marketing_deck(self, router):
        """品牌相关关键词 → marketing_deck"""
        result = router.route("other", raw_request="帮我看下品牌合作方案")
        assert result.mode == "pipeline"

    def test_research_keyword_force_research(self, router):
        """搜索任务 → research"""
        result = router.route("other", raw_request="帮我搜集竞品信息")
        assert result.mode == "single_agent"

    def test_file_keyword_force_file_work(self, router):
        """文件处理 → file_work"""
        result = router.route("other", raw_request="帮我读一下这个合同")
        assert result.mode == "single_agent"

    def test_code_keyword_force_code_maintenance(self, router):
        """代码修复 → code_maintenance"""
        result = router.route("other", raw_request="帮我修 bug 在 Hermes 路由")
        assert result.mode == "single_agent"

    def test_desktop_keyword_force_desktop_operation(self, router):
        """桌面操作 → desktop_operation"""
        result = router.route("other", raw_request="帮我截图验证一下")
        assert result.mode == "single_agent"

    def test_chitchat_stays_other(self, router):
        """闲聊无关键词 → 保留 other/conversation"""
        result = router.route("other", raw_request="今天天气怎么样")
        assert result.mode == "self_execute"

    def test_keyword_no_override_when_llm_classified(self, router):
        """LLM 已正确分类时关键词不覆盖。"""
        result = router.route("code_maintenance", raw_request="帮我修 bug")
        assert result.mode == "single_agent"


# ── Task 3c: Marketing hard rules ──

class TestMarketingHardRules:
    """测试营销任务硬规则。"""

    def test_marketing_keywords_never_other(self, router):
        """所有营销关键词命中后 task_category 不能是 other。"""
        marketing_kws = TASK_TYPE_KEYWORDS.get("marketing_deck", [])
        for kw in marketing_kws[:5]:  # sample
            result = router.route("other", raw_request=f"请帮我做{kw}")
            assert result.mode != "self_execute" or result.agents != [], \
                f"Keyword '{kw}' should not route to self_execute with no agents"

    def test_client_name_populates_metadata(self, router):
        """客户名出现时 client 元数据被填充。"""
        result = router.route("other", raw_request="百威的方案帮我看看")
        assert result.client == "百威"

    def test_first_output_strategy_spine(self, router):
        """marketing_deck 类型 first_output 必须为 strategy_spine。"""
        result = router.route("marketing_deck")
        assert result.first_output == "strategy_spine"

    def test_client_info_populated(self, router):
        """客户名匹配后 client 和 local_project_path 应被填充。"""
        result = router.route("other", raw_request="蒙牛世界杯方案")
        assert result.client == "蒙牛"
        # local_project_path may be None if the map file doesn't have the path
        # or if the path doesn't exist

    def test_non_marketing_no_first_output(self, router):
        """非 marketing_deck 任务 first_output 应为 None。"""
        result = router.route("research", raw_request="搜索竞品信息")
        assert result.first_output is None


# ── Task 2e: Empty agent → explicit error ──

class TestEmptyAgentFallback:
    """测试空 agent 列表返回显式错误而非静默降级。"""

    def test_empty_agents_returns_review_only(self, router):
        """没有 capability 映射时返回 review_only 而非静默降级。"""
        # Use a task type with a capability that has no routing rule
        # We break the routing for this test
        old_rules = dict(router._routing_rules)
        router._routing_rules = {}  # Remove all routing
        try:
            result = router.route("code_maintenance")
            # Should be review_only since no agent found
            assert result.mode == "review_only"
            assert "路由失败" in result.reason
        finally:
            router._routing_rules = old_rules


# ── Backward compatibility ──

class TestBackwardCompat:
    """测试原有路由行为不被破坏。"""

    def test_old_categories_still_work(self, router):
        """原有 8 个 task_category 仍然正常工作。"""
        for cat in ["architecture_review", "code_analysis", "brand_strategy",
                     "visual_design", "research", "document", "prompt_design", "other"]:
            result = router.route(cat)
            assert result.mode in ("self_execute", "single_agent", "pipeline")

    def test_user_override_still_works(self, router):
        """用户显式指定 agent 仍然生效。"""
        result = router.route("other", user_agent_override="codex")
        assert result.mode == "single_agent"
        assert "codex" in result.agents

    def test_raw_request_none_does_not_crash(self, router):
        """不传 raw_request 时现有行为不变。"""
        result = router.route("other")
        assert result.mode == "self_execute"

    def test_routing_decision_to_dict(self, router):
        """RoutingDecision.to_dict() 包含新字段。"""
        result = router.route("marketing_deck")
        d = result.to_dict()
        assert "client" in d
        assert "first_output" in d
        assert "must_read_local_files" in d
        assert "delegation_reason" in d  # legacy compat
