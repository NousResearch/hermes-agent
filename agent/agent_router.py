"""Agent Router — task-to-agent routing for Hermes 2.8.

Routes user tasks to the appropriate agent based on task_category, required
capabilities, and risk level. Supports pipeline mode and fallback.

Design:
- Default routing maps task_category → execution_mode + target agents
- Override conditions allow the main agent to deviate from defaults
- Fallback ensures degraded but complete delivery when sub-agents fail
- Router does NOT call agents; it provides a plan that the main agent executes
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Default routing rules ──
# Maps task_category → (mode, agents, reason)

DEFAULT_ROUTES: Dict[str, Dict[str, Any]] = {
    "architecture_review": {
        "mode": "self_execute",
        "agents": [],
        "reason": "架构评审类任务的核心价值在主 Agent 的判断力和对 Hermes 全局的把握",
    },
    "code_analysis": {
        "mode": "pipeline",
        "agents": ["kimi"],
        "reason": "代码分析需先由 Kimi 搜集代码/文档，再由主 Agent 分析判断",
    },
    "brand_strategy": {
        "mode": "self_execute",
        "agents": [],
        "reason": "品牌策略类任务需要主 Agent 的策略判断力",
    },
    "visual_design": {
        "mode": "single_agent",
        "agents": ["visual"],
        "reason": "视觉设计任务适合分发给图像/PPT Agent",
    },
    "research": {
        "mode": "single_agent",
        "agents": ["kimi"],
        "reason": "调研类任务适合 K2-thinking 的长上下文搜索能力",
    },
    "document": {
        "mode": "pipeline",
        "agents": ["kimi"],
        "reason": "文档类任务先由 Kimi 收集素材，再主 Agent 撰写整合",
    },
    "prompt_design": {
        "mode": "self_execute",
        "agents": [],
        "reason": "提示词设计需要理解整体系统上下文",
    },
    "other": {
        "mode": "self_execute",
        "agents": [],
        "reason": "未分类任务默认主 Agent 自行处理",
    },
}

# ── Agent capability registry ──
# Maps agent name → capabilities, description

AGENT_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "kimi": {
        "name": "Kimi (K2-thinking)",
        "capabilities": ["search", "long_context_reading", "information_gathering"],
        "use_when": ["需要搜索网页", "需要读长文", "需要整理信息", "需要调研"],
    },
    "claude_code": {
        "name": "Claude Code",
        "capabilities": ["code_execution", "file_modification", "script_writing"],
        "use_when": ["需要改代码", "需要执行脚本", "需要具体文件修改"],
        "constraint": "执行器而非思考器——收到的 prompt 需是明确的修改指令",
    },
    "visual": {
        "name": "图像/PPT Agent",
        "capabilities": ["visual_design", "ppt_generation", "image_creation"],
        "use_when": ["需要视觉创意", "需要 PPT 生成", "需要图像设计"],
    },
}


class RoutingDecision:
    """Result of the routing decision process."""

    def __init__(
        self,
        mode: str,
        agents: List[str],
        reason: str,
        routing_basis: List[str],
        overrides: List[str],
        fallback_plan: Optional[str] = None,
        risk_level: str = "low",
    ):
        self.mode = mode  # self_execute | single_agent | pipeline | review_only
        self.agents = agents
        self.reason = reason
        self.routing_basis = routing_basis  # what drove this decision
        self.overrides = overrides  # any overrides applied over the default
        self.fallback_plan = fallback_plan
        self.risk_level = risk_level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "agents": self.agents,
            "delegation_reason": self.reason,
            "routing_basis": self.routing_basis,
            "overrides": self.overrides or [],
            "fallback_plan": self.fallback_plan,
            "risk_level": self.risk_level,
        }


class AgentRouter:
    """Routes tasks to agents based on task_category, capabilities, and risk."""

    def __init__(self):
        pass

    def route(
        self,
        task_category: str,
        user_agent_override: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        risk_level: str = "low",
    ) -> RoutingDecision:
        """Determine the execution plan for a task.

        Args:
            task_category: From TaskCard.compiled_intent.task_category
            user_agent_override: User explicitly said "use X"
            required_capabilities: Capabilities the task requires
            risk_level: "low" | "medium" | "high"

        Returns:
            RoutingDecision with mode, agents, reason, routing_basis, overrides
        """
        default = DEFAULT_ROUTES.get(task_category, DEFAULT_ROUTES["other"])
        mode = default["mode"]
        agents = list(default["agents"])
        reason = default["reason"]
        routing_basis = ["task_category_default"]
        overrides: List[str] = []

        # Override 1: user explicitly specifies an agent
        if user_agent_override:
            if user_agent_override in AGENT_CAPABILITIES:
                mode = "single_agent"
                agents = [user_agent_override]
                reason = f"用户显式指定使用 {user_agent_override}"
                routing_basis.append("user_override")
                overrides.append("user_override")
            else:
                logger.warning("Unknown agent override: %s", user_agent_override)

        # Override 2: required capabilities not covered by default route
        if required_capabilities:
            default_caps = set()
            for a in agents:
                caps = AGENT_CAPABILITIES.get(a, {}).get("capabilities", [])
                default_caps.update(caps)
            missing = set(required_capabilities) - default_caps
            if missing and mode != "self_execute":
                # Check if another agent covers the missing caps
                for name, info in AGENT_CAPABILITIES.items():
                    if name in agents:
                        continue
                    if set(required_capabilities).issubset(set(info.get("capabilities", []))):
                        agents.append(name)
                        reason += f"; 需要 {missing} 能力，增加 {name}"
                        routing_basis.append("required_capability")
                        overrides.append("capability_expansion")
                        break

        # Override 3: high risk → force pipeline + review gate
        if risk_level == "high" and mode == "self_execute":
            mode = "pipeline"
            agents = ["kimi"]
            reason += "; 高风险任务，强制走 pipeline + Review Gate"
            routing_basis.append("risk_level")
            overrides.append("risk_escalation")

        # Determine fallback plan
        fallback_plan = self._fallback_for(mode, agents)

        return RoutingDecision(
            mode=mode,
            agents=agents,
            reason=reason,
            routing_basis=routing_basis,
            overrides=overrides,
            fallback_plan=fallback_plan,
            risk_level=risk_level,
        )

    # ── Fallback ──

    @staticmethod
    def _fallback_for(mode: str, agents: List[str]) -> str:
        if mode == "self_execute":
            return "主 Agent 自行完成"
        if mode == "single_agent" and agents:
            agent_name = AGENT_CAPABILITIES.get(agents[0], {}).get("name", agents[0])
            return f"{agent_name} 失败 → 主 Agent 接管，标记信息不足"
        if mode == "pipeline":
            return "Pipeline 任一步失败 → 主 Agent 接管，Task Card status=partial 或 blocked"
        return "主 Agent 接管"

    # ── Validation ──

    @staticmethod
    def validate_delegation_prompt(agent_name: str, prompt: str) -> List[str]:
        """Validate a delegation prompt for a target agent.

        Returns list of issues (empty = valid).
        """
        issues = []
        if agent_name == "claude_code":
            # Claude Code must receive specific instructions, not vague requests
            vague_patterns = ["优化一下", "改善性能", "改得好一点", "看着改"]
            for pattern in vague_patterns:
                if pattern in prompt:
                    issues.append(f"Claude Code prompt 包含模糊指令 '{pattern}'，必须是明确修改")
            if len(prompt) < 30:
                issues.append("Claude Code prompt 太短，需要具体的修改描述")
        return issues

    # ── Static helpers ──

    @staticmethod
    def get_available_agents() -> List[str]:
        return list(AGENT_CAPABILITIES.keys())

    @staticmethod
    def get_agent_info(name: str) -> Optional[Dict[str, Any]]:
        return AGENT_CAPABILITIES.get(name)
