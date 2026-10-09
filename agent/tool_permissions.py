"""Session-static exact tool grants, independent of mutable schemas and toolsets."""
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolPermissionPolicy:
    allowed_names: frozenset[str] | None = None

    @classmethod
    def from_config(cls, config):
        section = config.get("agent", {}) if isinstance(config, dict) else None
        raw = section.get("allowed_tools") if isinstance(section, dict) else False
        if raw is None:
            return cls()
        if not isinstance(raw, list) or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in raw
        ):
            logger.warning("Invalid agent.allowed_tools: denying all tools; expected null or a list of exact tool names")
            return cls(frozenset())
        return cls(frozenset(raw))

    def denial(self, name: str) -> str | None:
        if self.allowed_names is not None and name not in self.allowed_names:
            return f"Tool '{name}' is denied by agent.allowed_tools for this session."
        return None

    def filter_definitions(self, definitions):
        if self.allowed_names is None:
            return list(definitions or [])
        return [definition for definition in definitions or []
                if (definition.get("function") or {}).get("name") in self.allowed_names]


def load_tool_policy():
    from hermes_cli.config import load_config_readonly
    try:
        return ToolPermissionPolicy.from_config(load_config_readonly())
    except Exception:
        logger.warning("Cannot load tool permissions: denying all tools", exc_info=True)
        return ToolPermissionPolicy(frozenset())


def agent_tool_policy(agent):
    policy = getattr(agent, "_tool_policy", None)
    return policy if isinstance(policy, ToolPermissionPolicy) else ToolPermissionPolicy()


def apply_agent_tool_policy(agent):
    agent.tools = agent._tool_policy.filter_definitions(agent.tools)
    agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools}
