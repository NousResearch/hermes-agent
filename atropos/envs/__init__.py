"""
Environment implementations for atropos-agent.

NOTE: AgentEnv is the OLD environment system, replaced by
environments/hermes_base_env.py (HermesAgentBaseEnv).
Import is lazy to avoid pulling in deleted dependencies.
"""


def __getattr__(name):
    """Lazy import to avoid breaking when old dependencies are removed."""
    if name in ("AgentEnv", "AgentEnvConfig"):
        from .agent_env import AgentEnv, AgentEnvConfig
        return {"AgentEnv": AgentEnv, "AgentEnvConfig": AgentEnvConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentEnv", "AgentEnvConfig"]
