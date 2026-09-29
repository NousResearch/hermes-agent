"""Agent Sandbox terminal backend."""

from .provider import AgentSandboxProvider


def register(ctx) -> None:
    ctx.register_terminal_environment_provider(AgentSandboxProvider(ctx))
