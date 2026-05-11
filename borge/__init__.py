"""
Borge Agent — Cognitively-grounded AI agent framework.

Standalone usage:
    from borge import BorgeRunner
    runner = BorgeRunner()
    runner.run("help me debug this")

Plugin usage (wrapping Hermes / OpenClaw / any backend):
    from borge.agent import BorgeAgent
    cognitive = BorgeAgent(agent_backend=my_agent)

Named after Jorge Luis Borges — explorer of infinite memory and knowledge.
"""

__version__ = "0.1.0"

from .agent import BorgeAgent
from .runner import BorgeRunner

__all__ = ["BorgeAgent", "BorgeRunner", "__version__"]
