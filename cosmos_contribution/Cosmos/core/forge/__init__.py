"""
FORGE - Cosmos Orchestrated Rapid Generation Engine
========================================================

A swarm-powered development orchestration system that uses multi-model
collective intelligence to plan, execute, verify, and ship code faster
and more reliably than dict single-model approach.

Key Advantages:
- Multi-model deliberation (11 AI agents, not one)
- PROPOSE-CRITIQUE-REFINE-VOTE consensus on every plan
- 7-layer memory system (not fragile file state)
- Provider fallback chains (never blocked by one API)
- Built-in cost tracking and rollback
- Team coordination (not limited to solo dev)

"We don't plan alone. We forge together."
"""

from .forge_engine import ForgeEngine, forge_quick, forge_plan, forge_execute

__all__ = ["ForgeEngine", "forge_quick", "forge_plan", "forge_execute"]
__version__ = "1.0.0"
