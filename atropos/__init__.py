"""
Atropos integration for Hermes-Agent.

This package is intentionally optional: Hermes-Agent should work without Atropos.
If you import anything from `atropos.*` without having `atroposlib` installed,
we raise a clear error with install instructions.

Install (recommended, from repo checkout):
  uv sync --extra atropos

Or (pip / editable):
  pip install -e '.[atropos]'
"""

from __future__ import annotations


def _require_atroposlib() -> None:
    try:
        import atroposlib  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Hermes-Agent Atropos integration requires `atroposlib`, but it is not installed.\n"
            "Install it with:\n"
            "  uv sync --extra atropos\n"
            "or:\n"
            "  pip install -e '.[atropos]'\n"
        ) from exc


_require_atroposlib()

# Re-export the most commonly used pieces for convenience.
# Agent imports are eager (always available).
from .agent import AgentConfig, AgentResult, AgentStep, AtroposAgent, SequenceData  # noqa: E402

# Env imports are lazy to avoid pulling in deleted atropos.tools dependencies.
# Use: from atropos.envs import AgentEnv, AgentEnvConfig  (if needed)

__all__ = [
    "AtroposAgent",
    "AgentConfig",
    "AgentResult",
    "AgentStep",
    "SequenceData",
]

