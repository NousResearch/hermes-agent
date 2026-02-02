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
from .agent import AgentConfig, AgentResult, AgentStep, AtroposAgent, SequenceData  # noqa: E402
from .envs import AgentEnv, AgentEnvConfig  # noqa: E402

__all__ = [
    "AtroposAgent",
    "AgentConfig",
    "AgentResult",
    "AgentStep",
    "SequenceData",
    "AgentEnv",
    "AgentEnvConfig",
]

