"""Ares collaboration feature flags.

All flags default OFF. Each flag gates a specific phase's behavior.
Flags are read from environment variables only (no config.yaml, no DB).
"""
from __future__ import annotations

import os


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0")
    return raw == "1"


# Phase 1: Role/Mission contracts
COLLAB_CONTRACTS_V1 = lambda: _flag("ARES_COLLAB_CONTRACTS_V1")

# Phase 2: Typed findings, context compiler
TYPED_FINDINGS_V1 = lambda: _flag("ARES_TYPED_FINDINGS_V1")
CONTEXT_COMPILER_V1 = lambda: _flag("ARES_CONTEXT_COMPILER_V1")

# Phase 3: strict effect argument validation. Production permit canary
# admission is owned by ares.permit_daemon configuration in collaboration.py.
STRICT_EFFECT_TOOL_ARGS_V1 = lambda: _flag("ARES_STRICT_EFFECT_TOOL_ARGS_V1")

# Phase 4: Witness, closure projection
WITNESS_V1 = lambda: _flag("ARES_WITNESS_V1")
CLOSURE_PROJECTION_V1 = lambda: _flag("ARES_CLOSURE_PROJECTION_V1")

# Phase 5: Evaluation/replay
REPLAY_EVALUATION_V1 = lambda: _flag("ARES_REPLAY_EVALUATION_V1")
