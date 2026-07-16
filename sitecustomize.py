"""Early Beta-mode bootstrap for source and editable installations.

Python imports ``sitecustomize`` during interpreter startup when the project
root is on ``sys.path``. This lets BETA-001 select the identity without
changing Hermes' default mode or its runtime loop.
"""

from __future__ import annotations


def _apply_beta_mode() -> None:
    try:
        from agent import prompt_builder
        from agent.beta_identity import identity_for_mode

        prompt_builder.DEFAULT_AGENT_IDENTITY = identity_for_mode(
            prompt_builder.DEFAULT_AGENT_IDENTITY
        )
    except Exception:
        # Startup customization must never prevent Hermes from launching.
        return


_apply_beta_mode()
