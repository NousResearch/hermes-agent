"""RecursiveIntell stack auto-loader for Hermes.

Imported at agent startup to register RI PyO3 components as available
transports, compressors, and vector backends.

Add to Hermes config.yaml::

    plugins:
      auto_load:
        - recursiveintell_stack

Or import manually in a startup hook::

    import agent.transports.ri_autoload  # noqa: F401 — side-effect import
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_registered = False


def _register_all() -> None:
    """One-shot registration of all RI components."""
    global _registered
    if _registered:
        return
    _registered = True

    # ── llm-pipeline ──────────────────────────────────────────────
    try:
        from llm_pipeline._native import LlmConfig, Pipeline  # noqa: F401

        logger.info("ri-autoload: llm-pipeline native extension available")
    except ImportError:
        logger.debug("ri-autoload: llm-pipeline not installed")

    # ── context-governor ──────────────────────────────────────────
    try:
        from context_governor._native import compact  # noqa: F401

        logger.info("ri-autoload: context-governor native extension available")
    except ImportError:
        logger.debug("ri-autoload: context-governor not installed")

    # ── agent-graph ───────────────────────────────────────────────
    try:
        from agent_graph._native import AgentState  # noqa: F401

        logger.info("ri-autoload: agent-graph native extension available")
    except ImportError:
        logger.debug("ri-autoload: agent-graph not installed")

    # ── poly-kv ───────────────────────────────────────────────────
    try:
        from poly_kv._native import validate_shape_json  # noqa: F401

        logger.info("ri-autoload: poly-kv native extension available")
    except ImportError:
        logger.debug("ri-autoload: poly-kv not installed")


# Auto-register on import
_register_all()
