"""BlockRun / ClawRouter provider plugin for hermes-agent.

50+ LLM models via x402 micropayments (USDC on Base or Solana).
No API key needed — your crypto wallet IS your credential.

Provider names: "blockrun", "clawrouter"
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Aliases that map to this provider (checked by plugins/providers/__init__.py)
ALIASES = ["clawrouter"]


def resolve(
    *,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve BlockRun runtime provider config.

    Called by runtime_provider.py when provider="blockrun" or "clawrouter".
    Returns the standard hermes runtime dict.
    """
    from .provider import resolve_blockrun_provider

    chain = None
    if model_cfg:
        chain = str(model_cfg.get("chain") or "").strip().lower() or None

    return resolve_blockrun_provider(
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
        chain=chain,
    )


def register_tools() -> None:
    """Register BlockRun tools (wallet, image, prediction markets).

    Importing tools.py triggers registry.register() calls at module level.
    """
    from . import tools  # noqa: F401 — side-effect: registers tools


# Auto-register tools on plugin load so they appear in the toolset menu
try:
    register_tools()
except Exception:
    pass  # blockrun-llm not installed — tools just won't appear
