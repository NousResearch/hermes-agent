"""BlockRun / ClawRouter provider plugin for hermes-agent.

50+ LLM models via x402 micropayments (USDC on Base or Solana).
No API key needed — your crypto wallet IS your credential.

Provider names: "blockrun", "clawrouter"
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# Aliases that map to this provider (checked by plugins/providers/__init__.py)
ALIASES = ["clawrouter"]


def _export_session_wallet() -> None:
    """If BLOCKRUN_WALLET_KEY is not set but ~/.blockrun/.session exists,
    export it so auth.py auto-detection can find BlockRun as a provider."""
    if os.environ.get("BLOCKRUN_WALLET_KEY"):
        return
    session_file = os.path.expanduser("~/.blockrun/.session")
    try:
        if os.path.exists(session_file):
            key = open(session_file).read().strip()
            if key:
                os.environ["BLOCKRUN_WALLET_KEY"] = key
    except OSError:
        pass


# Export wallet from session file at import time — enables auto-detection
_export_session_wallet()


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
