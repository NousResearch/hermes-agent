"""hf-inspector plugin — Hugging Face Model & GGUF Quant Explorer for Hermes Agent.

Provides zero-dependency inspection tools for Hugging Face model repositories:
  - hf_inspect_model: fetch architecture, parameter count, context length, license, and stats.
  - hf_list_quants: discover GGUF, AWQ, and GPTQ quantized files with sizes and direct links.
"""

from __future__ import annotations

import logging
from typing import Any

from .tools import (
    HF_INSPECT_MODEL_SCHEMA,
    HF_LIST_QUANTS_SCHEMA,
    handle_hf_inspect_model,
    handle_hf_list_quants,
)

logger = logging.getLogger(__name__)

_TOOLS = (
    ("hf_inspect_model", HF_INSPECT_MODEL_SCHEMA, handle_hf_inspect_model, "🤗"),
    ("hf_list_quants",   HF_LIST_QUANTS_SCHEMA,   handle_hf_list_quants,   "📦"),
)


def register(ctx: Any) -> None:
    """Register tools with Hermes Agent plugin context."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="hf_inspector",
            schema=schema,
            handler=handler,
            emoji=emoji,
        )
