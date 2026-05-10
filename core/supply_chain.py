"""
core/supply_chain.py — Supply Chain Security Scan for Skills and MCP.

Scans skill content and MCP tool descriptions for prompt injection
before they reach the LLM context. Uses ``core/sanitize.py`` as the
underlying engine, with trust scoring by source provenance.

Usage:
    from core.supply_chain import scan_skill_content, scan_mcp_description

    # Skills
    result = scan_skill_content(content, source_type="github")
    if result.blocked:
        content = "[SKILL CONTENT BLOCKED]"
    elif result.text != content:
        content = result.text  # sanitized

    # MCP
    desc = scan_mcp_description(description, server_type="npx")
    if desc.blocked:
        description = "[MCP DESCRIPTION BLOCKED]"
    elif desc.text != description:
        description = desc.text  # sanitized
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module-level availability flag ─────────────────────────────────────────
# Set once at import time. If true, sanitize pipeline is loaded and scans run.
# If false, all scans silently pass through with a startup warning.
_SANITIZE_AVAILABLE = False
try:
    from core.sanitize import sanitize_input as _sanitize_input

    _SANITIZE_AVAILABLE = True
except ImportError:
    logger.warning(
        "core.sanitize not available — supply chain scan DISABLED. "
        "All skill content and MCP descriptions will pass unscanned."
    )
except Exception as _exc:
    logger.warning(
        "core.sanitize import error (%s) — supply chain scan DISABLED.", _exc
    )

# ── Source trust scores ────────────────────────────────────────────────────
# These define the base reputation of content based on where it came from.
# Passed as channel_reputation to sanitize_input so injection penalties
# apply on top (e.g., local=0.9 - 0.50 critical = 0.40, still above block).
# Unlike channel-based reputations (telegram=0.6, api=0.5) which assume
# user-initiated traffic, these model content-provenance trust.

SOURCE_TRUST: dict[str, float] = {
    # Skills sources
    "skill_local": 0.9,    # Local filesystem skill
    "skill_github": 0.4,   # Downloaded from GitHub
    "skill_openclaw": 0.2, # OpenClaw marketplace
    "skill_remote": 0.3,   # Other remote source
    # MCP server sources
    "mcp_local": 0.9,      # Local command (binary or script)
    "mcp_npx": 0.4,        # npx package
    "mcp_uvx": 0.4,        # uvx package
    "mcp_remote": 0.3,     # Remote HTTP/SSE server
}

# Block threshold — trust score below this = blocked
_BLOCK_THRESHOLD = 0.3


@dataclass
class SupplyChainResult:
    """Result of a supply chain content scan."""

    text: str
    blocked: bool = False
    trust_score: float = 1.0
    redacted_patterns: list = field(default_factory=list)
    source: str = ""


def _get_source_reputation(source_type: str) -> float:
    """Get base reputation for a given source type.

    Unknown source types default to block threshold (conservative).

    Args:
        source_type: One of the SOURCE_TRUST keys (e.g. 'skill_github', 'mcp_npx').

    Returns:
        Float reputation score.
    """
    return SOURCE_TRUST.get(source_type, _BLOCK_THRESHOLD)


def _sanitize_with_source(
    text: str,
    channel: str,
    source_type: str,
    enable_semantic: bool = True,
) -> tuple[str, float, list[str], bool]:
    """Run sanitize_input with source-based channel_reputation.

    Falls back to pass-through if core.sanitize is unavailable.

    Returns:
        (sanitized_text, trust_score, redacted_patterns, blocked)
    """
    if not _SANITIZE_AVAILABLE:
        return text, 1.0, [], False

    try:
        source_rep = _get_source_reputation(source_type)
        context = {"channel_reputation": source_rep}

        san = _sanitize_input(
            text,
            channel=channel,
            is_data=True,
            enable_semantic=enable_semantic,
            enable_data_fence=True,  # Wrap in DATA fence so LLM sees it as data, not instructions
            context=context,
        )
        return san.text, san.trust_score, san.redacted_patterns, san.blocked

    except ImportError:
        # Should not happen since we check _SANITIZE_AVAILABLE, but be safe
        logger.debug("core.sanitize import lost after startup — scan skipped")
        return text, 1.0, [], False

    except Exception as exc:
        logger.exception("Supply chain scan error: %s", exc)
        return text, 1.0, [], False


def scan_skill_content(
    content: str,
    source_type: str = "skill_local",
    channel: str = "skill",
) -> SupplyChainResult:
    """Scan skill content for prompt injection.

    Args:
        content: Raw SKILL.md content.
        source_type: Provenance key (e.g. 'skill_local', 'skill_github').
        channel: Sanitize channel name.

    Returns:
        SupplyChainResult with sanitized text and scan verdict.
    """
    if not isinstance(content, str) or not content.strip():
        return SupplyChainResult(text=content, source=source_type)

    sanitized, trust, redacted, blocked = _sanitize_with_source(
        content, channel, source_type,
    )

    if blocked or trust < _BLOCK_THRESHOLD:
        logger.info(
            "SKILL SCAN BLOCKED (trust=%.2f, source=%s, patterns=%s)",
            trust, source_type, redacted,
        )
        return SupplyChainResult(
            text=sanitized, blocked=True, trust_score=trust,
            redacted_patterns=redacted, source=source_type,
        )

    return SupplyChainResult(
        text=sanitized, blocked=False, trust_score=trust,
        redacted_patterns=redacted, source=source_type,
    )


def scan_mcp_description(
    description: str,
    server_type: str = "mcp_local",
) -> SupplyChainResult:
    """Scan an MCP tool description for prompt injection.

    Args:
        description: MCP tool description string.
        server_type: Provenance key (e.g. 'mcp_local', 'mcp_npx').

    Returns:
        SupplyChainResult with sanitized description and scan verdict.
    """
    if not isinstance(description, str) or not description.strip():
        return SupplyChainResult(text=description, source=server_type)

    sanitized, trust, redacted, blocked = _sanitize_with_source(
        description, "mcp", server_type,
    )

    if blocked or trust < _BLOCK_THRESHOLD:
        logger.info(
            "MCP DESCRIPTION BLOCKED (trust=%.2f, source=%s, patterns=%s)",
            trust, server_type, redacted,
        )
        return SupplyChainResult(
            text=sanitized, blocked=True, trust_score=trust,
            redacted_patterns=redacted, source=server_type,
        )

    return SupplyChainResult(
        text=sanitized, blocked=False, trust_score=trust,
        redacted_patterns=redacted, source=server_type,
    )


def infer_skill_source(skill_dir) -> str:
    """Infer skill source type from skill directory path.

    Args:
        skill_dir: Path object or string for the skill directory.

    Returns:
        Source type string for SOURCE_TRUST lookup.
    """
    if skill_dir is None:
        return "skill_remote"

    from pathlib import Path

    p = Path(str(skill_dir)).resolve()
    if p.is_relative_to(Path.home()):
        return "skill_local"

    return "skill_local"


def infer_mcp_server_type(config: dict) -> str:
    """Infer MCP server type from its config.

    Args:
        config: MCP server config dict. Typically has 'command' or 'url' key.

    Returns:
        Server type string for SOURCE_TRUST lookup.
    """
    if not isinstance(config, dict):
        return "mcp_local"

    command = config.get("command", "")
    url = config.get("url", "")

    if url:
        return "mcp_remote"

    if isinstance(command, str):
        cmd_str = command.lower()
        if cmd_str.startswith("npx"):
            return "mcp_npx"
        if cmd_str.startswith("uvx"):
            return "mcp_uvx"

    return "mcp_local"
