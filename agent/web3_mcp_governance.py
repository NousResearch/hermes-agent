"""Optional policy layer for Web3-oriented MCP tools (no duplicate MCP servers).

When ``web3_mcp_governance.enabled`` is true, Hermes can block high-risk tool
names, return dry-run stubs instead of executing, persist shallow chain hints
under ``HERMES_HOME/web3_mcp/wallet_context.json`` (for prompts / Honcho sync),
and add bounded retries on slow cross-chain calls.

This module intentionally does not import Honcho — mirror ``wallet_context.json``
into a peer card via ``honcho_profile`` if you want SDK-backed persistence.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)

_CTX_NAME = "wallet_context.json"
_CTX_SUBDIR = "web3_mcp"


@dataclass(frozen=True)
class Web3GovernanceOutcome:
    """Result of evaluating whether an MCP tool call may proceed."""

    mode: str  # allow | block | simulate
    message: str = ""
    simulated_payload: Optional[Dict[str, Any]] = None


def _defaults() -> Dict[str, Any]:
    return {
        "enabled": False,
        # When non-empty, governance applies only to these MCP server ids.
        "server_names": [],
        "blocked_tool_substrings": [
            "send_raw",
            "submit_transaction",
            "sign_typed_data",
            "eth_send_transaction",
        ],
        # When non-empty, tool name must match at least one substring (after blocks).
        "allowed_tool_substrings": [],
        # If True, blocked writes return a simulated JSON payload instead of error.
        "simulate_on_block": False,
        "mcp_call_extra_retries": 0,
        "retry_backoff_seconds": 1.5,
        "persist_chain_hints": True,
        # Include wallet_context.json summary in the cached system prompt (new session).
        "inject_wallet_context_in_prompt": False,
    }


def _effective_policy() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        raw = load_config().get("web3_mcp_governance")
        if isinstance(raw, dict):
            cfg = raw
    except Exception as exc:
        logger.debug("web3_mcp_governance: load_config failed: %s", exc)
    merged = _defaults()
    merged.update(cfg)
    return merged


def _scoped_server(policy: Mapping[str, Any], server_name: str) -> bool:
    if not policy.get("enabled"):
        return False
    names = policy.get("server_names") or []
    if not names:
        logger.debug(
            "web3_mcp_governance.enabled is true but server_names is empty; "
            "skipping governance (set server_names explicitly)."
        )
        return False
    return server_name in names


def _norm_tool(name: str) -> str:
    return name.lower()


def evaluate_mcp_tool_call(
    server_name: str,
    tool_name: str,
    args: Optional[Mapping[str, Any]],
) -> Web3GovernanceOutcome:
    """Apply block / allowlist / simulate policy before an MCP tool runs."""
    policy = _effective_policy()
    if not _scoped_server(policy, server_name):
        return Web3GovernanceOutcome(mode="allow")

    tn = _norm_tool(tool_name)
    blocked: List[str] = list(policy.get("blocked_tool_substrings") or [])
    for sub in blocked:
        if sub and sub.lower() in tn:
            msg = (
                f"MCP tool '{tool_name}' is blocked by web3_mcp_governance "
                f"(matched '{sub}'). Use read-only or estimation tools, or adjust config."
            )
            if policy.get("simulate_on_block"):
                body = {
                    "simulated": True,
                    "governance": "blocked_stub",
                    "note": msg,
                    "suggested_alternatives": [
                        "read-only contract calls",
                        "gas estimation",
                        "route simulation",
                    ],
                }
                return Web3GovernanceOutcome(
                    mode="simulate",
                    message=msg,
                    simulated_payload=body,
                )
            return Web3GovernanceOutcome(mode="block", message=msg)

    allow = list(policy.get("allowed_tool_substrings") or [])
    if allow:
        if not any(a and a.lower() in tn for a in allow):
            msg = (
                f"MCP tool '{tool_name}' is not on web3_mcp_governance "
                "allowed_tool_substrings for this server."
            )
            return Web3GovernanceOutcome(mode="block", message=msg)

    return Web3GovernanceOutcome(mode="allow")


def extra_retries_for_mcp_call(server_name: str, tool_name: str) -> int:
    """Extra MCP call attempts (beyond the first) for scoped servers."""
    policy = _effective_policy()
    if not _scoped_server(policy, server_name):
        return 0
    try:
        n = int(policy.get("mcp_call_extra_retries", 0))
    except (TypeError, ValueError):
        n = 0
    return max(0, min(n, 5))


def retry_backoff_seconds() -> float:
    policy = _effective_policy()
    try:
        return float(policy.get("retry_backoff_seconds", 1.5))
    except (TypeError, ValueError):
        return 1.5


def _wallet_context_file() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _CTX_SUBDIR / _CTX_NAME


def maybe_persist_chain_hints(
    server_name: str,
    tool_name: str,
    args: Optional[Mapping[str, Any]],
) -> None:
    """Best-effort: record chain ids / addresses seen in tool args."""
    policy = _effective_policy()
    if not policy.get("persist_chain_hints"):
        return
    if not _scoped_server(policy, server_name):
        return
    if not args:
        return

    chain_keys = (
        "chainId",
        "chain_id",
        "chain",
        "network",
        "fromChain",
        "toChain",
    )
    addr_keys = ("from", "to", "owner", "wallet", "address")

    snapshot: Dict[str, Any] = {}
    for key in chain_keys:
        if key in args and args[key] is not None:
            snapshot[key] = args[key]
    addresses: Dict[str, Any] = {}
    for key in addr_keys:
        val = args.get(key)
        if isinstance(val, str) and re.match(r"^0x[a-fA-F0-9]{40}$", val):
            addresses[key] = val[:10] + "…"

    if not snapshot and not addresses:
        return

    path = _wallet_context_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: MutableMapping[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}

    hist = data.setdefault("recent_calls", [])
    entry = {
        "server": server_name,
        "tool": tool_name,
        "chains": snapshot,
        "address_hints": addresses,
    }
    hist.append(entry)
    data["recent_calls"] = hist[-50:]

    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("web3_mcp_governance: could not persist context: %s", exc)


def load_wallet_context_snapshot() -> Dict[str, Any]:
    """Return parsed wallet_context.json or {} (for tests / prompt injection)."""
    path = _wallet_context_file()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def wallet_context_prompt_block() -> str:
    """Format persisted wallet context for the system prompt (session-stable).

    Injected during :meth:`AIAgent._build_system_prompt` only — not refreshed
    mid-conversation — so provider prompt-cache prefixes stay stable (see AGENTS.md).
    """
    policy = _effective_policy()
    if not policy.get("inject_wallet_context_in_prompt"):
        return ""
    snap = load_wallet_context_snapshot()
    recent = snap.get("recent_calls") or []
    if not isinstance(recent, list) or not recent:
        return ""
    tail = recent[-5:]
    lines = [
        "### Web3 MCP wallet context (session-stable snapshot)",
        "Recent governed MCP calls (from `web3_mcp/wallet_context.json`). "
        "Prefer these chains and servers when routing; do not broadcast "
        "transactions unless the user explicitly asked.",
        "",
    ]
    for i, entry in enumerate(tail, 1):
        if not isinstance(entry, dict):
            continue
        srv = entry.get("server", "?")
        tool = entry.get("tool", "?")
        chains = entry.get("chains") or {}
        addr = entry.get("address_hints") or {}
        lines.append(
            f"{i}. `{srv}` / `{tool}` — chains: {chains!r} — address_hints: {addr!r}"
        )
    lines.append("")
    lines.append(
        "(Refreshes when a new agent session builds the system prompt, not each API turn.)"
    )
    return "\n".join(lines)
