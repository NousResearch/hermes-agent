"""Prototype MCP permission bridge controlled by Hermes OS."""

from .errors import PERMISSION_DENIED, adapter_error


SUPPORTED_MCP_CATEGORIES = {
    "github",
    "discord",
    "browser",
    "filesystem",
    "documentation",
    "research",
    "kalshi",
    "broker",
}


def grant_tool(permission_policy, requested_tool):
    allowed = set(permission_policy.get("allowed_tools", []))
    denied = set(permission_policy.get("denied_tools", []))
    if requested_tool in denied or requested_tool not in allowed:
        return None, adapter_error(PERMISSION_DENIED, "Tool access denied: " + requested_tool)
    return {
        "tool": requested_tool,
        "granted": True,
        "permission_authority": "Hermes OS",
    }, None
