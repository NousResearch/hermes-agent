"""
Shared Utilities Package.
"""

from typing import Any, Dict


def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitizes dict keys for privacy compliance."""
    sanitized = {}
    sensitive = {"token", "secret", "password", "key", "auth"}
    for k, v in payload.items():
        if any(s in k.lower() for s in sensitive):
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = v
    return sanitized


__all__ = ["sanitize_payload"]
