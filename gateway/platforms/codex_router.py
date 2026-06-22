"""
Codex smart router: determines whether a Feishu message should be forwarded to Codex CLI.

Trigger conditions:
1. Explicit prefix: starts with "codex" or "/codex" (case-insensitive)
2. Keyword matching: contains specific action phrases
"""

_CODEX_KEYWORDS = [
    "fix bug",
    "fix the bug",
    "帮我写",
    "加个功能",
    "帮我加",
    "单测",
    "异常",
    "帮我改",
    "帮我修",
    "重构",
    "写个脚本",
    "写个测试",
    "改这个bug",
    "修这个bug",
]


def should_route_to_codex(text: str) -> bool:
    """Return True if the message should be routed to Codex CLI."""
    if not text:
        return False
    stripped = text.strip().lower()
    # Explicit prefix
    if stripped.startswith("codex") or stripped.startswith("/codex"):
        return True
    # Keyword matching
    for kw in _CODEX_KEYWORDS:
        if kw in stripped:
            return True
    return False