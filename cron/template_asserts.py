"""WARN-only unified report template assertions (TKT-0033 Phase A).

Before a unified report leaves the delivery pipeline, check it for required
markers and return any violations as warning strings. The caller logs them
as WARNINGS — delivery is NEVER blocked on these checks.

Required markers:
  * a verdict line (the word "verdict", case-insensitive)
  * section structure (a line starting with ``##`` or containing ``━``)
  * at least one health icon (✅ ⚠️ ❌ 🟢 🟠 🔴)

Pure stdlib, deterministic, ~zero latency: simple substring/regex checks
over the outgoing text.
"""

from __future__ import annotations

import re

_HEALTH_ICONS = ("✅", "⚠️", "❌", "🟢", "🟠", "🔴")
_VERDICT_RE = re.compile(r"verdict", re.IGNORECASE)
_SECTION_RE = re.compile(r"^##", re.MULTILINE)
_RULE_CHAR = "━"


def check_report_markers(body: str) -> list[str]:
    """Return a list of template warnings for *body*; empty list = pass."""
    warnings: list[str] = []
    if not _VERDICT_RE.search(body):
        warnings.append("report template: missing verdict line")
    if not any(icon in body for icon in _HEALTH_ICONS):
        warnings.append("report template: missing health icon")
    if not (_SECTION_RE.search(body) or _RULE_CHAR in body):
        warnings.append("report template: missing section structure")
    return warnings
