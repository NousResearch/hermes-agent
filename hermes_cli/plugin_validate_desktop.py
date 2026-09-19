"""Static admission lint for a plugin's Desktop surface (``desktop/plugin.js``).

A ``plugin.js`` is evaluated as ESM in the Electron renderer realm with the app's full authority
(``apps/desktop/src/contrib/runtime-loader.ts`` says so in its header: error isolation only, no
capability boundary). The loader accepts that for files the user put on disk; a catalog install is a
remote source, so listed plugins must stay inside the SDK surface. This lint refuses the moves that
step outside it. It is a tripwire for review, not a sandbox.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

# (rule, regex) applied to comment-stripped source; every hit fails the "desktop surface" check.
_FORBIDDEN: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("prototype patching",
     re.compile(r"\b[A-Za-z_$][\w$]*\.prototype\.[\w$]+\s*=[^=]")),
    ("prototype patching",
     re.compile(r"\bObject\.definePropert(?:y|ies)\(\s*[\w$.]+\.prototype\b")),
    ("prototype patching",
     re.compile(r"\b(?:Reflect|Object)\.setPrototypeOf\(|\.__proto__\s*=")),
    ("dynamic code evaluation",
     re.compile(r"(?<![\w$.])eval\(|\bnew\s+Function\(")),
    ("dynamic import outside the SDK",
     re.compile(r"\bimport\(\s*(?!['\"](?:@hermes/plugin-sdk|react)(?:/[\w/-]*)?['\"]\s*\))")),
    ("script injection",
     re.compile(r"createElement\(\s*['\"]script['\"]\s*\)|<script\b")),
)

_COMMENT = re.compile(r"/\*.*?\*/|(?<![:\w])//[^\n]*", re.S)


def desktop_surface_findings(source: str) -> List[Tuple[str, int]]:
    """Return ``[(rule, line)]`` for every forbidden construct in a plugin.js source."""
    stripped = _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), source)
    findings: List[Tuple[str, int]] = []
    for rule, pattern in _FORBIDDEN:
        for match in pattern.finditer(stripped):
            findings.append((rule, stripped.count("\n", 0, match.start()) + 1))
    return sorted(findings, key=lambda f: f[1])


def check_desktop_surface(report, plugin_dir: Path) -> None:
    """Fail the report when ``desktop/*.js`` steps outside the SDK surface; silent when there is none."""
    desktop = Path(plugin_dir) / "desktop"
    if not desktop.is_dir():
        return
    hits: List[str] = []
    for js in sorted(desktop.rglob("*.js")):
        try:
            source = js.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = js.relative_to(plugin_dir).as_posix()
        hits.extend(f"{rule} ({rel}:{line})" for rule, line in desktop_surface_findings(source))
    report.add(
        "desktop surface", not hits,
        "; ".join(hits[:8]) + (f" (+{len(hits) - 8} more)" if len(hits) > 8 else "")
        if hits else "stays inside the plugin SDK surface",
    )
