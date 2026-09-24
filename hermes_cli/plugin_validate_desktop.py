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
    # A static `import 'https://…'` / `import x from 'file:…'` is the same second stage as the
    # dynamic form above (the renderer would fetch and evaluate it); the loader refuses every
    # URL-scheme specifier too (runtime-loader.ts::unsupportedImports) — this keeps admission and
    # the loader in agreement instead of letting review wave through what the app rejects.
    ("remote import outside the SDK",
     re.compile(r"\bimport\s+(?:[^;'\"]*?\bfrom\s*)?['\"][a-zA-Z][\w+.-]*:")),
    ("script injection",
     re.compile(r"createElement\(\s*['\"]script['\"]\s*\)|<script\b")),
)

_COMMENT = re.compile(r"/\*.*?\*/|(?<![:\w])//[^\n]*", re.S)

# A JS regex literal (``/<script[\s\S]*?<\/script>/gi``) matches markup, it cannot inject any: a
# feed sanitiser that STRIPS script tags is the opposite of the move the rule refuses. Regex
# literals are masked for the markup-shaped rules only; a ``<script`` inside a string literal is
# still the payload of an ``innerHTML`` write and keeps firing. The lookbehind keeps division
# (``a / b / c``) from reading as a literal. The pattern string handed straight to ``new RegExp(``
# is the same sanitiser spelled for a dynamic flag (rss-reader split it into ``"<scr"+"ipt"`` to
# dodge this rule) — but only where the constructor is USED as a matcher: the argument of a string
# method (``html.replace(new RegExp("<script…", flags), '')``) or the receiver of ``.test``/``.exec``.
# Anywhere else (``el.innerHTML = new RegExp("<script src=x></script>").source``) the constructor is
# a string-builder and its literal keeps firing.
_REGEX_LITERAL = re.compile(r"(?<![\w)\]])/(?:[^/\\\n\[]|\\.|\[(?:[^\]\\\n]|\\.)*\])+/[a-z]*")
_JS_STRING = r"(?:\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*')"
_REGEXP_CTOR_MATCHER = re.compile(
    r"\.(?:replace|replaceAll|split|match|matchAll|search)\(\s*new\s+RegExp\(\s*" + _JS_STRING
    + r"|\bnew\s+RegExp\(\s*" + _JS_STRING + r"(?=[^()\n]*\)\s*\.\s*(?:test|exec)\()")
_MARKUP_RULES = frozenset({"script injection"})


def _mask_regex_literals(source: str) -> str:
    masked = _REGEX_LITERAL.sub(lambda m: " " * len(m.group(0)), source)
    return _REGEXP_CTOR_MATCHER.sub(lambda m: " " * len(m.group(0)), masked)


def desktop_surface_findings(source: str) -> List[Tuple[str, int]]:
    """Return ``[(rule, line)]`` for every forbidden construct in a plugin.js source."""
    stripped = _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), source)
    no_regex = _mask_regex_literals(stripped)
    findings: List[Tuple[str, int]] = []
    for rule, pattern in _FORBIDDEN:
        haystack = no_regex if rule in _MARKUP_RULES else stripped
        for match in pattern.finditer(haystack):
            findings.append((rule, haystack.count("\n", 0, match.start()) + 1))
    return sorted(findings, key=lambda f: f[1])


def is_desktop_surface(rel_path: str) -> bool:
    """Whether a file is part of the Desktop surface this lint governs: JS under ``desktop/``.

    The renderer loads ``desktop/plugin.js`` (and what it imports from beside it). A Node sidecar
    (``sidecar/*.mjs``), a build script or a ``tests/*.test.mjs`` never runs in the renderer, so a
    lazy ``import('jszip')`` there is ordinary Node code — running the rules over every ``*.js`` /
    ``*.mjs`` in a repository reports noise, not a surface violation. Batch tooling should scope
    with this predicate (or call ``desktop_surface_hits``) instead of ``rglob``-ing the tree.
    """
    parts = Path(rel_path).parts
    return len(parts) > 1 and parts[0] == "desktop" and Path(rel_path).suffix == ".js"


def desktop_surface_hits(plugin_dir: Path) -> List[str]:
    """``["<rule> (<rel>:<line>)", ...]`` over the plugin's Desktop surface files only."""
    plugin_dir = Path(plugin_dir)
    desktop = plugin_dir / "desktop"
    if not desktop.is_dir():
        return []
    hits: List[str] = []
    for js in sorted(desktop.rglob("*.js")):
        rel = js.relative_to(plugin_dir).as_posix()
        if not is_desktop_surface(rel):
            continue
        try:
            source = js.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits.extend(f"{rule} ({rel}:{line})" for rule, line in desktop_surface_findings(source))
    return hits


def check_desktop_surface(report, plugin_dir: Path) -> None:
    """Fail the report when ``desktop/*.js`` steps outside the SDK surface; silent when there is none."""
    if not (Path(plugin_dir) / "desktop").is_dir():
        return
    hits = desktop_surface_hits(plugin_dir)
    report.add(
        "desktop surface", not hits,
        "; ".join(hits[:8]) + (f" (+{len(hits) - 8} more)" if len(hits) > 8 else "")
        if hits else "stays inside the plugin SDK surface",
    )
