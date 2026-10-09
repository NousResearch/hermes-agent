"""Static admission lint for a plugin's Desktop surface (``plugin.js`` / ``desktop/plugin.js``).

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
_FORBIDDEN: tuple[tuple[str, "re.Pattern[str]"], ...] = (
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
    # The app's own markup (`data-slot` / `data-tour` / `data-sidebar` / `data-testid`) is not plugin
    # surface: a plugin that queries it to restyle, hide, click or rewrite core UI collides with every
    # other plugin doing the same and breaks on any Desktop release. Use a slot, route or SDK hook.
    ("app DOM reach",
     re.compile(r"\bdocument\.(?:querySelector(?:All)?|getElementsBy\w+|getElementById)\(\s*[`'\"][^`'\"]*"
                r"\[data-(?:slot|tour|sidebar|testid)\b")),
    ("app DOM reach",
     re.compile(r"\.observe\(\s*document\.body\s*,\s*\{[^}]*\b(?:childList|subtree)\b")),
)

# String literals are matched first and kept, so a ``/*`` or ``//`` INSIDE a string (a glob such as
# ``'**/*.md'``, a URL) cannot open a "comment" that blanks every line up to the next ``*/`` and
# hides whatever forbidden construct sits there.
_COMMENT = re.compile(
    r"(\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)"
    r"|/\*.*?\*/|(?<![:\w])//[^\n]*",
    re.DOTALL,
)


def _strip_comments(source: str) -> str:
    return _COMMENT.sub(lambda m: m.group(1) or "\n" * m.group(0).count("\n"), source)

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


def desktop_surface_findings(source: str) -> list[tuple[str, int]]:
    """Return ``[(rule, line)]`` for every forbidden construct in a plugin.js source."""
    stripped = _strip_comments(source)
    no_regex = _mask_regex_literals(stripped)
    findings: list[tuple[str, int]] = []
    for rule, pattern in _FORBIDDEN:
        haystack = no_regex if rule in _MARKUP_RULES else stripped
        for match in pattern.finditer(haystack):
            findings.append((rule, haystack.count("\n", 0, match.start()) + 1))
    return sorted(findings, key=lambda f: f[1])


# The Desktop installer's entry rules (apps/desktop/electron/desktop-plugin-install.ts
# ``findDesktopEntry``): a repo-root ``plugin.js`` wins and the whole repo root is published as the
# plugin folder; otherwise ``desktop/plugin.js`` and its ``desktop/`` tree. Admission must lint every
# layout the installer accepts, or a root-layout plugin ships to the renderer unlinted.
_ROOT_ENTRY = "plugin.js"
_DESKTOP_DIR = "desktop"


def _has_root_entry(plugin_dir: Path) -> bool:
    return (plugin_dir / _ROOT_ENTRY).exists()


def is_desktop_surface(rel_path: str, root_entry: bool = False) -> bool:
    """Whether a file is part of the Desktop surface this lint governs.

    That is the repo-root ``plugin.js`` (an installer entry on its own), JS under ``desktop/``, and,
    when the root entry exists (*root_entry*), the root-level JS published beside it. A Node sidecar
    (``sidecar/*.mjs``), a build script or a ``tests/*.test.mjs`` never runs in the renderer, so a
    lazy ``import('jszip')`` there is ordinary Node code — running the rules over every ``*.js`` /
    ``*.mjs`` in a repository reports noise, not a surface violation. Batch tooling should scope
    with this predicate (or call ``desktop_surface_hits``) instead of ``rglob``-ing the tree.
    """
    path = Path(rel_path)
    if path.suffix != ".js":
        return False
    parts = path.parts
    if len(parts) == 1:
        return parts[0] == _ROOT_ENTRY or root_entry
    return parts[0] == _DESKTOP_DIR


def _desktop_surface_files(plugin_dir: Path) -> list[Path]:
    root_entry = _has_root_entry(plugin_dir)
    candidates = list(plugin_dir.glob("*.js"))
    desktop = plugin_dir / _DESKTOP_DIR
    if desktop.is_dir():
        candidates.extend(desktop.rglob("*.js"))
    return sorted(
        f for f in candidates
        if f.is_file() and is_desktop_surface(f.relative_to(plugin_dir).as_posix(), root_entry)
    )


def desktop_surface_hits(plugin_dir: Path) -> list[str]:
    """``["<rule> (<rel>:<line>)", ...]`` over the plugin's Desktop surface files only."""
    plugin_dir = Path(plugin_dir)
    hits: list[str] = []
    for js in _desktop_surface_files(plugin_dir):
        rel = js.relative_to(plugin_dir).as_posix()
        try:
            source = js.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        hits.extend(f"{rule} ({rel}:{line})" for rule, line in desktop_surface_findings(source))
    return hits


def check_desktop_surface(report, plugin_dir: Path) -> None:
    """Fail the report when the Desktop surface steps outside the SDK; silent when there is none."""
    plugin_dir = Path(plugin_dir)
    if not ((plugin_dir / _DESKTOP_DIR).is_dir() or _has_root_entry(plugin_dir)):
        return
    hits = desktop_surface_hits(plugin_dir)
    report.add(
        "desktop surface", not hits,
        "; ".join(hits[:8]) + (f" (+{len(hits) - 8} more)" if len(hits) > 8 else "")
        if hits else "stays inside the plugin SDK surface",
    )
