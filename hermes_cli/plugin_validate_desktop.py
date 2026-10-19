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
# ``X.prototype.y =`` is checked separately (``_foreign_prototype_writes``): it passes only where ``X``
# is provably the file's own class/function, because a bundled library setting fields on its own
# classes (three.js: ``Vector2.prototype.isVector2 = true``) is not reaching into the renderer.
_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
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


_PROTOTYPE_WRITE = re.compile(r"\b([A-Za-z_$][\w$]*)\.prototype\.[\w$]+\s*=[^=]")
_RENDERER_GLOBAL_NAMES = (
    r"Object|Function|Array|String|Number|Boolean|Symbol|Promise|RegExp|Date|Error|JSON|Map|Set"
    r"|WeakMap|WeakSet|Proxy|Reflect|Storage|Window|Document|Node|EventTarget|Event|Location|History"
    r"|Navigator|WebSocket|XMLHttpRequest|Request|Response|Headers|URL|Blob|File|FileReader"
    r"|MutationObserver|Worker|BroadcastChannel|MessagePort|IDB\w*|\w*Element|\w*Event|CSS\w*")
# A regex cannot see scope, so a same-named local declaration elsewhere in the file must not excuse
# a write to a renderer global: these receivers always fail, declared or not.
_RENDERER_GLOBAL = re.compile(rf"^(?:{_RENDERER_GLOBAL_NAMES})$")
# A regex cannot see scope, and minifiers reuse one-letter names (three.js minified is
# ``class s{constructor(){s.prototype.isVector2=!0}}`` beside a dozen unrelated ``s=`` locals), so
# file-wide name checks cannot decide. The write passes only where ``X`` provably IS the class:
# inside the body of ``class X``/``function X``, or after such a declaration statement in the same
# block, with nothing rebinding ``X`` in between. ``window["Storage"]``, ``(0, window.Storage)`` and every
# other aliasing spelling then fail by default, wherever an unrelated ``class X`` lives.
_OWN_DECLARATION = r"(?:\bclass\s+{0}\b[^{{]*|\bfunction\s*\*?\s*{0}\s*\([^)]*\)\s*)\{{"
# ``var X = class {...}`` / ``X = function () {...}``: what bundlers emit; binds ``X`` in its block.
_ASSIGNED_DECLARATION = (r"(?<![\w$.]){0}\s*=\s*(?:class\b[^{{]*|function\s*\*?\s*[\w$]*\s*\([^)]*\)\s*)\{{")
# What can make ``X`` name something else, read liberally (a false hit only refuses a bundle for
# review; a miss would excuse a renderer patch): any assignment operator on ``X``; a declaration;
# ``X`` inside a destructuring pattern that is assigned or iterated (``({X} = o)``, ``for ([X] of
# a)``); a parameter (function, method, arrow, catch); ``for (X of/in …)``; an import binding.
# A reference (``register(X)``, ``[X, Y]``, ``{ Array: X }``) is not a binding.
_REBINDING = (
    r"(?<![\w$.]){0}\s*(?:&&|\|\||\?\?|\*\*|<<|>>>?|[-+*/%&|^])?=(?![=>])"
    r"|\b(?:let|const|var|class|function\s*\*?)\s+{0}(?![\w$])"
    # a PATTERN opens where an expression cannot (not ``obj[`` / ``f()[``), names ``X`` itself as a
    # target (not ``X.name``, not the key in ``{X: other}``), and is assigned or iterated
    r"|(?:^|(?<=[;(,{{}}=?:&|!\n])|\b(?:let|const|var|return|yield|void))\s*[\[{{][^;=()]*?(?<![\w$.]){0}(?![\w$.]|\s*:)[^;=()]*?[}}\]][\s}}\]]*"
    r"(?:=(?![=>])|\b(?:of|in)\b)"
    r"|(?:\bfunction\s*\*?\s*[\w$]*|(?<![\w$.])(?!(?:if|for|while|switch|with|return|typeof|await)\b)[\w$]+)"
    r"\s*\([^()]*?(?<![\w$.]){0}(?![\w$])[^()]*\)\s*\{{"
    r"|\(([^()]*?(?<![\w$.]){0}(?![\w$])[^()]*)\)\s*=>|(?<![\w$.]){0}\s*=>"
    r"|\bcatch\s*\([^)]*(?<![\w$.]){0}(?![\w$])"
    r"|(?<![\w$.]){0}\s+(?:of|in)\b|\bas\s+{0}(?![\w$])|\bimport\s+{0}(?![\w$])")


_REGEX_AFTER_WORD = re.compile(r"(?:^|[^\w$.])(?:return|typeof|case|do|else|in|of|void|delete|throw|new|yield|await)\s*$")
_MAX_READINGS = 32


def _slash_readings(source: str, at: int) -> tuple[bool, bool]:
    """(may be a regex, may be division) for the ``/`` at *at* in code. After ``)`` or ``}`` both
    are legal JavaScript (``if (x) /re/.test(s)`` vs ``(a) / b``); only a parser could tell."""
    if source.startswith("/", at + 1):
        return False, True
    j = at - 1
    while j >= 0 and source[j] in " \t\r\n":
        j -= 1
    if j < 0:
        return True, False
    prev = source[j]
    if prev in ")}":
        return True, True
    if prev in "+-" and j > 0 and source[j - 1] == prev:  # ``a++ / b``
        return False, True
    if prev in "(,=:[!&|?{;+-*%<>~^":
        return True, False
    return bool(_REGEX_AFTER_WORD.search(source, max(0, j - 12), j + 1)), True


def _step(source: str, i: int, state: tuple) -> tuple[bool, list[tuple]]:
    """Advance one reading of the source by the character at *i*: (is that character literal text,
    next readings). An empty list means this reading is not valid JavaScript (it dies)."""
    kind, holes = state
    c = source[i]
    if kind in ("\"", "'"):
        if c == "\\":
            return True, [(kind + "\\", holes)]
        if c == "\n":
            return True, []
        return c != kind, [("c" if c == kind else kind, holes)]
    if kind in ("\"\\", "'\\", "t\\", "r\\", "k\\"):
        return True, [(kind[0], holes)]
    if kind == "t":
        if c == "\\":
            return True, [("t\\", holes)]
        if c == "`":
            return False, [("c", holes)]
        if c == "$" and source.startswith("${", i):
            return True, [("$", holes + (0,))]
        return True, [("t", holes)]
    if kind == "$":  # the ``{`` of ``${``
        return True, [("c", holes)]
    if kind in ("r", "k"):
        if c == "\n":
            return True, []
        if c == "\\":
            return True, [(kind + "\\", holes)]
        if kind == "r" and c == "/":
            return False, [("c", holes)]
        nxt = {"[": "k"} if kind == "r" else {"]": "r"}
        return True, [(nxt.get(c, kind), holes)]
    # code
    if c in "\"'":
        return False, [(c, holes)]
    if c == "`":
        return False, [("t", holes)]
    if c == "/":
        regex, division = _slash_readings(source, i)
        return False, ([("r", holes)] if regex else []) + ([("c", holes)] if division else [])
    if holes and c == "{":
        return False, [("c", holes[:-1] + (holes[-1] + 1,))]
    if holes and c == "}":
        if holes[-1] == 0:
            return True, [("t", holes[:-1])]
        return False, [("c", holes[:-1] + (holes[-1] - 1,))]
    return False, [("c", holes)]


def _code_view(source: str) -> str:
    """*source* with the text of quoted strings, template literals and regex literals blanked, same
    length; a template's ``${…}`` stays code. Where a ``/`` could be either a regex or division,
    BOTH readings are followed, and a character is blanked only when every live reading agrees it
    is literal text. Since the true reading is always among them, executable code is never blanked
    (a wrong guess cannot hide a patch); ambiguity only costs a refusal for manual review."""
    out = list(source)
    readings: set[tuple] = {("c", ())}
    for i in range(len(source)):
        literal_everywhere = True
        nxt: set[tuple] = set()
        for state in readings:
            literal, follow = _step(source, i, state)
            literal_everywhere &= literal
            nxt.update(follow)
        if literal_everywhere and source[i] != "\n":
            out[i] = " "
        if not nxt or len(nxt) > _MAX_READINGS:  # beyond what we can follow: blank nothing more
            break
        readings = nxt
    return "".join(out)


def _brace_pairs(code: str) -> dict[int, int]:
    """``{`` offset -> matching ``}`` offset in *code* (a :func:`_code_view`)."""
    pairs: dict[int, int] = {}
    stack: list[int] = []
    for i, ch in enumerate(code):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            pairs[stack.pop()] = i
    return pairs


def _enclosing_block_end(pairs: dict[int, int], at: int, size: int) -> int:
    """End of the innermost ``{...}`` containing *at* (``size`` at top level)."""
    ends = [close for open_, close in pairs.items() if open_ < at < close]
    return min(ends) if ends else size


def _own_class_write(source: str, name: str, at: int, pairs: dict[int, int]) -> bool:
    """The write is inside ``X``'s own body (any declaration form), or follows a ``class X`` /
    ``function X`` STATEMENT in the same block; either way with no rebinding of ``X`` in between."""
    escaped = re.escape(name)
    rebinding = re.compile(_REBINDING.format(escaped))
    named = re.compile(_OWN_DECLARATION.format(escaped))
    assigned = re.compile(_ASSIGNED_DECLARATION.format(escaped))
    decls = sorted([(d, False) for d in named.finditer(source, 0, at)] + [(d, True) for d in assigned.finditer(source, 0, at)],
                   key=lambda item: item[0].start())
    for decl, is_assignment in decls:
        body = decl.end() - 1
        close = pairs.get(body, -1)
        if close > at:
            if not rebinding.search(source, body + 1, at):
                return True
            continue
        # A class EXPRESSION's own name is visible only in its body; a statement or ``X = class``
        # binds ``X`` for the rest of its block.
        statement = is_assignment or not re.search(
            r"[=(,:?&|]\s*$|\breturn\s*$", source[max(0, decl.start() - 20):decl.start()])
        if (statement and close != -1 and _enclosing_block_end(pairs, decl.start(), len(source)) > at
                and not rebinding.search(source, close + 1, at)):
            return True
    return False


def _foreign_prototype_writes(source: str) -> list[int]:
    """Offsets of ``X.prototype.y =`` that are not provably on ``X``'s own class (see above)."""
    code = _code_view(source)  # offsets unchanged; string/template text never reads as code
    writes = [(m.start(), m.group(1)) for m in _PROTOTYPE_WRITE.finditer(code)]
    pairs = _brace_pairs(code) if writes else {}
    return [at for at, name in writes
            if _RENDERER_GLOBAL.match(name) or not _own_class_write(code, name, at, pairs)]


def desktop_surface_findings(source: str) -> list[tuple[str, int]]:
    """Return ``[(rule, line)]`` for every forbidden construct in a plugin.js source."""
    stripped = _strip_comments(source)
    no_regex = _mask_regex_literals(stripped)
    findings = [("prototype patching", stripped.count("\n", 0, at) + 1)
                for at in _foreign_prototype_writes(stripped)]
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
