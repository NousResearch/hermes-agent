#!/usr/bin/env python3
"""Portable static page checker (mw-page-check).

Given a RENDERED HTML file + per-project YAML config, verify delivery rules
offline on any laptop/CI (no browser, no network).

  exit 0  — zero blocking failures (manual items do not block)
  exit 1  — one or more blocking check failures
  exit 2  — usage / config error

stdlib-only core · optional PyYAML · self-contained mini-YAML fallback ·
Python 3.9+

Task: MW-P3-I2b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urljoin, urlparse

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERR = 2

DEFAULT_CONFIG_REL = Path(".work") / "page-check.yaml"

KNOWN_CHECKS = frozenset(
    {
        "soft_404",
        "links_internal",
        "language",
        "file_size",
        "pagination",
        "video_attrs",
        "related_no_self",
        "sticky_cover",
    }
)

# Core delivery checks: must be present, enabled, and blocking (fail closed).
# Only sticky_cover may be manual / optional.
CORE_CHECKS = frozenset(
    {
        "soft_404",
        "links_internal",
        "language",
        "file_size",
        "video_attrs",
        "pagination",
        "related_no_self",
    }
)

# Visible-text extraction skips these (and their descendants).
_SKIP_TEXT_TAGS = frozenset({"script", "style", "noscript", "template"})
# Structural chrome excluded when soft_404 falls back to <body>.
_CHROME_TAGS = frozenset({"nav", "header", "footer", "aside"})
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

# File-like extensions for offline link existence checks.
_FILE_EXT_RE = re.compile(
    r"\.(html?|xhtml|php|asp|aspx|jsp|pdf|txt|xml|json|csv|"
    r"jpe?g|png|gif|webp|svg|ico|bmp|avif|"
    r"mp4|webm|ogg|mp3|wav|m4a|"
    r"css|js|mjs|map|woff2?|ttf|otf|eot)$",
    re.I,
)

_FORCE_MINI_YAML = False

try:
    import yaml as _yaml  # type: ignore
except ImportError:  # pragma: no cover
    _yaml = None


# ---------------------------------------------------------------------------
# minimal YAML subset loader (self-contained copy; do not import menu_gate)
# ---------------------------------------------------------------------------


class MiniYamlError(ValueError):
    """Raised when the mini YAML loader hits an ambiguous construct."""


def _split_flow_items(inner: str) -> List[str]:
    """Split comma-separated items inside ``[]`` or ``{}``, respecting quotes."""
    parts: List[str] = []
    buf = ""
    in_q: Optional[str] = None
    depth = 0
    for ch in inner:
        if in_q:
            buf += ch
            if ch == in_q:
                in_q = None
            continue
        if ch in "\"'":
            in_q = ch
            buf += ch
            continue
        if ch in "[{":
            depth += 1
            buf += ch
            continue
        if ch in "]}":
            depth = max(0, depth - 1)
            buf += ch
            continue
        if ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
            continue
        buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _scalar(v: str) -> Any:
    """Parse a YAML scalar: quoted string, bool, null, int/float, flow list/map, or raw str."""
    v = v.strip()
    if not v:
        return None
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    if " #" in v:
        v = v.split(" #", 1)[0].strip()
    # flow mapping: {blocking: true, manual: true}
    if v.startswith("{") and v.endswith("}"):
        inner = v[1:-1].strip()
        if not inner:
            return {}
        out: Dict[str, Any] = {}
        for part in _split_flow_items(inner):
            if ":" not in part:
                raise MiniYamlError(f"unrepresentable flow mapping entry: {part!r}")
            k, _, raw = part.partition(":")
            out[k.strip()] = _scalar(raw)
        return out
    # flow list: [a, b, "c"]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_scalar(p) for p in _split_flow_items(inner)]
    low = v.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _mini_yaml(text: str) -> Any:
    """Minimal YAML-subset parser for page-check config.

    Handles nested mappings by 2-space indent, lists of mappings, quoted/
    unquoted scalars, booleans, null, inline ``[a, b]`` lists, ``#`` comments.
    Fail closed on unrepresentable constructs (MiniYamlError).
    """
    lines = [
        l
        for l in (raw.rstrip() for raw in text.splitlines())
        if l.strip() and not l.strip().startswith("#")
    ]
    pos = 0

    def indent_of(l: str) -> int:
        return len(l) - len(l.lstrip(" "))

    def _is_list_item(s: str) -> bool:
        return s == "-" or s.startswith("- ")

    def _list_item_body(s: str) -> str:
        if s == "-":
            return ""
        return s[2:].strip()

    def _parse_mapping_continuation(d: Dict[str, Any], base_indent: int) -> None:
        nonlocal pos
        while pos < len(lines):
            nline = lines[pos]
            ni = indent_of(nline)
            if ni <= base_indent:
                break
            ns = nline.strip()
            if _is_list_item(ns):
                break
            if ":" not in ns:
                raise MiniYamlError(
                    f"unrepresentable YAML at line content {ns!r}: "
                    "expected key: value under list item"
                )
            nk, _, nv = ns.partition(":")
            pos += 1
            if nv.strip():
                d[nk.strip()] = _scalar(nv)
            elif pos < len(lines) and indent_of(lines[pos]) > ni:
                d[nk.strip()] = parse_block(indent_of(lines[pos]))
            else:
                d[nk.strip()] = None

    def parse_block(indent: int) -> Any:
        nonlocal pos
        result: Any = None
        while pos < len(lines):
            line = lines[pos]
            cur = indent_of(line)
            if cur < indent:
                break
            if cur > indent:
                raise MiniYamlError(
                    f"orphan deeper indent without a parent key near: {line.strip()!r}"
                )
            s = line.strip()
            if _is_list_item(s):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise MiniYamlError(f"mixed mapping/list structure near: {s!r}")
                item = _list_item_body(s)
                pos += 1
                if item == "":
                    if pos < len(lines) and indent_of(lines[pos]) > indent:
                        nested_indent = indent_of(lines[pos])
                        nested_s = lines[pos].strip()
                        if not _is_list_item(nested_s) and ":" in nested_s:
                            d: Dict[str, Any] = {}
                            _parse_mapping_continuation(d, indent)
                            result.append(d)
                        else:
                            result.append(parse_block(nested_indent))
                    else:
                        result.append(None)
                elif ":" in item and item[0] not in "\"'":
                    k, _, v = item.partition(":")
                    d = {k.strip(): _scalar(v) if v.strip() else None}
                    _parse_mapping_continuation(d, indent)
                    if d and list(d.values()) == [None]:
                        only_k = next(iter(d))
                        if pos < len(lines) and indent_of(lines[pos]) > indent:
                            d[only_k] = parse_block(indent_of(lines[pos]))
                    result.append(d)
                else:
                    result.append(_scalar(item))
            else:
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise MiniYamlError(f"mixed list/mapping structure near: {s!r}")
                if ":" not in s:
                    raise MiniYamlError(
                        f"unrepresentable YAML near: {s!r} (expected key: value)"
                    )
                k, _, v = s.partition(":")
                pos += 1
                if v.strip():
                    result[k.strip()] = _scalar(v)
                elif pos < len(lines) and indent_of(lines[pos]) > cur:
                    result[k.strip()] = parse_block(indent_of(lines[pos]))
                else:
                    result[k.strip()] = None
        return result

    return parse_block(0) if lines else {}


def load_yaml_text(text: str, force_mini: bool = False) -> Any:
    """Load YAML text: prefer PyYAML unless forced to mini loader."""
    use_mini = force_mini or _FORCE_MINI_YAML or _yaml is None
    if not use_mini:
        data = _yaml.safe_load(text)
        return data if data is not None else {}
    return _mini_yaml(text) or {}


def load_yaml_file(path: Path, force_mini: bool = False) -> Any:
    text = path.read_text(encoding="utf-8")
    return load_yaml_text(text, force_mini=force_mini)


# ---------------------------------------------------------------------------
# path helpers (asset-root containment)
# ---------------------------------------------------------------------------


def _realpath(p: Path) -> Path:
    return Path(os.path.realpath(str(p)))


def under_root(root: Path, rel: str) -> Path:
    """Join a path under root. Absolute-looking paths still join under root."""
    if rel is None:
        return root
    s = str(rel).replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    if s.startswith("/"):
        s = s.lstrip("/")
    return root / s


def path_escapes_root(root: Path, candidate: Path) -> bool:
    root_r = _realpath(root)
    cand_r = _realpath(candidate)
    try:
        cand_r.relative_to(root_r)
        return False
    except ValueError:
        return True


def contained_path(root: Path, rel: str) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve rel under root; fail closed if real path escapes asset-root."""
    if rel is None or str(rel).strip() == "":
        return None, "empty path"
    joined = under_root(root, rel)
    norm = str(rel).replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    if ".." in Path(norm).parts:
        return None, f"path escapes asset-root: {rel}"
    # For non-existent files, realpath still resolves parents; check explicit ..
    # and whether the would-be path is under root when parents exist.
    try:
        root_r = _realpath(root)
        # Build candidate without requiring the leaf to exist
        candidate = under_root(root, rel)
        # Walk up to first existing ancestor for realpath base
        probe = candidate
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        probe_r = _realpath(probe)
        # Reconstruct remaining relative parts under real parent
        try:
            probe_r.relative_to(root_r)
        except ValueError:
            return None, f"path escapes asset-root: {rel}"
        if path_escapes_root(root, candidate if candidate.exists() else probe):
            # if leaf missing, still ensure join string has no escape
            if ".." in Path(norm).parts:
                return None, f"path escapes asset-root: {rel}"
        if candidate.exists() and path_escapes_root(root, candidate):
            return None, f"path escapes asset-root: {rel}"
    except OSError as e:
        return None, f"path resolve error: {rel} ({e})"
    return joined, None


# ---------------------------------------------------------------------------
# lightweight DOM (html.parser)
# ---------------------------------------------------------------------------


@dataclass
class DomNode:
    tag: str  # lowercase tag, or "#text"
    attrs: Dict[str, Optional[str]] = field(default_factory=dict)
    children: List["DomNode"] = field(default_factory=list)
    parent: Optional["DomNode"] = field(default=None, repr=False)
    text: str = ""

    def attr(self, name: str, default: Optional[str] = None) -> Optional[str]:
        if name in self.attrs:
            val = self.attrs[name]
            return "" if val is None else val
        # case-insensitive fallback
        low = name.lower()
        for k, v in self.attrs.items():
            if k.lower() == low:
                return "" if v is None else v
        return default

    def has_attr(self, name: str) -> bool:
        low = name.lower()
        return any(k.lower() == low for k in self.attrs)

    def classes(self) -> List[str]:
        c = self.attr("class") or ""
        return [p for p in c.split() if p]

    def id(self) -> Optional[str]:
        return self.attr("id")

    def ancestors(self) -> List["DomNode"]:
        out: List[DomNode] = []
        p = self.parent
        while p is not None:
            out.append(p)
            p = p.parent
        return out

    def walk(self) -> List["DomNode"]:
        out = [self]
        for ch in self.children:
            out.extend(ch.walk())
        return out


class _DomBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = DomNode(tag="#document")
        self._stack: List[DomNode] = [self.root]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_l = tag.lower()
        node = DomNode(tag=tag_l, attrs={k: v for k, v in attrs})
        parent = self._stack[-1]
        node.parent = parent
        parent.children.append(node)
        if tag_l not in _VOID_TAGS:
            self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in _VOID_TAGS:
            return
        # pop until matching tag (tolerant of mismatched HTML)
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag_l:
                del self._stack[i:]
                return

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_l = tag.lower()
        node = DomNode(tag=tag_l, attrs={k: v for k, v in attrs})
        parent = self._stack[-1]
        node.parent = parent
        parent.children.append(node)

    def handle_data(self, data: str) -> None:
        if not data:
            return
        parent = self._stack[-1]
        # merge adjacent text
        if parent.children and parent.children[-1].tag == "#text":
            parent.children[-1].text += data
        else:
            t = DomNode(tag="#text", text=data, parent=parent)
            parent.children.append(t)


def parse_html(html: str) -> DomNode:
    builder = _DomBuilder()
    builder.feed(html)
    builder.close()
    return builder.root


# ---------------------------------------------------------------------------
# small selector engine
# ---------------------------------------------------------------------------

# token: tag | #id | .class | [attr] | [attr=val] | [attr*=val]
_SEL_PART = re.compile(
    r"""
    (?:
        \[
            (?P<attr>[\w:-]+)
            (?:
                (?P<op>\*?=)
                (?P<q>["']?)(?P<val>[^\]"']*)(?P=q)
            )?
        \]
      | \#(?P<id>[\w:-]+)
      | \.(?P<cls>[\w:-]+)
      | (?P<tag>[\w:-]+)
    )
    """,
    re.VERBOSE,
)


def _parse_simple_selector(sel: str) -> Dict[str, Any]:
    """Parse a compound simple selector like ``nav[aria-label*=pag].x`` into constraints."""
    s = sel.strip()
    if not s:
        return {}
    constraints: Dict[str, Any] = {
        "tag": None,
        "id": None,
        "classes": [],
        "attrs": [],  # list of (name, op|None, val|None)
    }
    pos = 0
    while pos < len(s):
        m = _SEL_PART.match(s, pos)
        if not m:
            # fail closed: unparseable selector fragment
            raise ValueError(f"unparseable selector fragment near: {s[pos:]!r} in {sel!r}")
        if m.group("tag"):
            if constraints["tag"] is not None:
                raise ValueError(f"multiple tags in selector: {sel!r}")
            constraints["tag"] = m.group("tag").lower()
        elif m.group("id"):
            constraints["id"] = m.group("id")
        elif m.group("cls"):
            constraints["classes"].append(m.group("cls"))
        elif m.group("attr"):
            name = m.group("attr").lower()
            op = m.group("op")
            val = m.group("val")
            constraints["attrs"].append((name, op, val))
        pos = m.end()
    if pos != len(s):
        raise ValueError(f"unparseable selector: {sel!r}")
    return constraints


def _node_matches_constraints(node: DomNode, c: Dict[str, Any]) -> bool:
    if node.tag.startswith("#"):
        return False
    if c.get("tag") and node.tag != c["tag"]:
        return False
    if c.get("id") is not None:
        if (node.id() or "") != c["id"]:
            return False
    for cls in c.get("classes") or []:
        if cls not in node.classes():
            return False
    for name, op, val in c.get("attrs") or []:
        if op is None:
            if not node.has_attr(name):
                return False
        else:
            actual = node.attr(name)
            if actual is None:
                return False
            if op == "=":
                if actual != (val or ""):
                    return False
            elif op == "*=":
                if (val or "") not in actual:
                    return False
            else:
                return False
    return True


def split_selector_group(selector: str) -> List[str]:
    """Split comma-separated selector group, ignoring commas inside []."""
    parts: List[str] = []
    buf = ""
    depth = 0
    for ch in selector:
        if ch == "[":
            depth += 1
            buf += ch
        elif ch == "]":
            depth = max(0, depth - 1)
            buf += ch
        elif ch == "," and depth == 0:
            if buf.strip():
                parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def matches_selector(node: DomNode, selector: str) -> bool:
    """True if node matches any simple selector in a comma group."""
    try:
        for part in split_selector_group(selector):
            c = _parse_simple_selector(part)
            if _node_matches_constraints(node, c):
                return True
        return False
    except ValueError:
        return False  # caller should treat unparseable as fail-closed where needed


def matches_selector_strict(node: DomNode, selector: str) -> bool:
    """Like matches_selector but raises on unparseable selectors (fail closed)."""
    for part in split_selector_group(selector):
        c = _parse_simple_selector(part)
        if _node_matches_constraints(node, c):
            return True
    return False


def query_all(root: DomNode, selector: str) -> List[DomNode]:
    """Find all element nodes under root matching selector (comma groups OR)."""
    # Validate selector parse first (fail closed for callers that care)
    parts = split_selector_group(selector)
    constraints = [_parse_simple_selector(p) for p in parts]
    out: List[DomNode] = []
    for node in root.walk():
        if node.tag.startswith("#"):
            continue
        for c in constraints:
            if _node_matches_constraints(node, c):
                out.append(node)
                break
    return out


def query_first(root: DomNode, selector: str) -> Optional[DomNode]:
    found = query_all(root, selector)
    return found[0] if found else None


def find_by_tag(root: DomNode, tag: str) -> Optional[DomNode]:
    tag_l = tag.lower()
    for node in root.walk():
        if node.tag == tag_l:
            return node
    return None


def _style_is_hidden(style: Optional[str]) -> bool:
    """True if inline style hides the element (display:none / visibility:hidden)."""
    if not style:
        return False
    s = re.sub(r"\s+", "", style.lower())
    return "display:none" in s or "visibility:hidden" in s


def _node_is_hidden(node: DomNode) -> bool:
    """True if element is hidden from visible-text counting."""
    if node.tag.startswith("#"):
        return False
    if node.has_attr("hidden"):
        return True
    aria = (node.attr("aria-hidden") or "").strip().lower()
    if aria == "true":
        return True
    if _style_is_hidden(node.attr("style")):
        return True
    return False


def element_text(node: DomNode) -> str:
    """Raw concatenated text under node (including style/script content)."""
    parts: List[str] = []
    for n in node.walk():
        if n.tag == "#text":
            parts.append(n.text)
    return "".join(parts)


def visible_text(node: DomNode, *, exclude_chrome: bool = False) -> str:
    """Concatenate visible text descendants.

    Skips script/style/noscript/template, elements with ``hidden``,
    ``aria-hidden="true"``, or inline ``display:none`` / ``visibility:hidden``,
    and (when *exclude_chrome*) structural chrome tags nav/header/footer/aside.
    """
    parts: List[str] = []

    def rec(n: DomNode, skip: bool) -> None:
        if n.tag in _SKIP_TEXT_TAGS:
            skip = True
        if not n.tag.startswith("#"):
            if _node_is_hidden(n):
                skip = True
            if exclude_chrome and n.tag in _CHROME_TAGS:
                skip = True
        if n.tag == "#text":
            if not skip:
                parts.append(n.text)
            return
        for ch in n.children:
            rec(ch, skip)

    rec(node, False)
    # collapse whitespace for length checks
    return re.sub(r"\s+", " ", "".join(parts)).strip()


# ---------------------------------------------------------------------------
# check result
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    id: str
    status: str  # pass | fail | skip | manual
    blocking: bool
    reason: str
    detail: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "blocking": self.blocking,
            "reason": self.reason,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# URL / link helpers
# ---------------------------------------------------------------------------


def _is_external_scheme(href: str) -> bool:
    low = href.strip().lower()
    if low.startswith(("mailto:", "tel:", "javascript:", "data:", "blob:")):
        return True
    return False


def _is_http_url(href: str) -> bool:
    return href.strip().lower().startswith(("http://", "https://"))


def _strip_query_fragment(href: str) -> str:
    # split off # and ?
    s = href.split("#", 1)[0]
    s = s.split("?", 1)[0]
    return s


def _same_origin(href: str, base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    try:
        h = urlparse(href)
        b = urlparse(base_url)
        if not h.scheme and not h.netloc:
            return True  # relative
        return (h.scheme or "http") == (b.scheme or "http") and h.netloc == b.netloc
    except Exception:
        return False


def _is_internal_href(href: str, base_url: Optional[str]) -> bool:
    if not href or href.strip() == "":
        return False
    h = href.strip()
    if h.startswith("#"):
        return False  # fragment-only — not a navigable local file
    if _is_external_scheme(h):
        return False
    if _is_http_url(h):
        return _same_origin(h, base_url)
    # relative or root-relative
    return True


def _href_to_local_rel(href: str, base_url: Optional[str]) -> Optional[str]:
    """Map href to a path string under asset-root, or None if not mappable."""
    h = href.strip()
    if _is_http_url(h):
        if not _same_origin(h, base_url):
            return None
        parsed = urlparse(h)
        path = unquote(parsed.path or "")
        return path if path else None
    path = _strip_query_fragment(h)
    path = unquote(path)
    if not path or path == "#":
        return None
    return path


def _looks_like_file(path: str) -> bool:
    base = path.rstrip("/").rsplit("/", 1)[-1]
    if not base:
        return False
    return bool(_FILE_EXT_RE.search(base))


def _normalize_path_identity(path_or_url: str) -> str:
    """Normalize a path/URL to a comparable page identity.

    Drops query + fragment. Treats ``""``, ``.``, ``./``, trailing slash, and
    trailing ``index.html`` as the same directory/self form. Result is a
    path starting with ``/`` without a trailing slash (except root ``/``).
    """
    raw = (path_or_url or "").strip()
    if not raw:
        return "/"
    # If it looks like a bare relative path (no scheme), urlparse still works.
    parsed = urlparse(raw)
    if parsed.scheme or parsed.netloc or raw.startswith("/"):
        path = unquote(parsed.path or "")
    else:
        # relative token like "page.html" or "./" or "."
        path = unquote(_strip_query_fragment(raw))
    path = path.strip()
    if path in ("", ".", "./"):
        path = "/"
    # collapse /./ segments lightly
    while "/./" in path:
        path = path.replace("/./", "/")
    if path.endswith("/."):
        path = path[:-1]  # keep trailing /
    # index.html ↔ directory
    if path.endswith("/index.html"):
        path = path[: -len("index.html")]  # leaves trailing /
    elif path.endswith("index.html"):
        path = path[: -len("index.html")]
        if not path:
            path = "/"
        elif not path.endswith("/"):
            # "index.html" alone → /
            if "/" not in path.rstrip("/"):
                path = "/"
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path or "/"


def _directory_identity(path_id: str) -> str:
    """Parent-directory identity for a page path (e.g. /news/page.html → /news)."""
    if path_id in ("", "/"):
        return "/"
    if "/" not in path_id.lstrip("/"):
        # "/page.html" → "/"
        return "/"
    parent = path_id.rsplit("/", 1)[0]
    return parent or "/"


def _page_base_url(html_path: Path, base_url: Optional[str], root: DomNode) -> str:
    """Base URL used with urljoin for resolving relative hrefs."""
    if base_url and str(base_url).strip():
        return str(base_url).strip()
    # canonical link preferred when no --base-url
    for link in query_all(root, "link"):
        rel = (link.attr("rel") or "").lower().split()
        if "canonical" in rel:
            href = (link.attr("href") or "").strip()
            if href:
                return href
    # synthetic file base so urljoin resolves relatives against this page
    return f"file://{html_path.resolve().as_posix()}"


def _current_page_identities(
    html_path: Path, base_url: Optional[str], root: DomNode
) -> set:
    """Set of normalized path identities that count as 'this page'."""
    ids: set = set()
    bases: List[str] = []
    if base_url and str(base_url).strip():
        bases.append(str(base_url).strip())
    for link in query_all(root, "link"):
        rel = (link.attr("rel") or "").lower().split()
        if "canonical" in rel:
            href = (link.attr("href") or "").strip()
            if href:
                bases.append(href)
    bases.append("/" + html_path.name)
    bases.append(html_path.name)
    bases.append(f"file://{html_path.resolve().as_posix()}")

    for b in bases:
        nid = _normalize_path_identity(b)
        ids.add(nid)
        # directory form (./ and index.html land here)
        ids.add(_directory_identity(nid))
        # index.html under directory
        if nid != "/":
            ids.add(_normalize_path_identity(nid + "/index.html"))
            ids.add(_normalize_path_identity(nid + "/"))
        else:
            ids.add(_normalize_path_identity("/index.html"))
    # always include bare filename forms
    ids.add(_normalize_path_identity(html_path.name))
    ids.add(_normalize_path_identity("/" + html_path.name))
    return ids


def _current_page_paths(html_path: Path, base_url: Optional[str], root: DomNode) -> List[str]:
    """Paths that count as 'this page' for related_no_self (list form)."""
    return sorted(_current_page_identities(html_path, base_url, root))


def _href_is_self(
    href: str,
    self_ids: Sequence[str],
    base_url: Optional[str],
    *,
    html_path: Optional[Path] = None,
    root: Optional[DomNode] = None,
    page_base: Optional[str] = None,
) -> bool:
    """True if href resolves to the current page (query/fragment/./ evasions)."""
    h = (href or "").strip()
    if not h or h == "#" or h.startswith("#"):
        return True  # same-page fragment in related = self
    if _is_external_scheme(h):
        return False

    base = page_base
    if not base:
        if base_url and str(base_url).strip():
            base = str(base_url).strip()
        elif html_path is not None:
            base = _page_base_url(html_path, base_url, root or DomNode("#document"))
        else:
            base = "file:///page.html"

    # urljoin resolves "", ".", "./", "?q", "./#frag", "index.html", etc.
    resolved = urljoin(base, h)
    if _is_http_url(h) and base_url and not _same_origin(h, base_url):
        # absolute other-origin — not self
        if not _same_origin(resolved, base_url):
            return False

    target = _normalize_path_identity(resolved)
    id_set = set(self_ids)
    if target in id_set:
        return True
    # basename match against current html file name
    if html_path is not None:
        t_name = Path(target.rstrip("/")).name
        if t_name and t_name == html_path.name:
            return True
    # also compare against directory of each self id for "./" style
    if target in {_directory_identity(s) for s in id_set}:
        return True
    return False


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def _check_enabled(cfg: Dict[str, Any], check_id: str) -> Tuple[bool, bool, bool]:
    """Return (enabled, blocking, manual). Missing entry → not enabled (skip)."""
    checks = cfg.get("checks") or {}
    if check_id not in checks:
        return False, False, False
    entry = checks[check_id]
    if entry is None:
        return True, True, False
    if not isinstance(entry, dict):
        # bare true/false
        if entry is False:
            return False, False, False
        return True, True, False
    enabled = entry.get("enabled", True)
    if enabled is False:
        return False, False, False
    blocking = bool(entry.get("blocking", True))
    manual = bool(entry.get("manual", False))
    return True, blocking, manual


def check_soft_404(root: DomNode, cfg: Dict[str, Any], blocking: bool) -> CheckResult:
    soft = cfg.get("soft404") or {}
    phrases = soft.get("phrases") or []
    if not isinstance(phrases, list):
        return CheckResult(
            "soft_404",
            "fail",
            blocking,
            "soft404.phrases unparseable (not a list)",
            {"error": "phrases not list"},
        )
    min_chars = soft.get("min_content_chars", 200)
    try:
        min_chars = int(min_chars)
    except (TypeError, ValueError):
        return CheckResult(
            "soft_404",
            "fail",
            blocking,
            "soft404.min_content_chars unparseable",
            None,
        )

    main_sel = cfg.get("main_selector") or "main"
    main_node: Optional[DomNode] = None
    used_body_fallback = False
    try:
        main_node = query_first(root, main_sel)
    except ValueError as e:
        # Malformed selectors are config errors (exit 2) via validate_config;
        # if we still reach here, fail closed.
        return CheckResult(
            "soft_404",
            "fail",
            blocking,
            f"main_selector unparseable: {e}",
            None,
        )
    if main_node is None:
        main_node = find_by_tag(root, "article")
    if main_node is None:
        main_node = find_by_tag(root, "body")
        used_body_fallback = True
    if main_node is None:
        return CheckResult(
            "soft_404",
            "fail",
            blocking,
            "main content element not found (main_selector/article/body)",
            {"main_selector": main_sel},
        )

    # When counting body fallback (or the node is body), exclude chrome so
    # nav/footer boilerplate cannot satisfy min_content_chars.
    exclude_chrome = used_body_fallback or main_node.tag == "body"
    text = visible_text(main_node, exclude_chrome=exclude_chrome)
    text_len = len(text)

    # Empty main while body still has visible content → fail closed.
    if text_len == 0:
        body = find_by_tag(root, "body")
        if body is not None:
            body_text = visible_text(body, exclude_chrome=False)
            if body_text:
                return CheckResult(
                    "soft_404",
                    "fail",
                    blocking,
                    "empty main (main content empty while body has content)",
                    {"content_chars": 0, "body_chars": len(body_text)},
                )

    hits = []
    text_lower = text.lower()
    for ph in phrases:
        if ph is None:
            continue
        s = str(ph)
        if s.lower() in text_lower:
            hits.append(s)
    if hits:
        return CheckResult(
            "soft_404",
            "fail",
            blocking,
            f"soft-404 phrase(s) in main content: {hits}",
            {"phrases": hits, "content_chars": text_len},
        )
    if text_len < min_chars:
        return CheckResult(
            "soft_404",
            "fail",
            blocking,
            f"main content too short: {text_len} < {min_chars} chars",
            {"content_chars": text_len, "min_content_chars": min_chars},
        )
    return CheckResult(
        "soft_404",
        "pass",
        blocking,
        f"main content ok ({text_len} chars, no soft-404 phrases)",
        {"content_chars": text_len},
    )


def _route_key(rel: str) -> str:
    """Normalize an extensionless route path for manifest membership."""
    s = (rel or "").strip()
    s = _strip_query_fragment(s)
    if not s:
        return "/"
    if not s.startswith("/"):
        s = "/" + s
    if len(s) > 1 and s.endswith("/"):
        s = s.rstrip("/")
    return s


def check_links_internal(
    root: DomNode,
    cfg: Dict[str, Any],
    blocking: bool,
    asset_root: Path,
    base_url: Optional[str],
) -> CheckResult:
    links_cfg = cfg.get("links") or {}
    manifest_raw = links_cfg.get("routes_manifest")
    if manifest_raw is not None and not isinstance(manifest_raw, list):
        return CheckResult(
            "links_internal",
            "fail",
            blocking,
            "links.routes_manifest unparseable (not a list)",
            None,
        )
    manifest: Optional[set] = None
    if isinstance(manifest_raw, list):
        manifest = {_route_key(str(x)) for x in manifest_raw if x is not None}
    allow_unverified = bool(links_cfg.get("allow_unverified_routes", False))

    anchors = query_all(root, "a")
    checked = 0
    missing: List[str] = []
    skipped = 0
    ok = 0
    escape_fails: List[str] = []
    unverified_routes: List[str] = []

    for a in anchors:
        href = a.attr("href")
        if href is None or href.strip() == "":
            skipped += 1
            continue
        href = href.strip()
        if href.startswith("#") or _is_external_scheme(href):
            skipped += 1
            continue
        if _is_http_url(href) and not _same_origin(href, base_url):
            skipped += 1
            continue
        if not _is_internal_href(href, base_url):
            skipped += 1
            continue

        rel = _href_to_local_rel(href, base_url)
        if rel is None:
            skipped += 1
            continue
        rel = _strip_query_fragment(rel)
        if not rel or rel == "/":
            skipped += 1
            continue

        # Only enforce existence for file-like targets offline
        if not _looks_like_file(rel):
            # try as-is under asset-root; if exists count ok, else route policy
            path, err = contained_path(asset_root, rel)
            if err:
                escape_fails.append(f"{href} ({err})")
                checked += 1
                continue
            assert path is not None
            if path.is_file():
                checked += 1
                ok += 1
            elif path.is_dir() and (path / "index.html").is_file():
                checked += 1
                ok += 1
            else:
                # Extensionless SPA/server route — fail closed unless allowed.
                route = _route_key(rel)
                checked += 1
                if manifest is not None:
                    if route in manifest or rel in manifest or href in manifest:
                        ok += 1
                    else:
                        unverified_routes.append(href)
                elif allow_unverified:
                    skipped += 1
                    checked -= 1  # not really "checked" if skipped
                else:
                    unverified_routes.append(href)
            continue

        checked += 1
        path, err = contained_path(asset_root, rel)
        if err:
            escape_fails.append(f"{href} ({err})")
            missing.append(href)
            continue
        assert path is not None
        if not path.is_file():
            missing.append(href)
        else:
            ok += 1

    if escape_fails or missing or unverified_routes:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing: {', '.join(missing[:5])}")
        if escape_fails:
            parts.append(f"{len(escape_fails)} escape: {', '.join(escape_fails[:3])}")
        if unverified_routes:
            parts.append(
                f"{len(unverified_routes)} unverified routes — provide "
                f"routes_manifest or set allow_unverified_routes: "
                f"{', '.join(unverified_routes[:5])}"
            )
        return CheckResult(
            "links_internal",
            "fail",
            blocking,
            f"internal links fail ({'; '.join(parts)})",
            {
                "checked": checked,
                "ok": ok,
                "missing": missing,
                "skipped": skipped,
                "escape": escape_fails,
                "unverified_routes": unverified_routes,
            },
        )
    return CheckResult(
        "links_internal",
        "pass",
        blocking,
        f"internal links ok (checked={checked}, ok={ok}, skipped={skipped})",
        {
            "checked": checked,
            "ok": ok,
            "missing": [],
            "skipped": skipped,
            "unverified_routes": [],
        },
    )


def check_language(root: DomNode, cfg: Dict[str, Any], blocking: bool) -> CheckResult:
    lang_cfg = cfg.get("lang") or {}
    allowed = lang_cfg.get("allowed") or []
    if not isinstance(allowed, list) or not allowed:
        return CheckResult(
            "language",
            "fail",
            blocking,
            "lang.allowed missing or empty (fail closed)",
            None,
        )
    allowed_norm = [str(a).lower() for a in allowed]

    html_el = find_by_tag(root, "html")
    if html_el is None:
        return CheckResult(
            "language",
            "fail",
            blocking,
            "missing <html> element",
            None,
        )
    lang_attr = html_el.attr("lang")
    if lang_attr is None or str(lang_attr).strip() == "":
        return CheckResult(
            "language",
            "fail",
            blocking,
            "missing html lang attribute",
            None,
        )
    lang_val = str(lang_attr).strip().lower()
    # allow th-TH style prefix match against allowed base
    lang_base = lang_val.split("-", 1)[0]
    if lang_val not in allowed_norm and lang_base not in allowed_norm:
        return CheckResult(
            "language",
            "fail",
            blocking,
            f"html lang={lang_attr!r} not in allowed {allowed}",
            {"lang": lang_attr, "allowed": allowed},
        )

    bilingual = bool(lang_cfg.get("bilingual", False))
    if bilingual:
        body = find_by_tag(root, "body") or root
        body_text = visible_text(body)
        markers_th = lang_cfg.get("markers_th") or []
        markers_en = lang_cfg.get("markers_en") or []
        if not isinstance(markers_th, list) or not isinstance(markers_en, list):
            return CheckResult(
                "language",
                "fail",
                blocking,
                "bilingual markers unparseable (not lists)",
                None,
            )
        if not markers_th or not markers_en:
            return CheckResult(
                "language",
                "fail",
                blocking,
                "bilingual=true but markers_th/markers_en empty (fail closed)",
                None,
            )
        th_hit = [m for m in markers_th if str(m) in body_text]
        en_hit = [m for m in markers_en if str(m) in body_text]
        if not th_hit or not en_hit:
            missing = []
            if not th_hit:
                missing.append("TH")
            if not en_hit:
                missing.append("EN")
            return CheckResult(
                "language",
                "fail",
                blocking,
                f"bilingual page missing markers: {', '.join(missing)}",
                {"th_hits": th_hit, "en_hits": en_hit},
            )
        return CheckResult(
            "language",
            "pass",
            blocking,
            f"lang={lang_attr} allowed; bilingual markers present",
            {"lang": lang_attr, "th_hits": th_hit, "en_hits": en_hit},
        )

    return CheckResult(
        "language",
        "pass",
        blocking,
        f"lang={lang_attr} in allowed",
        {"lang": lang_attr},
    )


def _is_local_media_src(src: str) -> bool:
    if not src or not src.strip():
        return False
    s = src.strip()
    if s.startswith("data:") or s.startswith("blob:"):
        return False
    if _is_http_url(s):
        return False
    if _is_external_scheme(s):
        return False
    return True


def _node_or_ancestor_matches(node: DomNode, selector: str) -> bool:
    """Match node/ancestors against selector; re-raise parse errors (fail closed)."""
    if matches_selector_strict(node, selector):
        return True
    for anc in node.ancestors():
        if anc.tag.startswith("#"):
            continue
        if matches_selector_strict(anc, selector):
            return True
    return False


def check_file_size(
    root: DomNode,
    cfg: Dict[str, Any],
    blocking: bool,
    asset_root: Path,
) -> CheckResult:
    caps = cfg.get("file_caps") or {}
    try:
        hero_kb = float(caps.get("hero_kb", 300))
        content_kb = float(caps.get("content_kb", 150))
        video_kb = float(caps.get("video_kb", 2048))
    except (TypeError, ValueError):
        return CheckResult(
            "file_size",
            "fail",
            blocking,
            "file_caps values unparseable",
            None,
        )
    hero_sel = caps.get("hero_selector") or ".hero, [data-role=hero]"
    # Fail closed on malformed hero_selector (must not silently treat all as content).
    try:
        for part in split_selector_group(str(hero_sel)):
            _parse_simple_selector(part)
    except ValueError as e:
        return CheckResult(
            "file_size",
            "fail",
            blocking,
            f"file_caps.hero_selector unparseable: {e}",
            None,
        )

    offenders: List[Dict[str, Any]] = []
    checked = 0

    # images
    for img in query_all(root, "img"):
        src = img.attr("src")
        if src is None or not _is_local_media_src(src):
            continue
        src = src.strip()
        checked += 1
        path, err = contained_path(asset_root, _strip_query_fragment(src))
        if err:
            offenders.append({"src": src, "reason": err, "kind": "img"})
            continue
        assert path is not None
        if not path.is_file():
            offenders.append({"src": src, "reason": "file missing", "kind": "img"})
            continue
        size_b = path.stat().st_size
        size_kb = size_b / 1024.0
        try:
            is_hero = _node_or_ancestor_matches(img, hero_sel)
        except ValueError as e:
            return CheckResult(
                "file_size",
                "fail",
                blocking,
                f"file_caps.hero_selector unparseable: {e}",
                None,
            )
        cap = hero_kb if is_hero else content_kb
        klass = "hero" if is_hero else "content"
        if size_kb > cap:
            offenders.append(
                {
                    "src": src,
                    "reason": f"{klass} {size_kb:.0f}KB>{cap:.0f}",
                    "kind": "img",
                    "kb": round(size_kb, 1),
                    "cap_kb": cap,
                    "class": klass,
                }
            )

    # video src + source src
    media_nodes: List[Tuple[DomNode, str]] = []
    for v in query_all(root, "video"):
        src = v.attr("src")
        if src and _is_local_media_src(src):
            media_nodes.append((v, src.strip()))
        for ch in v.children:
            if ch.tag == "source":
                ssrc = ch.attr("src")
                if ssrc and _is_local_media_src(ssrc):
                    media_nodes.append((ch, ssrc.strip()))
    for source in query_all(root, "source"):
        # top-level source not already counted under video walk is fine to re-check
        ssrc = source.attr("src")
        if ssrc and _is_local_media_src(ssrc):
            pair = (source, ssrc.strip())
            if pair not in media_nodes and (source, ssrc.strip()) not in media_nodes:
                # avoid double-count: check by src string
                if not any(s == ssrc.strip() for _, s in media_nodes):
                    media_nodes.append(pair)

    seen_src: set = set()
    for _node, src in media_nodes:
        if src in seen_src:
            continue
        seen_src.add(src)
        checked += 1
        path, err = contained_path(asset_root, _strip_query_fragment(src))
        if err:
            offenders.append({"src": src, "reason": err, "kind": "video"})
            continue
        assert path is not None
        if not path.is_file():
            offenders.append({"src": src, "reason": "file missing", "kind": "video"})
            continue
        size_b = path.stat().st_size
        size_kb = size_b / 1024.0
        if size_kb > video_kb:
            offenders.append(
                {
                    "src": src,
                    "reason": f"video {size_kb:.0f}KB>{video_kb:.0f}",
                    "kind": "video",
                    "kb": round(size_kb, 1),
                    "cap_kb": video_kb,
                    "class": "video",
                }
            )

    if offenders:
        bits = []
        for o in offenders[:8]:
            name = Path(str(o.get("src", ""))).name or o.get("src")
            if "kb" in o:
                bits.append(f"{name} {o['kb']}KB>{o.get('cap_kb')}")
            else:
                bits.append(f"{name} ({o.get('reason')})")
        return CheckResult(
            "file_size",
            "fail",
            blocking,
            f"{len(offenders)} over cap/missing: {', '.join(bits)}",
            {"checked": checked, "offenders": offenders},
        )
    return CheckResult(
        "file_size",
        "pass",
        blocking,
        f"all local media within caps (checked={checked})",
        {"checked": checked, "offenders": []},
    )


def _count_element_descendants(node: DomNode) -> int:
    n = 0
    for ch in node.children:
        if ch.tag != "#text":
            n += 1 + _count_element_descendants(ch)
    return n


def check_pagination(root: DomNode, cfg: Dict[str, Any], blocking: bool) -> CheckResult:
    pag = cfg.get("pagination") or {}
    list_sel = pag.get("list_selector") or ".list, [data-list]"
    item_sel = pag.get("item_selector") or "li, .card"
    control_sel = pag.get("control_selector") or ".pagination, nav[aria-label*=pag]"
    try:
        threshold = int(pag.get("threshold", 12))
    except (TypeError, ValueError):
        return CheckResult(
            "pagination",
            "fail",
            blocking,
            "pagination.threshold unparseable",
            None,
        )

    try:
        containers = query_all(root, list_sel)
    except ValueError as e:
        return CheckResult(
            "pagination",
            "fail",
            blocking,
            f"list_selector unparseable: {e}",
            None,
        )

    if not containers:
        return CheckResult(
            "pagination",
            "pass",
            blocking,
            "no list containers found (nothing to paginate)",
            {"containers": 0},
        )

    offenders: List[Dict[str, Any]] = []
    for i, cont in enumerate(containers):
        try:
            items = query_all(cont, item_sel)
        except ValueError as e:
            return CheckResult(
                "pagination",
                "fail",
                blocking,
                f"item_selector unparseable: {e}",
                None,
            )
        count = len(items)
        if count == 0:
            # Stale item_selector: container has element children but 0 items
            # matched — cannot verify pagination -> fail closed.
            if _count_element_descendants(cont) > 0:
                cid = cont.attr("id") or cont.attr("class") or cont.tag
                offenders.append(
                    {"index": i, "items": 0, "id": cid, "stale": True}
                )
            continue
        if count <= threshold:
            continue
        # control within container or near (parent / siblings of parent)
        has_control = False
        try:
            if query_all(cont, control_sel):
                has_control = True
            else:
                # near: parent and its descendants one level up
                parent = cont.parent
                if parent is not None:
                    if query_all(parent, control_sel):
                        has_control = True
        except ValueError as e:
            return CheckResult(
                "pagination",
                "fail",
                blocking,
                f"control_selector unparseable: {e}",
                None,
            )
        if not has_control:
            cid = cont.attr("id") or cont.attr("class") or cont.tag
            offenders.append({"index": i, "items": count, "id": cid})

    if offenders:
        desc = ", ".join(
            f"#{o['index']}({o['id']}) "
            + ("stale item_selector (0 items but has children)" if o.get("stale") else f"n={o['items']} no control")
            for o in offenders
        )
        return CheckResult(
            "pagination",
            "fail",
            blocking,
            f"{len(offenders)} list issue(s) (threshold={threshold}): {desc}",
            {"threshold": threshold, "offenders": offenders},
        )
    return CheckResult(
        "pagination",
        "pass",
        blocking,
        f"pagination ok ({len(containers)} container(s), threshold={threshold})",
        {"containers": len(containers), "threshold": threshold},
    )


_PRM_TOKEN = "prefers-reduced-motion"


def _strip_html_comments(html: str) -> str:
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


def _reduced_motion_supported(raw_html: str, asset_root: Path) -> bool:
    """True only if ``prefers-reduced-motion`` appears in REAL CSS: an inline
    <style> block or a locally-linked stylesheet read from disk. HTML comments,
    <script> text, and page prose do NOT count (fail closed against false green).
    External http(s) CSS cannot be read offline, so it never counts as support.
    """
    stripped = _strip_html_comments(raw_html)
    for block in re.findall(
        r"<style\b[^>]*>(.*?)</style>", stripped, flags=re.DOTALL | re.IGNORECASE
    ):
        if _PRM_TOKEN in block:
            return True
    for m in re.finditer(r"<link\b[^>]*>", stripped, flags=re.IGNORECASE):
        tag = m.group(0)
        if not re.search(r'rel\s*=\s*["\']?[^"\'>]*stylesheet', tag, re.IGNORECASE):
            continue
        href_m = re.search(r'href\s*=\s*["\']([^"\']+)["\']', tag, re.IGNORECASE)
        if not href_m:
            continue
        href = href_m.group(1).strip()
        if _is_http_url(href) or href.startswith("data:") or href.startswith("//"):
            continue  # external / not readable offline
        rel = _strip_query_fragment(href)
        path, err = contained_path(asset_root, rel)
        if err or path is None or not path.is_file():
            continue
        try:
            css = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _PRM_TOKEN in css:
            return True
    return False


def check_video_attrs(
    root: DomNode,
    cfg: Dict[str, Any],
    blocking: bool,
    raw_html: str,
    asset_root: Path,
) -> CheckResult:
    vcfg = cfg.get("video") or {}
    require_muted = bool(vcfg.get("require_muted_if_autoplay", True))
    require_playsinline = bool(vcfg.get("require_playsinline", True))
    forbid_autoplay_audio = bool(vcfg.get("forbid_autoplay_with_audio", True))
    require_prm = bool(vcfg.get("require_reduced_motion_support", True))

    videos = query_all(root, "video")
    if not videos:
        return CheckResult(
            "video_attrs",
            "pass",
            blocking,
            "no <video> elements",
            {"videos": 0},
        )

    problems: List[str] = []
    any_autoplay = False
    for i, v in enumerate(videos):
        autoplay = v.has_attr("autoplay")
        muted = v.has_attr("muted")
        playsinline = v.has_attr("playsinline") or v.has_attr("webkit-playsinline")
        if autoplay:
            any_autoplay = True
            if require_muted and not muted:
                problems.append(f"video[{i}] autoplay without muted")
            if forbid_autoplay_audio and not muted:
                # same condition; still record if not already
                msg = f"video[{i}] autoplay with audio (not muted)"
                if msg not in problems and f"video[{i}] autoplay without muted" not in problems:
                    problems.append(msg)
                elif not require_muted:
                    problems.append(msg)
            if require_playsinline and not playsinline:
                problems.append(f"video[{i}] missing playsinline")
        elif require_playsinline and not playsinline:
            # playsinline required for all videos when flag set
            problems.append(f"video[{i}] missing playsinline")

    if any_autoplay and require_prm:
        if not _reduced_motion_supported(raw_html, asset_root):
            problems.append(
                "autoplay video present but no prefers-reduced-motion in inline/local CSS"
            )

    if problems:
        return CheckResult(
            "video_attrs",
            "fail",
            blocking,
            f"{len(problems)} issue(s): {'; '.join(problems[:6])}",
            {"videos": len(videos), "problems": problems},
        )
    return CheckResult(
        "video_attrs",
        "pass",
        blocking,
        f"video attributes ok ({len(videos)} video(s))",
        {"videos": len(videos)},
    )


def check_related_no_self(
    root: DomNode,
    cfg: Dict[str, Any],
    blocking: bool,
    html_path: Path,
    base_url: Optional[str],
) -> CheckResult:
    rel_cfg = cfg.get("related") or {}
    selector = rel_cfg.get("selector") or ".related, [data-related]"
    try:
        regions = query_all(root, selector)
    except ValueError as e:
        return CheckResult(
            "related_no_self",
            "fail",
            blocking,
            f"related.selector unparseable: {e}",
            None,
        )
    if not regions:
        return CheckResult(
            "related_no_self",
            "pass",
            blocking,
            "no related regions found",
            {"regions": 0},
        )

    self_paths = _current_page_paths(html_path, base_url, root)
    self_links: List[str] = []
    for region in regions:
        for a in query_all(region, "a"):
            href = a.attr("href")
            if href is None:
                continue
            href = href.strip()
            if not href:
                continue
            if _href_is_self(href, self_paths, base_url):
                self_links.append(href)

    if self_links:
        return CheckResult(
            "related_no_self",
            "fail",
            blocking,
            f"related region links to current page: {self_links[:5]}",
            {"self_links": self_links, "self_paths": self_paths},
        )
    return CheckResult(
        "related_no_self",
        "pass",
        blocking,
        f"related regions ok ({len(regions)} region(s), no self-links)",
        {"regions": len(regions)},
    )


def check_sticky_cover(
    root: DomNode,
    cfg: Dict[str, Any],
    blocking: bool,
    manual: bool,
) -> CheckResult:
    """Browser/layout-dependent — always report manual, never auto-pass/fail."""
    sticky_cfg = (cfg.get("checks") or {}).get("sticky_cover") or {}
    extra_class = None
    if isinstance(sticky_cfg, dict):
        extra_class = sticky_cfg.get("class") or sticky_cfg.get("sticky_class")

    found: List[str] = []
    for node in root.walk():
        if node.tag.startswith("#"):
            continue
        style = (node.attr("style") or "").lower().replace(" ", "")
        if "position:sticky" in style or "position:fixed" in style:
            label = node.attr("id") or (node.classes()[0] if node.classes() else node.tag)
            found.append(f"{node.tag}#{label}" if node.attr("id") else f"{node.tag}.{label}")
            continue
        classes = node.classes()
        if "sticky" in classes or "fixed" in classes:
            found.append(f"{node.tag}.{'/'.join(classes)}")
            continue
        if extra_class and extra_class in classes:
            found.append(f"{node.tag}.{extra_class}")

    # Dedup preserve order
    seen = set()
    uniq = []
    for f in found:
        if f not in seen:
            seen.add(f)
            uniq.append(f)

    reason = (
        f"layout check requires browser/visual review; sticky/fixed candidates: {uniq}"
        if uniq
        else "layout check requires browser/visual review; no sticky/fixed candidates detected in markup"
    )
    return CheckResult(
        "sticky_cover",
        "manual",
        blocking,
        reason,
        {"candidates": uniq},
    )


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

CHECK_ORDER = [
    "soft_404",
    "links_internal",
    "language",
    "file_size",
    "pagination",
    "video_attrs",
    "related_no_self",
    "sticky_cover",
]


def run_checks(
    html_path: Path,
    cfg: Dict[str, Any],
    asset_root: Path,
    base_url: Optional[str],
) -> List[CheckResult]:
    raw = html_path.read_text(encoding="utf-8", errors="replace")
    root = parse_html(raw)
    results: List[CheckResult] = []

    for cid in CHECK_ORDER:
        enabled, blocking, manual_flag = _check_enabled(cfg, cid)
        if not enabled:
            results.append(
                CheckResult(
                    cid,
                    "skip",
                    False,
                    "check not enabled in config",
                    None,
                )
            )
            continue

        if cid == "soft_404":
            results.append(check_soft_404(root, cfg, blocking))
        elif cid == "links_internal":
            results.append(
                check_links_internal(root, cfg, blocking, asset_root, base_url)
            )
        elif cid == "language":
            results.append(check_language(root, cfg, blocking))
        elif cid == "file_size":
            results.append(check_file_size(root, cfg, blocking, asset_root))
        elif cid == "pagination":
            results.append(check_pagination(root, cfg, blocking))
        elif cid == "video_attrs":
            results.append(
                check_video_attrs(root, cfg, blocking, raw, asset_root)
            )
        elif cid == "related_no_self":
            results.append(
                check_related_no_self(root, cfg, blocking, html_path, base_url)
            )
        elif cid == "sticky_cover":
            # always manual regardless of status computation
            results.append(check_sticky_cover(root, cfg, blocking, manual_flag))
        else:  # pragma: no cover — guarded by KNOWN_CHECKS
            results.append(
                CheckResult(cid, "fail", blocking, f"unknown check id: {cid}", None)
            )
    return results


def _selector_parse_error(selector: str, label: str) -> Optional[str]:
    """Return error message if selector is malformed; else None."""
    if selector is None:
        return None
    s = str(selector).strip()
    if not s:
        return f"malformed selector {label}: empty"
    try:
        parts = split_selector_group(s)
        if not parts:
            return f"malformed selector {label}: empty"
        for part in parts:
            _parse_simple_selector(part)
    except ValueError as e:
        return f"malformed selector {label}: {e}"
    return None


def _core_check_enabled_blocking(entry: Any) -> bool:
    """True if a checks.<id> entry is present as enabled + blocking."""
    if entry is False:
        return False
    if entry is None or entry is True:
        return True
    if not isinstance(entry, dict):
        # bare non-false scalar → treat as enabled blocking
        return True
    if entry.get("enabled") is False:
        return False
    if entry.get("blocking") is False:
        return False
    return True


def validate_config(cfg: Any) -> Optional[str]:
    """Return error string if config invalid; else None."""
    if not isinstance(cfg, dict):
        return "config root must be a mapping"
    checks = cfg.get("checks")
    if checks is None:
        return "config missing 'checks' section"
    if not isinstance(checks, dict):
        return "config 'checks' must be a mapping"
    if not checks:
        return "config 'checks' is empty"
    unknown = [k for k in checks.keys() if k not in KNOWN_CHECKS]
    if unknown:
        return f"unknown check id(s): {', '.join(sorted(unknown))}"

    # Core mandatory set — present, enabled, blocking (fail closed).
    for cid in sorted(CORE_CHECKS):
        if cid not in checks:
            return f"core check {cid} must be enabled and blocking"
        if not _core_check_enabled_blocking(checks[cid]):
            return f"core check {cid} must be enabled and blocking"

    # Page selectors — malformed = config error (exit 2), never silent green.
    main_sel = cfg.get("main_selector") or "main"
    err = _selector_parse_error(main_sel, "main_selector")
    if err:
        return err

    pag = cfg.get("pagination") or {}
    if not isinstance(pag, dict):
        return "pagination must be a mapping"
    for key, default in (
        ("list_selector", ".list, [data-list]"),
        ("item_selector", "li, .card"),
        ("control_selector", ".pagination, nav[aria-label*=pag]"),
    ):
        sel = pag.get(key) if key in pag else default
        if sel is None:
            sel = default
        err = _selector_parse_error(sel, f"pagination.{key}")
        if err:
            return err

    rel = cfg.get("related") or {}
    if rel is not None and not isinstance(rel, dict):
        return "related must be a mapping"
    if isinstance(rel, dict):
        rsel = rel.get("selector") if "selector" in rel else ".related, [data-related]"
        if rsel is None:
            rsel = ".related, [data-related]"
        err = _selector_parse_error(rsel, "related.selector")
        if err:
            return err

    caps = cfg.get("file_caps") or {}
    if caps is not None and not isinstance(caps, dict):
        return "file_caps must be a mapping"
    if isinstance(caps, dict):
        hsel = (
            caps.get("hero_selector")
            if "hero_selector" in caps
            else ".hero, [data-role=hero]"
        )
        if hsel is None:
            hsel = ".hero, [data-role=hero]"
        err = _selector_parse_error(hsel, "file_caps.hero_selector")
        if err:
            return err

    links = cfg.get("links")
    if links is not None:
        if not isinstance(links, dict):
            return "links must be a mapping"
        manifest = links.get("routes_manifest")
        if manifest is not None and not isinstance(manifest, list):
            return "links.routes_manifest must be a list"

    return None


def summarize(results: Sequence[CheckResult]) -> Dict[str, Any]:
    counts = {"pass": 0, "fail": 0, "skip": 0, "manual": 0, "count": len(results)}
    blocking_fail: List[str] = []
    manual_pending: List[str] = []
    blocking_pass_count = 0
    for r in results:
        st = r.status
        if st not in counts:
            # fail closed: treat unknown status as fail
            st = "fail"
        counts[st] = counts.get(st, 0) + 1
        if r.status == "fail" and r.blocking:
            blocking_fail.append(r.id)
        if r.status == "manual":
            manual_pending.append(r.id)
        # Defense-in-depth: deliverable needs real PASS among blocking checks.
        if r.status == "pass" and r.blocking:
            blocking_pass_count += 1
    # manual never counts as pass; zero verified passes → not deliverable
    deliverable = len(blocking_fail) == 0 and blocking_pass_count >= 1
    return {
        "total": counts,
        "blocking_fail": blocking_fail,
        "deliverable": deliverable,
        "manual_pending": manual_pending,
    }


def format_human(html_path: Path, results: Sequence[CheckResult], summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    for r in results:
        if r.status == "pass":
            lines.append(f"{r.id}: PASS")
        elif r.status == "fail":
            lines.append(f"{r.id}: FAIL ({r.reason})")
        elif r.status == "skip":
            lines.append(f"{r.id}: SKIP ({r.reason})")
        elif r.status == "manual":
            lines.append(f"{r.id}: MANUAL ({r.reason})")
        else:
            lines.append(f"{r.id}: {r.status.upper()} ({r.reason})")

    t = summary["total"]
    # live counts among non-skip for "X/Y" style
    considered = t["pass"] + t["fail"] + t["manual"]
    passed = t["pass"]
    bf = len(summary["blocking_fail"])
    deliv = "YES" if summary["deliverable"] else "NO"
    man = t["manual"]
    lines.append(
        f"page {html_path}: {passed}/{considered} · blocking fail {bf} · "
        f"DELIVERABLE={deliv} · manual:{man}"
    )
    fails = [r for r in results if r.status == "fail"]
    manuals = [r for r in results if r.status == "manual"]
    if fails:
        lines.append("failures:")
        for r in fails:
            lines.append(f"  - {r.id}: {r.reason}")
    if manuals:
        lines.append("manual:")
        for r in manuals:
            lines.append(f"  - {r.id}: {r.reason}")
    return "\n".join(lines)


def build_json(
    html_path: Path, results: Sequence[CheckResult], summary: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "file": str(html_path),
        "checks": [r.to_dict() for r in results],
        "total": summary["total"],
        "blocking_fail": summary["blocking_fail"],
        "deliverable": summary["deliverable"],
        "manual_pending": summary["manual_pending"],
    }


def resolve_config_path(
    config_arg: Optional[str], asset_root: Path
) -> Tuple[Optional[Path], Optional[str]]:
    if config_arg:
        p = Path(config_arg)
        if not p.is_file():
            return None, f"config file not found: {config_arg}"
        return p, None
    default = asset_root / DEFAULT_CONFIG_REL
    if default.is_file():
        return default, None
    return None, (
        f"config required: pass --config or create {DEFAULT_CONFIG_REL} under asset-root"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="page_check.py",
        description="Portable static page checker (mw-page-check). Offline, config-driven.",
    )
    parser.add_argument("html_file", help="Path to rendered HTML file")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to page-check YAML (default: <asset-root>/.work/page-check.yaml)",
    )
    parser.add_argument(
        "--asset-root",
        default=None,
        help="Root for resolving local asset paths (default: directory of html-file)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Site base URL for origin comparison / current-page identity",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    html_path = Path(args.html_file)
    if not html_path.is_file():
        print(f"error: html file not found: {args.html_file}", file=sys.stderr)
        return EXIT_ERR

    asset_root = Path(args.asset_root) if args.asset_root else html_path.parent
    if not asset_root.is_dir():
        print(f"error: asset-root is not a directory: {asset_root}", file=sys.stderr)
        return EXIT_ERR
    asset_root = asset_root.resolve()

    cfg_path, cfg_err = resolve_config_path(args.config, asset_root)
    if cfg_err:
        print(f"error: {cfg_err}", file=sys.stderr)
        return EXIT_ERR
    assert cfg_path is not None

    try:
        cfg = load_yaml_file(cfg_path)
    except MiniYamlError as e:
        print(f"error: config YAML unparseable: {e}", file=sys.stderr)
        return EXIT_ERR
    except OSError as e:
        print(f"error: cannot read config: {e}", file=sys.stderr)
        return EXIT_ERR
    except Exception as e:  # PyYAML errors etc.
        print(f"error: config load failed: {e}", file=sys.stderr)
        return EXIT_ERR

    verr = validate_config(cfg)
    if verr:
        print(f"error: {verr}", file=sys.stderr)
        return EXIT_ERR

    results = run_checks(html_path, cfg, asset_root, args.base_url)
    summary = summarize(results)

    if args.json:
        print(json.dumps(build_json(html_path, results, summary), ensure_ascii=False, indent=2))
    else:
        print(format_human(html_path, results, summary))

    if summary["blocking_fail"]:
        return EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
