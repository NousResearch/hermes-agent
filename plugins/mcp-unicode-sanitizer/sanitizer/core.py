"""Core sanitization engine for MCP tool metadata.

Implements Rules 1-9 of the spec (see package docstring). This module is
dependency-free (stdlib only) so it can be vendored into the Hermes MCP
Gateway without pulling in third-party packages.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Codepoint sets
# ---------------------------------------------------------------------------

# Rule 2: Unicode TAG block (U+E0000 - U+E007F). No assigned glyph in any
# mainstream renderer; the paper's central concealment mechanism (T7).
TAG_BLOCK = range(0xE0000, 0xE0080)

# Rule 3: Unicode bidi control characters.
BIDI_CONTROLS: Set[int] = {
    0x061C,  # ARABIC LETTER MARK
    0x202A,  # LEFT-TO-RIGHT EMBEDDING (LRE)
    0x202B,  # RIGHT-TO-LEFT EMBEDDING (RLE)
    0x202C,  # POP DIRECTIONAL FORMATTING (PDF)
    0x202D,  # LEFT-TO-RIGHT OVERRIDE (LRO)
    0x202E,  # RIGHT-TO-LEFT OVERRIDE (RLO)
    0x2066,  # LEFT-TO-RIGHT ISOLATE (LRI)
    0x2067,  # RIGHT-TO-LEFT ISOLATE (RLI)
    0x2068,  # FIRST STRONG ISOLATE (FSI)
    0x2069,  # POP DIRECTIONAL ISOLATE (PDI)
}

# Rule 4: invisible / zero-width characters. ZWJ/ZWNJ are handled
# contextually (see _strip_invisible) so legitimate emoji ZWJ sequences and
# Persian orthographic ZWNJ are preserved.
INVISIBLE: Set[int] = {
    0x00AD,  # SOFT HYPHEN
    0x180E,  # MONGOLIAN VOWEL SEPARATOR
    0x200B,  # ZERO WIDTH SPACE (ZWSP)
    0x200C,  # ZERO WIDTH NON-JOINER (ZWNJ)
    0x200D,  # ZERO WIDTH JOINER (ZWJ)
    0x2060,  # WORD JOINER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
}

# Rule 4: ZWJ/ZWNJ are only stripped when they are NOT part of a legitimate
# construct. ZWJ joins emoji into multi-codepoint sequences (family, skin
# tone, flags); ZWNJ is a required orthographic character in Persian and
# other scripts. Both are stripped when they appear in a suspicious context
# (adjacent to ASCII letters, i.e. a keyword-splitting attempt).
_ZWJ = 0x200D
_ZWNJ = 0x200C
_ASCII_LETTER = re.compile(r"[A-Za-z]")

# ---------------------------------------------------------------------------
# Detection pass (Rule 6 / paper §3.3) - conjunctive keyword scan
# ---------------------------------------------------------------------------

IMPERATIVE_KEYWORDS: List[str] = [
    "before answering",
    "system override",
    "<system>",
    "ignore previous",
    "disregard",
    "you must",
    "do not tell",
    "recovery required",
    "important",
]

SENSITIVE_KEYWORDS: List[str] = [
    "id_rsa",
    "api key",
    "token",
    "credential",
    "curl",
    "no-sandbox",
    "disable-seatbelt",
    "allow-network",
    "bcc",
    "exfiltrat",
    "password",
    "secret",
]

# Rule 5: confusable homoglyph folding (Cyrillic/Greek -> Latin) applied
# before the keyword scan so homoglyph-substituted keywords are caught.
HOMOGLYPH_MAP: Dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "α": "a", "ο": "o", "ρ": "p", "σ": "s", "τ": "t", "ν": "v",
    "χ": "x", "ι": "i", "κ": "k", "μ": "m", "η": "n", "ω": "w",
}


def _fold_homoglyphs(text: str) -> str:
    return "".join(HOMOGLYPH_MAP.get(c, c) for c in text)


def _detect(text: str) -> bool:
    """Conjunctive lexical detection on NFKC-normalized, lowercased text.

    Flags only if at least one imperative-framing keyword AND at least one
    sensitive-action keyword co-occur. This is the paper's baseline; it is
    deliberately conjunctive so ordinary metadata is not rejected (0/25
    benign false positives in the paper).
    """
    norm = unicodedata.normalize("NFKC", text).lower()
    norm = _fold_homoglyphs(norm)
    has_imperative = any(k in norm for k in IMPERATIVE_KEYWORDS)
    has_sensitive = any(k in norm for k in SENSITIVE_KEYWORDS)
    return has_imperative and has_sensitive


def is_dangerous_default(value: str) -> bool:
    """Rule 9: a schema default/enum is dangerous if it carries a
    sensitive-action keyword, regardless of imperative framing.

    Defaults are configuration, not prose, so the imperative-framing
    requirement of the keyword sanitizer does not apply - the value itself
    is the risk (T8).
    """
    norm = unicodedata.normalize("NFKC", value).lower()
    norm = _fold_homoglyphs(norm)
    return any(k in norm for k in SENSITIVE_KEYWORDS)


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------


@dataclass
class SanitizeResult:
    """Result of sanitizing a single string field."""

    text: str
    flagged: bool
    removed: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def concealment_present(self) -> bool:
        """True if any concealment encoding (TAG/bidi/invisible) was stripped."""
        cats = {c for c, _ in self.removed}
        return bool(cats & {"TAG_BLOCK", "BIDI", "INVISIBLE"})


def _strip_invisible(s: str) -> Tuple[str, int]:
    """Rule 4: strip invisible/zero-width chars, preserving legitimate
    ZWJ/ZWNJ constructs.

    ZWJ and ZWNJ are only removed when they sit between two ASCII letters
    (a keyword-splitting attempt, e.g. ``exfiltrat\\u200De``). Otherwise they
    are preserved so legitimate emoji ZWJ sequences and Persian orthography
    are not corrupted.
    """
    out: List[str] = []
    count = 0
    for i, ch in enumerate(s):
        cp = ord(ch)
        if cp in (INVISIBLE - {_ZWJ, _ZWNJ}):
            count += 1
            continue
        if cp in (_ZWJ, _ZWNJ):
            prev_ascii = i > 0 and bool(_ASCII_LETTER.match(s[i - 1]))
            next_ascii = i + 1 < len(s) and bool(_ASCII_LETTER.match(s[i + 1]))
            if prev_ascii and next_ascii:
                count += 1
                continue
        out.append(ch)
    return "".join(out), count


def sanitize(text: str) -> SanitizeResult:
    """Apply Rules 1-6 to a single string field.

    Returns the sanitized text plus a fail-closed flag. A field is flagged
    if EITHER (a) the residual text trips the keyword detection pass, OR
    (b) any concealment encoding was present and stripped (TAG block, bidi,
    invisible). (b) is the fail-closed half: the paper shows T7's residual
    text is benign after stripping, so keyword detection alone would miss it.
    """
    removed: List[Tuple[str, int]] = []

    # Rule 1: NFC normalization (canonical composition).
    s = unicodedata.normalize("NFC", text)

    # Rule 2: strip TAG block.
    tag_count = sum(1 for c in s if ord(c) in TAG_BLOCK)
    if tag_count:
        removed.append(("TAG_BLOCK", tag_count))
    s = "".join(c for c in s if ord(c) not in TAG_BLOCK)

    # Rule 3: strip bidi controls.
    bidi_count = sum(1 for c in s if ord(c) in BIDI_CONTROLS)
    if bidi_count:
        removed.append(("BIDI", bidi_count))
    s = "".join(c for c in s if ord(c) not in BIDI_CONTROLS)

    # Rule 4: strip invisible / zero-width (contextual for ZWJ/ZWNJ).
    s, inv_count = _strip_invisible(s)
    if inv_count:
        removed.append(("INVISIBLE", inv_count))

    # Rule 5: mixed-script confusable flag (defense-in-depth).
    latin = bool(re.search(r"[A-Za-z]", s))
    nonlatin = bool(re.search(r"[^\x00-\x7F]", s))
    if latin and nonlatin:
        removed.append(("MIXED_SCRIPT", 1))

    # Rule 6: post-sanitization re-validation (fail-closed).
    flagged = _detect(s) or any(
        cat in {c for c, _ in removed} for cat in ("TAG_BLOCK", "BIDI", "INVISIBLE")
    )

    return SanitizeResult(s, flagged, removed)


# ---------------------------------------------------------------------------
# Rule 7: re-consent on mutation (hash pinning)
# ---------------------------------------------------------------------------


def tool_hash(name: str, description: str, input_schema: dict) -> str:
    """Canonical SHA-256 of a tool definition (name + description + schema)."""
    canonical = json.dumps(
        {"name": name, "description": description, "inputSchema": input_schema},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def definition_mutated(
    approved_hash: str, name: str, description: str, input_schema: dict
) -> bool:
    """True if the current definition differs from the approved hash (T3)."""
    return tool_hash(name, description, input_schema) != approved_hash


# ---------------------------------------------------------------------------
# Rule 8: provenance-scoped tool namespaces
# ---------------------------------------------------------------------------


def namespaces_collide(host_tools: Set[str], server_tools: Set[str]) -> Set[str]:
    """Return the set of tool names a server registers that collide with
    host-trusted tools. A non-empty result means the server must be
    provenance-scoped (T6)."""
    return set(host_tools) & set(server_tools)


# ---------------------------------------------------------------------------
# Whole-tool sanitization (entry point for the gateway pipeline)
# ---------------------------------------------------------------------------


@dataclass
class SanitizedTool:
    """A tool whose metadata has been sanitized across every string surface."""

    name: str
    description: str
    input_schema: dict
    flagged: bool
    field_results: Dict[str, SanitizeResult] = field(default_factory=dict)
    dangerous_defaults: List[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        """True if no field was flagged and no dangerous default was found."""
        return not self.flagged and not self.dangerous_defaults


def _sanitize_schema(schema: dict) -> Tuple[dict, Dict[str, SanitizeResult], List[str]]:
    """Recursively sanitize every string field of an inputSchema.

    Covers parameter ``description`` fields (Rule 6/9) and ``default``/``enum``
    values (Rule 9). Returns (sanitized_schema, per-field results,
    dangerous_defaults).
    """
    results: Dict[str, SanitizeResult] = {}
    dangerous: List[str] = []

    def walk(node, path: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(value, str):
                if key == "description":
                    res = sanitize(value)
                    results[child_path] = res
                    node[key] = res.text
                elif key in ("default", "const"):
                    if is_dangerous_default(value):
                        dangerous.append(child_path)
            elif isinstance(value, dict):
                walk(value, child_path)
            elif isinstance(value, list):
                if key == "enum":
                    # Rule 9: enum values are configuration, not prose; a
                    # sensitive-action keyword alone makes them dangerous (T8).
                    for i, item in enumerate(value):
                        if isinstance(item, str) and is_dangerous_default(item):
                            dangerous.append(f"{child_path}[{i}]")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        walk(item, f"{child_path}[{i}]")

    walk(schema, "")
    return schema, results, dangerous


def sanitize_tool_metadata(tool: dict) -> SanitizedTool:
    """Sanitize every string surface of a single MCP tool definition.

    ``tool`` is a dict shaped like an MCP ``tools/list`` entry::

        {
            "name": str,
            "description": str,
            "inputSchema": { ... },
        }

    Returns a :class:`SanitizedTool` with all string fields sanitized and a
    fail-closed ``flagged``/``safe`` verdict. The caller MUST quarantine the
    tool (not deliver to the model) if ``safe`` is False.
    """
    name = str(tool.get("name", ""))
    description = str(tool.get("description", ""))
    schema = tool.get("inputSchema") or {}

    name_res = sanitize(name)
    desc_res = sanitize(description)
    schema, schema_results, dangerous = _sanitize_schema(schema)

    field_results: Dict[str, SanitizeResult] = {
        "name": name_res,
        "description": desc_res,
    }
    field_results.update(schema_results)

    flagged = name_res.flagged or desc_res.flagged or any(
        r.flagged for r in schema_results.values()
    )

    return SanitizedTool(
        name=name_res.text,
        description=desc_res.text,
        input_schema=schema,
        flagged=flagged,
        field_results=field_results,
        dangerous_defaults=dangerous,
    )
