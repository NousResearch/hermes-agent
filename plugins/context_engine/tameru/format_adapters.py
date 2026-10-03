"""Bounded, extractive format adapters for industrial compaction."""
from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Callable

from .contract_gates import GENERIC_WORDS, distinctive_query_terms
from .unicode_profile import graphemes, matching_shadow, search_units


@dataclass(frozen=True)
class FormatLimits:
    max_records: int = 100_000
    max_record_chars: int = 1_000_000
    max_fields: int = 4_096


@dataclass(frozen=True)
class FormatResult:
    format: str
    applied: bool
    text: str
    total_records: int = 0
    kept_records: int = 0
    structurally_valid: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _decline(
    text: str,
    format_name: str,
    reason: str,
    *,
    total_records: int = 0,
    structurally_valid: bool = True,
) -> FormatResult:
    return FormatResult(
        format=format_name,
        applied=False,
        text=text,
        total_records=total_records,
        structurally_valid=structurally_valid,
        reason=reason,
    )


def _query_selectors(query: str) -> tuple[str, ...]:
    candidates = list(distinctive_query_terms(query or ""))
    candidates.extend(search_units(query or ""))
    seen: set[str] = set()
    selectors: list[str] = []
    for raw in candidates:
        value = matching_shadow(raw).strip()
        if (
            not value
            or value.startswith("script:")
            or value in GENERIC_WORDS
            or value in seen
        ):
            continue
        if len(value) < 2 and value.isascii():
            continue
        seen.add(value)
        selectors.append(value)
        if len(selectors) >= 128:
            break
    strong = [
        value
        for value in selectors
        if any(char.isdigit() for char in value)
        or any(separator in value for separator in "._:/-")
        or any(ord(char) > 127 for char in value)
    ]
    return tuple(strong or selectors)


def _contains_selector(text: str, selectors: tuple[str, ...]) -> bool:
    shadow = matching_shadow(text)
    for selector in selectors:
        escaped = re.escape(selector)
        if selector[:1].isascii() and selector[-1:].isascii():
            pattern = re.compile(rf"(?<![\w]){escaped}(?![\w])", re.IGNORECASE)
            if pattern.search(shadow):
                return True
        elif selector in shadow:
            return True
    return False


def _json_loads(value: str):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, RecursionError):
        return None


def _looks_ndjson(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    candidates = sum(line.lstrip().startswith(("{", "[")) for line in lines[:32])
    parsed = sum(_json_loads(line) is not None for line in lines[:32])
    sample = min(len(lines), 32)
    return candidates >= max(2, sample // 2) and parsed >= 1


def _line_separator(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _split_delimited_records(text: str) -> list[str] | None:
    records: list[str] = []
    start = 0
    index = 0
    in_quotes = False
    while index < len(text):
        char = text[index]
        if char == '"':
            if in_quotes and index + 1 < len(text) and text[index + 1] == '"':
                index += 2
                continue
            in_quotes = not in_quotes
            index += 1
            continue
        if not in_quotes and char in {"\n", "\r"}:
            records.append(text[start:index])
            if char == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
            start = index + 1
        index += 1
    if in_quotes:
        return None
    if start < len(text):
        records.append(text[start:])
    elif text.endswith(("\n", "\r")):
        records.append("")
    while records and not records[-1]:
        records.pop()
    return records


def _parse_delimited_record(record: str, delimiter: str, limits: FormatLimits) -> list[str] | None:
    if len(record) > limits.max_record_chars:
        return None
    try:
        rows = list(csv.reader(io.StringIO(record), delimiter=delimiter, strict=True))
    except (csv.Error, UnicodeError):
        return None
    if len(rows) != 1 or len(rows[0]) > limits.max_fields:
        return None
    return rows[0]


def _detect_delimiter(text: str) -> str | None:
    records = _split_delimited_records(text)
    if not records or len(records) < 2:
        return None
    for delimiter in ("\t", ","):
        parsed = [_parse_delimited_record(record, delimiter, FormatLimits()) for record in records[:8]]
        if any(row is None for row in parsed):
            continue
        widths = [len(row or []) for row in parsed]
        if widths and min(widths) >= 2 and len(set(widths)) == 1:
            return delimiter
    return None


def _looks_vertical(text: str) -> bool:
    if _VERTICAL_MARKUP_RE.search(text):
        return True
    lines = [line for line in text.splitlines() if line.strip()][:128]
    if len(lines) < 6:
        return False
    short = sum(
        (len(line.strip()) if line.isascii() else len(graphemes(line.strip()))) <= 2
        for line in lines
    )
    return short / len(lines) >= 0.8


def detect_format(text: str) -> str:
    value = str(text or "")
    stripped = value.lstrip()
    if re.search(r"<!doctype\s+html|<html\b", stripped[:512], re.IGNORECASE):
        return "html"
    if stripped.startswith("<") and stripped.rstrip().endswith(">"):
        try:
            ET.fromstring(stripped)
        except (ET.ParseError, RecursionError):
            pass
        else:
            return "xml"
    if _looks_ndjson(value):
        return "ndjson"
    delimiter = _detect_delimiter(value)
    if delimiter == "\t":
        return "tsv"
    if delimiter == ",":
        return "csv"
    if len(re.findall(r"(?im)^\s*(?:CREATE|INSERT|UPDATE|DELETE|SELECT|ALTER|DROP|WITH)\b", value)) >= 2:
        return "sql"
    if (
        len(re.findall(r"(?m)^\s*\[[^\]\n]+\]\s*$", value)) >= 2
        and re.search(r"(?m)^\s*[A-Za-z_][\w.-]*\s*=", value)
    ):
        return "ini"
    if re.search(r"(?m)^#{1,6}\s+|^```|^~~~", value):
        return "markdown"
    yaml_parent = re.search(
        r"(?m)^[^\s:#][^:\n]{0,120}:\s*$",
        value,
    )
    yaml_child = re.search(
        r"(?m)^[ \t]+(?:[-?][ \t]+)?[^\s:#][^:\n]{0,120}:\s*(?:\S.*)?$",
        value,
    )
    if yaml_parent and yaml_child:
        return "yaml"
    if stripped[:1] in {"{", "["} and _json_loads(stripped) is not None:
        return "json"
    if _looks_vertical(value):
        return "vertical"
    return "text"


_VERTICAL_MARKUP_RE = re.compile(r"writing-mode\s*:", re.IGNORECASE)


def _adapt_ndjson(text: str, selectors: tuple[str, ...], limits: FormatLimits) -> FormatResult:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > limits.max_records:
        return _decline(text, "ndjson", "record limit exceeded", total_records=len(lines))
    kept: list[str] = []
    for line in lines:
        if len(line) > limits.max_record_chars or _json_loads(line) is None:
            return _decline(
                text,
                "ndjson",
                "malformed JSON record",
                total_records=len(lines),
                structurally_valid=False,
            )
        if _contains_selector(line, selectors):
            kept.append(line)
    if not kept or len(kept) == len(lines):
        return _decline(text, "ndjson", "no selective record match", total_records=len(lines))
    return FormatResult(
        format="ndjson",
        applied=True,
        text=_line_separator(text).join(kept),
        total_records=len(lines),
        kept_records=len(kept),
        reason="exact matching NDJSON records",
    )


def _adapt_delimited(
    text: str,
    selectors: tuple[str, ...],
    limits: FormatLimits,
    delimiter: str,
    format_name: str,
) -> FormatResult:
    records = _split_delimited_records(text)
    if records is None:
        return _decline(text, format_name, "unmatched quote", structurally_valid=False)
    if len(records) - 1 > limits.max_records:
        return _decline(text, format_name, "record limit exceeded", total_records=max(0, len(records) - 1))
    parsed = [_parse_delimited_record(record, delimiter, limits) for record in records]
    if not parsed or any(row is None for row in parsed):
        return _decline(text, format_name, "invalid delimited record", structurally_valid=False)
    width = len(parsed[0] or [])
    if width < 2 or any(len(row or []) != width for row in parsed[1:]):
        return _decline(text, format_name, "inconsistent field count", structurally_valid=False)
    kept = [record for record, row in zip(records[1:], parsed[1:]) if _contains_selector("\t".join(row or []), selectors)]
    total = max(0, len(records) - 1)
    if not kept or len(kept) == total:
        return _decline(text, format_name, "no selective record match", total_records=total)
    result_records = [records[0], *kept]
    return FormatResult(
        format=format_name,
        applied=True,
        text=_line_separator(text).join(result_records),
        total_records=total,
        kept_records=len(kept),
        reason="header plus exact matching records",
    )


def _adapt_markdown(text: str, selectors: tuple[str, ...], limits: FormatLimits) -> FormatResult:
    del limits
    lines = text.splitlines()
    sections: list[tuple[list[str], list[str]]] = []
    ancestors: dict[int, str] = {}
    current_ancestors: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        nonlocal current
        if current:
            sections.append((list(current_ancestors), current))
            current = []

    for line in lines:
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            current.append(line)
            continue
        heading = None if in_fence else re.match(r"^(#{1,6})\s+", line)
        if heading:
            flush()
            level = len(heading.group(1))
            ancestors = {depth: value for depth, value in ancestors.items() if depth < level}
            current_ancestors = [ancestors[depth] for depth in sorted(ancestors)]
            ancestors[level] = line
            current = [line]
        else:
            current.append(line)
    flush()
    selected = [(parents, body) for parents, body in sections if _contains_selector("\n".join(body), selectors)]
    if not selected or len(selected) == len(sections):
        return _decline(text, "markdown", "no selective section match", total_records=len(sections))
    output: list[str] = []
    seen_headings: set[str] = set()
    for parents, body in selected:
        for heading in parents:
            if heading not in seen_headings:
                output.append(heading)
                seen_headings.add(heading)
        if output and output[-1].strip() and body and body[0].strip():
            output.append("")
        output.extend(body)
        if body and body[0].startswith("#"):
            seen_headings.add(body[0])
    return FormatResult(
        format="markdown",
        applied=True,
        text="\n".join(output).strip("\n"),
        total_records=len(sections),
        kept_records=len(selected),
        reason="matching sections with ancestor headings",
    )


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _adapt_yaml(text: str, selectors: tuple[str, ...], limits: FormatLimits) -> FormatResult:
    del limits
    lines = text.splitlines()
    selected: set[int] = set()
    nonblank = [index for index, line in enumerate(lines) if line.strip()]
    matches = [index for index in nonblank if _contains_selector(lines[index], selectors)]
    for match in matches:
        match_indent = _indent(lines[match])
        anchor = match
        list_anchor_found = False
        for index in range(match, -1, -1):
            if not lines[index].strip() or lines[index].lstrip().startswith("#"):
                continue
            indent = _indent(lines[index])
            if indent <= match_indent and re.match(r"^\s*-\s+", lines[index]):
                anchor = index
                list_anchor_found = True
                break
        for index in range(match - 1, -1, -1):
            if list_anchor_found:
                break
            if not lines[index].strip() or lines[index].lstrip().startswith("#"):
                continue
            indent = _indent(lines[index])
            if indent < match_indent and lines[index].rstrip().endswith(":"):
                anchor = index
                break
        anchor_indent = _indent(lines[anchor])
        for index in range(anchor):
            if not lines[index].strip():
                continue
            indent = _indent(lines[index])
            if indent < anchor_indent and lines[index].rstrip().endswith(":"):
                selected.add(index)
        selected.add(anchor)
        for index in range(anchor + 1, len(lines)):
            if lines[index].strip() and _indent(lines[index]) <= anchor_indent:
                break
            selected.add(index)
    if not selected or len(selected) >= len(lines):
        return _decline(text, "yaml", "no selective subtree match", total_records=len(nonblank))
    output = "\n".join(lines[index] for index in sorted(selected)).strip("\n")
    return FormatResult(
        format="yaml",
        applied=True,
        text=output,
        total_records=len(nonblank),
        kept_records=sum(bool(lines[index].strip()) for index in selected),
        reason="matching subtree with parent keys",
    )


def _adapt_xml(text: str, selectors: tuple[str, ...], limits: FormatLimits) -> FormatResult:
    del limits
    upper = text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        return _decline(text, "xml", "DTD/entity declarations are not processed", structurally_valid=False)
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return _decline(text, "xml", "no line-oriented child records")
    root = re.match(r"^\s*<([A-Za-z_][\w.:-]*)(?:\s[^>]*)?>\s*$", lines[0])
    if root is None or not re.match(rf"^\s*</{re.escape(root.group(1))}>\s*$", lines[-1]):
        return _decline(text, "xml", "root is not line-oriented")
    children = lines[1:-1]
    for child in children:
        try:
            ET.fromstring(child)
        except (ET.ParseError, RecursionError):
            return _decline(text, "xml", "child spans multiple lines", structurally_valid=True)
    kept = [child for child in children if _contains_selector(child, selectors)]
    if not kept or len(kept) == len(children):
        return _decline(text, "xml", "no selective child match", total_records=len(children))
    return FormatResult(
        format="xml",
        applied=True,
        text="\n".join([lines[0], *kept, lines[-1]]),
        total_records=len(children),
        kept_records=len(kept),
        reason="balanced root plus exact child elements",
    )


def _split_sql_statements(text: str) -> list[str] | None:
    statements: list[str] = []
    start = 0
    index = 0
    quote = ""
    dollar_tag = ""
    line_comment = False
    block_comment = False
    while index < len(text):
        if line_comment:
            if text[index] in "\r\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if text.startswith("*/", index):
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if dollar_tag:
            if text.startswith(dollar_tag, index):
                index += len(dollar_tag)
                dollar_tag = ""
            else:
                index += 1
            continue
        if quote:
            char = text[index]
            if char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 2
                    continue
                quote = ""
            elif char == "\\" and index + 1 < len(text):
                index += 2
                continue
            index += 1
            continue
        if text.startswith("--", index):
            line_comment = True
            index += 2
            continue
        if text.startswith("/*", index):
            block_comment = True
            index += 2
            continue
        char = text[index]
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char == "$":
            match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", text[index:])
            if match:
                dollar_tag = match.group(0)
                index += len(dollar_tag)
                continue
        if char == ";":
            statement = text[start : index + 1].strip()
            if statement:
                statements.append(statement)
            start = index + 1
        index += 1
    if quote or dollar_tag or block_comment:
        return None
    tail = text[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def _adapt_sql(text: str, selectors: tuple[str, ...], limits: FormatLimits) -> FormatResult:
    statements = _split_sql_statements(text)
    if statements is None:
        return _decline(text, "sql", "unterminated SQL quote or comment", structurally_valid=False)
    if len(statements) > limits.max_records:
        return _decline(text, "sql", "record limit exceeded", total_records=len(statements))
    if any(len(statement) > limits.max_record_chars for statement in statements):
        return _decline(text, "sql", "statement size limit exceeded")
    kept = [statement for statement in statements if _contains_selector(statement, selectors)]
    if not kept or len(kept) == len(statements):
        return _decline(text, "sql", "no selective statement match", total_records=len(statements))
    return FormatResult(
        format="sql",
        applied=True,
        text="\n".join(kept),
        total_records=len(statements),
        kept_records=len(kept),
        reason="exact matching SQL statements",
    )


def _adapt_ini(text: str, selectors: tuple[str, ...], limits: FormatLimits) -> FormatResult:
    del limits
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if re.match(r"^\s*\[[^\]]+\]\s*$", line)]
    if len(starts) < 2:
        return _decline(text, "ini", "fewer than two complete sections")
    sections: list[str] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        sections.append("\n".join(lines[start:end]).strip("\n"))
    kept = [section for section in sections if _contains_selector(section, selectors)]
    if not kept or len(kept) == len(sections):
        return _decline(text, "ini", "no selective section match", total_records=len(sections))
    return FormatResult(
        format="ini",
        applied=True,
        text="\n\n".join(kept),
        total_records=len(sections),
        kept_records=len(kept),
        reason="exact matching INI/TOML sections",
    )


_HTML_OPEN_WRAPPER_RE = re.compile(r"^\s*(?:<!doctype\s+html\s*>|<html(?:\s[^>]*)?>|<body(?:\s[^>]*)?>)\s*$", re.IGNORECASE)
_HTML_CLOSE_WRAPPER_RE = re.compile(r"^\s*</(?:body|html)>\s*$", re.IGNORECASE)
_HTML_CHILD_RE = re.compile(r"^\s*<([A-Za-z][\w:-]*)(?:\s[^>]*)?>.*</\1>\s*$", re.IGNORECASE)


def _adapt_html(text: str, selectors: tuple[str, ...], limits: FormatLimits) -> FormatResult:
    del limits
    lines = [line for line in text.splitlines() if line.strip()]
    opening: list[str] = []
    closing: list[str] = []
    children: list[str] = []
    for line in lines:
        if _HTML_OPEN_WRAPPER_RE.match(line):
            opening.append(line)
        elif _HTML_CLOSE_WRAPPER_RE.match(line):
            closing.append(line)
        elif _HTML_CHILD_RE.match(line):
            children.append(line)
        else:
            return _decline(text, "html", "child spans multiple lines")
    if not opening or not closing or not children:
        return _decline(text, "html", "missing line-oriented wrappers or children")
    kept = [child for child in children if _contains_selector(child, selectors)]
    if not kept or len(kept) == len(children):
        return _decline(text, "html", "no selective child match", total_records=len(children))
    return FormatResult(
        format="html",
        applied=True,
        text="\n".join([*opening, *kept, *closing]),
        total_records=len(children),
        kept_records=len(kept),
        reason="HTML wrappers plus exact child elements",
    )


def _adapt_vertical(text: str, selectors: tuple[str, ...], limits: FormatLimits) -> FormatResult:
    del limits
    blocks = [block for block in re.split(r"(?:\r?\n){2,}", text) if block.strip()]
    kept = [block for block in blocks if _contains_selector("".join(block.splitlines()), selectors)]
    if not kept or len(kept) == len(blocks):
        return _decline(text, "vertical", "no selective vertical column match", total_records=len(blocks))
    return FormatResult(
        format="vertical",
        applied=True,
        text="\n\n".join(kept),
        total_records=len(blocks),
        kept_records=len(kept),
        reason="matching logical-order OCR columns",
    )


_ADAPTERS: dict[str, Callable[[str, tuple[str, ...], FormatLimits], FormatResult]] = {
    "ndjson": _adapt_ndjson,
    "csv": lambda text, selectors, limits: _adapt_delimited(text, selectors, limits, ",", "csv"),
    "tsv": lambda text, selectors, limits: _adapt_delimited(text, selectors, limits, "\t", "tsv"),
    "markdown": _adapt_markdown,
    "yaml": _adapt_yaml,
    "xml": _adapt_xml,
    "html": _adapt_html,
    "sql": _adapt_sql,
    "ini": _adapt_ini,
    "vertical": _adapt_vertical,
}


def adapt_format(
    text: str,
    query: str,
    limits: FormatLimits | None = None,
    *,
    format_name: str | None = None,
) -> FormatResult:
    value = str(text or "")
    resolved_format = format_name or detect_format(value)
    selectors = _query_selectors(query)
    if not selectors:
        return _decline(value, resolved_format, "query has no distinctive selectors")
    adapter = _ADAPTERS.get(resolved_format)
    if adapter is None:
        return _decline(value, resolved_format, "no safe extractive adapter")
    return adapter(value, selectors, limits or FormatLimits())
