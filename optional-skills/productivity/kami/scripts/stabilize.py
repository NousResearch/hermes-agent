#!/usr/bin/env python3
"""kami stabilize

Stabilize HTML templates with deterministic normalization and optional overflow solving.

Usage:
    python3 scripts/stabilize.py all
    python3 scripts/stabilize.py one-pager resume-en --out-dir /tmp/kami-stabilized --report
    python3 scripts/stabilize.py one-pager --write --strict --report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared import COOL_GRAY_BLOCKLIST, ROOT, TEMPLATES, TOKENS_FILE

PROFILES_FILE = ROOT / "references" / "stabilizer_profiles.json"
DEFAULT_OUT_DIR = ROOT / "dist" / "stabilized"

# HTML targets only. Diagrams/slides are intentionally excluded from stabilize v0.
HTML_TARGETS: dict[str, tuple[str, int]] = {
    "one-pager": ("one-pager.html", 1),
    "letter": ("letter.html", 1),
    "long-doc": ("long-doc.html", 9),
    "portfolio": ("portfolio.html", 8),
    "resume": ("resume.html", 2),
    "one-pager-en": ("one-pager-en.html", 1),
    "letter-en": ("letter-en.html", 1),
    "long-doc-en": ("long-doc-en.html", 9),
    "portfolio-en": ("portfolio-en.html", 8),
    "resume-en": ("resume-en.html", 2),
}

STYLE_BLOCK_RE = re.compile(r"(<style>\s*)(?P<css>.*?)(\s*</style>)", re.DOTALL | re.IGNORECASE)
ROOT_BLOCK_RE = re.compile(r"(^[ \t]*:root[ \t]*\{)(?P<body>.*?)(^[ \t]*\})", re.DOTALL | re.MULTILINE)
BODY_BLOCK_RE = re.compile(r"(^[ \t]*body[ \t]*\{)(?P<body>.*?)(^[ \t]*\})", re.DOTALL | re.MULTILINE)
SECTION_BLOCK_RE = re.compile(r"(^[ \t]*section[ \t]*\{)(?P<body>.*?)(^[ \t]*\})", re.DOTALL | re.MULTILINE)
PAGE_BLOCK_RE = re.compile(r"(^[ \t]*@page[ \t]*\{)(?P<body>.*?)(^[ \t]*\})", re.DOTALL | re.MULTILINE)

CSS_VAR_RE = re.compile(r"(--[\w-]+\s*:\s*)([^;]+)(;)")
RGBA_RE = re.compile(
    r"rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*((?:\d*\.\d+)|\d+)\s*\)",
    re.IGNORECASE,
)
HEX_RE = re.compile(r"#[0-9a-fA-F]{3,6}\b")
LINE_HEIGHT_RE = re.compile(r"(line-height\s*:\s*)([0-9]*\.?[0-9]+)(\s*;)", re.IGNORECASE)
FONT_SIZE_PT_RE = re.compile(r"(font-size\s*:\s*)([0-9]*\.?[0-9]+)(\s*pt\s*;)", re.IGNORECASE)
MARGIN_BOTTOM_PT_RE = re.compile(r"(margin-bottom\s*:\s*)([0-9]*\.?[0-9]+)(\s*pt\s*;)", re.IGNORECASE)
PAGE_MARGIN_MM_RE = re.compile(
    r"(margin\s*:\s*)([0-9]*\.?[0-9]+)mm\s+([0-9]*\.?[0-9]+)mm\s+([0-9]*\.?[0-9]+)mm\s+([0-9]*\.?[0-9]+)mm(\s*;)",
    re.IGNORECASE,
)

PARCHMENT_RGB = (245, 244, 237)


@dataclass
class SolverStep:
    action: str
    before: float
    after: float
    unit: str
    pages_before: int
    pages_after: int


@dataclass
class TargetResult:
    target: str
    source: str
    output: str
    changed: bool
    max_pages: int
    pages_before: int | None
    pages_after: int | None
    status: str
    rules: dict[str, int] = field(default_factory=dict)
    solver_steps: list[SolverStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "source": self.source,
            "output": self.output,
            "changed": self.changed,
            "max_pages": self.max_pages,
            "pages": {"before": self.pages_before, "after": self.pages_after},
            "status": self.status,
            "rules": self.rules,
            "solver_steps": [
                {
                    "action": s.action,
                    "before": s.before,
                    "after": s.after,
                    "unit": s.unit,
                    "pages_before": s.pages_before,
                    "pages_after": s.pages_after,
                }
                for s in self.solver_steps
            ],
            "notes": self.notes,
        }


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def round_number(value: float, digits: int = 3) -> float:
    return float(f"{value:.{digits}f}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def extract_css(html: str) -> tuple[str, re.Match[str]]:
    match = STYLE_BLOCK_RE.search(html)
    if not match:
        raise ValueError("missing <style> block")
    return match.group("css"), match


def replace_css(html: str, css: str, style_match: re.Match[str]) -> str:
    return html[: style_match.start("css")] + css + html[style_match.end("css") :]


def parse_hex(hex_code: str) -> tuple[int, int, int]:
    h = hex_code.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError(f"unsupported hex: {hex_code}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def blend_rgba_on_parchment(r: int, g: int, b: int, alpha: float) -> str:
    a = clamp(alpha, 0.0, 1.0)
    out = (
        round(PARCHMENT_RGB[0] + (r - PARCHMENT_RGB[0]) * a),
        round(PARCHMENT_RGB[1] + (g - PARCHMENT_RGB[1]) * a),
        round(PARCHMENT_RGB[2] + (b - PARCHMENT_RGB[2]) * a),
    )
    return rgb_to_hex(out)


def normalize_tokens(css: str, canonical_tokens: dict[str, str]) -> tuple[str, int]:
    match = ROOT_BLOCK_RE.search(css)
    if not match:
        return css, 0

    block = match.group("body")
    changed = 0
    for token, expected in canonical_tokens.items():
        token_name = token if token.startswith("--") else f"--{token}"
        token_re = re.compile(rf"({re.escape(token_name)}\s*:\s*)([^;]+)(;)", re.IGNORECASE)

        def token_repl(m: re.Match[str]) -> str:
            nonlocal changed
            actual = m.group(2).strip()
            if actual.lower() == expected.lower():
                return m.group(0)
            changed += 1
            return f"{m.group(1)}{expected}{m.group(3)}"

        block = token_re.sub(token_repl, block)

    if changed == 0:
        return css, 0
    new_css = css[: match.start("body")] + block + css[match.end("body") :]
    return new_css, changed


def normalize_rgba(css: str) -> tuple[str, int]:
    changed = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal changed
        r, g, b = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        a = float(m.group(4))
        if max(r, g, b) > 255:
            return m.group(0)
        solid = blend_rgba_on_parchment(r, g, b, a)
        changed += 1
        return solid

    return RGBA_RE.sub(repl, css), changed


def luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def normalize_cool_grays(css: str) -> tuple[str, int]:
    changed = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal changed
        raw = m.group(0)
        normalized = raw.lower()
        if len(normalized) == 4:
            normalized = "#" + "".join(ch * 2 for ch in normalized[1:])
        if normalized not in COOL_GRAY_BLOCKLIST:
            return raw
        lum = luminance(parse_hex(normalized))
        if lum < 0.35:
            replacement = "#4D4C48"
        elif lum < 0.72:
            replacement = "#87867F"
        else:
            replacement = "#E8E6DC"
        if replacement.lower() == normalized:
            return raw
        changed += 1
        return replacement

    return HEX_RE.sub(repl, css), changed


def clamp_line_heights(css: str, minimum: float, maximum: float) -> tuple[str, int]:
    changed = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal changed
        value = float(m.group(2))
        new_value = clamp(value, minimum, maximum)
        if abs(value - new_value) < 1e-6:
            return m.group(0)
        changed += 1
        return f"{m.group(1)}{round_number(new_value, 3)}{m.group(3)}"

    return LINE_HEIGHT_RE.sub(repl, css), changed


def _replace_in_block(
    css: str,
    block_re: re.Pattern[str],
    property_re: re.Pattern[str],
    transform: Any,
) -> tuple[str, bool, float | None, float | None]:
    block_match = block_re.search(css)
    if not block_match:
        return css, False, None, None

    block_body = block_match.group("body")
    prop_match = property_re.search(block_body)
    if not prop_match:
        return css, False, None, None

    before_value = float(prop_match.group(2))
    after_value = float(transform(before_value))
    if abs(after_value - before_value) < 1e-6:
        return css, False, before_value, before_value

    replacement = f"{prop_match.group(1)}{round_number(after_value, 3)}{prop_match.group(3)}"
    new_block = block_body[: prop_match.start()] + replacement + block_body[prop_match.end() :]
    new_css = css[: block_match.start("body")] + new_block + css[block_match.end("body") :]
    return new_css, True, before_value, after_value


def clamp_body_font(css: str, minimum: float, maximum: float) -> tuple[str, int]:
    new_css, changed, _, _ = _replace_in_block(
        css,
        BODY_BLOCK_RE,
        FONT_SIZE_PT_RE,
        lambda value: clamp(value, minimum, maximum),
    )
    return new_css, 1 if changed else 0


def clamp_body_line_height(css: str, minimum: float, maximum: float) -> tuple[str, int]:
    new_css, changed, _, _ = _replace_in_block(
        css,
        BODY_BLOCK_RE,
        LINE_HEIGHT_RE,
        lambda value: clamp(value, minimum, maximum),
    )
    return new_css, 1 if changed else 0


def clamp_section_gap(css: str, minimum: float, maximum: float) -> tuple[str, int]:
    new_css, changed, _, _ = _replace_in_block(
        css,
        SECTION_BLOCK_RE,
        MARGIN_BOTTOM_PT_RE,
        lambda value: clamp(value, minimum, maximum),
    )
    return new_css, 1 if changed else 0


def get_current_value(css: str, block_re: re.Pattern[str], prop_re: re.Pattern[str]) -> float | None:
    block_match = block_re.search(css)
    if not block_match:
        return None
    prop_match = prop_re.search(block_match.group("body"))
    if not prop_match:
        return None
    return float(prop_match.group(2))


def tighten_section_gap(css: str, minimum: float, step: float) -> tuple[str, bool, float, float]:
    value = get_current_value(css, SECTION_BLOCK_RE, MARGIN_BOTTOM_PT_RE)
    if value is None or value <= minimum + 1e-6:
        return css, False, 0.0, 0.0
    target = max(minimum, value - step)
    new_css, changed, before, after = _replace_in_block(
        css,
        SECTION_BLOCK_RE,
        MARGIN_BOTTOM_PT_RE,
        lambda _: target,
    )
    return new_css, changed, before or value, after or value


def tighten_body_line_height(css: str, minimum: float, step: float) -> tuple[str, bool, float, float]:
    value = get_current_value(css, BODY_BLOCK_RE, LINE_HEIGHT_RE)
    if value is None or value <= minimum + 1e-6:
        return css, False, 0.0, 0.0
    target = max(minimum, value - step)
    new_css, changed, before, after = _replace_in_block(
        css,
        BODY_BLOCK_RE,
        LINE_HEIGHT_RE,
        lambda _: target,
    )
    return new_css, changed, before or value, after or value


def tighten_body_font(css: str, minimum: float, step: float) -> tuple[str, bool, float, float]:
    value = get_current_value(css, BODY_BLOCK_RE, FONT_SIZE_PT_RE)
    if value is None or value <= minimum + 1e-6:
        return css, False, 0.0, 0.0
    target = max(minimum, value - step)
    new_css, changed, before, after = _replace_in_block(
        css,
        BODY_BLOCK_RE,
        FONT_SIZE_PT_RE,
        lambda _: target,
    )
    return new_css, changed, before or value, after or value


def tighten_page_margin(
    css: str,
    min_margins: list[float],
    step: float,
) -> tuple[str, bool, float, float]:
    block_match = PAGE_BLOCK_RE.search(css)
    if not block_match:
        return css, False, 0.0, 0.0
    block = block_match.group("body")
    margin_match = PAGE_MARGIN_MM_RE.search(block)
    if not margin_match:
        return css, False, 0.0, 0.0

    before = [float(margin_match.group(i)) for i in range(2, 6)]
    target = [max(min_margins[idx], before[idx] - step) for idx in range(4)]
    if all(abs(before[idx] - target[idx]) < 1e-6 for idx in range(4)):
        return css, False, 0.0, 0.0

    replacement = (
        f"{margin_match.group(1)}"
        f"{round_number(target[0], 3)}mm {round_number(target[1], 3)}mm "
        f"{round_number(target[2], 3)}mm {round_number(target[3], 3)}mm"
        f"{margin_match.group(6)}"
    )
    new_block = block[: margin_match.start()] + replacement + block[margin_match.end() :]
    new_css = css[: block_match.start("body")] + new_block + css[block_match.end("body") :]
    before_sum = sum(before)
    after_sum = sum(target)
    return new_css, True, before_sum, after_sum


def count_pages(html: str, base_dir: Path) -> int:
    try:
        from weasyprint import HTML
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("missing deps: pip install weasyprint pypdf --break-system-packages") from exc

    tmp_pdf: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="kami-stabilize-", suffix=".pdf", delete=False) as tmp:
            tmp_pdf = Path(tmp.name)
        HTML(string=html, base_url=str(base_dir)).write_pdf(str(tmp_pdf))
        return len(PdfReader(str(tmp_pdf)).pages)
    finally:
        if tmp_pdf and tmp_pdf.exists():
            tmp_pdf.unlink()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid json root object (expected dict): {path}")
    return raw


def _as_float(value: Any, field: str, target: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{target}: {field} must be numeric, got {value!r}") from exc


def _validate_range(profile: dict[str, Any], key: str, target: str) -> None:
    node = profile.get(key)
    if not isinstance(node, dict):
        raise ValueError(f"{target}: missing object field '{key}'")
    if "min" not in node or "max" not in node:
        raise ValueError(f"{target}: '{key}' must contain 'min' and 'max'")
    minimum = _as_float(node["min"], f"{key}.min", target)
    maximum = _as_float(node["max"], f"{key}.max", target)
    if minimum > maximum:
        raise ValueError(f"{target}: '{key}.min' ({minimum}) must be <= '{key}.max' ({maximum})")


def validate_profile(profile: dict[str, Any], target: str) -> None:
    for key in ("line_height", "body_font_size_pt", "body_line_height", "section_gap_pt"):
        _validate_range(profile, key, target)

    page_margin = profile.get("page_margin_mm")
    if not isinstance(page_margin, dict):
        raise ValueError(f"{target}: missing object field 'page_margin_mm'")
    min_margins = page_margin.get("min")
    if not isinstance(min_margins, list) or len(min_margins) != 4:
        raise ValueError(f"{target}: 'page_margin_mm.min' must be an array of 4 numbers")
    for idx, value in enumerate(min_margins):
        _as_float(value, f"page_margin_mm.min[{idx}]", target)

    solver = profile.get("overflow_solver")
    if not isinstance(solver, dict):
        raise ValueError(f"{target}: missing object field 'overflow_solver'")
    if "enabled" in solver and not isinstance(solver["enabled"], bool):
        raise ValueError(f"{target}: 'overflow_solver.enabled' must be a boolean")


def resolve_targets(raw_targets: list[str]) -> list[str]:
    if not raw_targets or raw_targets == ["all"] or "all" in raw_targets:
        return list(HTML_TARGETS.keys())

    resolved: list[str] = []
    for target in raw_targets:
        if target in HTML_TARGETS:
            resolved.append(target)
            continue
        # Accept source file names as aliases.
        normalized = target.removesuffix(".html")
        if normalized in HTML_TARGETS:
            resolved.append(normalized)
            continue
        for name, (source, _max_pages) in HTML_TARGETS.items():
            if source == target or source == f"{normalized}.html":
                resolved.append(name)
                break
        else:
            raise ValueError(f"unknown target: {target}")
    return resolved


def run_for_target(
    target: str,
    source_file: str,
    max_pages: int,
    profile: dict[str, Any],
    canonical_tokens: dict[str, str],
    write_in_place: bool,
    out_dir: Path,
    strict: bool,
) -> TargetResult:
    validate_profile(profile, target)

    source_path = TEMPLATES / source_file
    html = source_path.read_text(encoding="utf-8")
    css, style_match = extract_css(html)

    pages_before: int | None = None
    pages_after: int | None = None
    notes: list[str] = []

    try:
        pages_before = count_pages(html, source_path.parent)
    except RuntimeError as exc:
        notes.append(str(exc))

    rule_hits: dict[str, int] = {}
    changed = False

    css, hits = normalize_tokens(css, canonical_tokens)
    if hits:
        changed = True
    rule_hits["token_sync"] = hits

    css, hits = normalize_rgba(css)
    if hits:
        changed = True
    rule_hits["rgba_to_solid"] = hits

    css, hits = normalize_cool_grays(css)
    if hits:
        changed = True
    rule_hits["cool_gray_normalized"] = hits

    line_min = float(profile["line_height"]["min"])
    line_max = float(profile["line_height"]["max"])
    css, hits = clamp_line_heights(css, line_min, line_max)
    if hits:
        changed = True
    rule_hits["line_height_clamped"] = hits

    body_font_min = float(profile["body_font_size_pt"]["min"])
    body_font_max = float(profile["body_font_size_pt"]["max"])
    css, hits = clamp_body_font(css, body_font_min, body_font_max)
    if hits:
        changed = True
    rule_hits["body_font_clamped"] = hits

    body_lh_min = float(profile["body_line_height"]["min"])
    body_lh_max = float(profile["body_line_height"]["max"])
    css, hits = clamp_body_line_height(css, body_lh_min, body_lh_max)
    if hits:
        changed = True
    rule_hits["body_line_height_clamped"] = hits

    gap_min = float(profile["section_gap_pt"]["min"])
    gap_max = float(profile["section_gap_pt"]["max"])
    css, hits = clamp_section_gap(css, gap_min, gap_max)
    if hits:
        changed = True
    rule_hits["section_gap_clamped"] = hits

    html_after = replace_css(html, css, style_match)
    try:
        pages_after = count_pages(html_after, source_path.parent)
    except RuntimeError as exc:
        if str(exc) not in notes:
            notes.append(str(exc))

    solver_steps: list[SolverStep] = []
    solver_config = profile["overflow_solver"]
    if (
        max_pages > 0
        and solver_config.get("enabled", False)
        and pages_after is not None
        and pages_after > max_pages
        and not notes
    ):
        max_iterations = int(solver_config.get("max_iterations", 40))
        for _ in range(max_iterations):
            if pages_after <= max_pages:
                break

            progressed = False
            actions = [
                (
                    "tighten_section_gap",
                    lambda c: tighten_section_gap(
                        c,
                        gap_min,
                        float(solver_config.get("section_gap_step_pt", 1.0)),
                    ),
                    "pt",
                ),
                (
                    "tighten_body_line_height",
                    lambda c: tighten_body_line_height(
                        c,
                        body_lh_min,
                        float(solver_config.get("body_line_height_step", 0.02)),
                    ),
                    "ratio",
                ),
                (
                    "tighten_body_font",
                    lambda c: tighten_body_font(
                        c,
                        body_font_min,
                        float(solver_config.get("body_font_step_pt", 0.2)),
                    ),
                    "pt",
                ),
                (
                    "tighten_page_margin",
                    lambda c: tighten_page_margin(
                        c,
                        [float(v) for v in profile["page_margin_mm"]["min"]],
                        float(solver_config.get("margin_step_mm", 1.0)),
                    ),
                    "mm_total",
                ),
            ]

            for action_name, action_fn, unit in actions:
                candidate_css, applied, before, after = action_fn(css)
                if not applied:
                    continue
                candidate_html = replace_css(html, candidate_css, style_match)
                candidate_pages = count_pages(candidate_html, source_path.parent)
                solver_steps.append(
                    SolverStep(
                        action=action_name,
                        before=round_number(before, 3),
                        after=round_number(after, 3),
                        unit=unit,
                        pages_before=pages_after,
                        pages_after=candidate_pages,
                    )
                )
                css = candidate_css
                html_after = candidate_html
                pages_after = candidate_pages
                changed = True
                progressed = True
                if pages_after <= max_pages:
                    break

            if not progressed:
                notes.append("solver reached lower bounds before meeting page constraint")
                break

    if write_in_place:
        output_path = source_path
        if changed:
            output_path.write_text(html_after, encoding="utf-8")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / source_file
        output_path.write_text(html_after, encoding="utf-8")

    status = "ok"
    page_ok = True
    if max_pages > 0:
        if pages_after is None:
            status = "unverified-pages"
            page_ok = not strict
        elif pages_after > max_pages:
            status = "overflow"
            page_ok = False
    if strict and not page_ok:
        status = "failed-strict"

    result = TargetResult(
        target=target,
        source=str(source_path.relative_to(ROOT)),
        output=str(output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path),
        changed=changed,
        max_pages=max_pages,
        pages_before=pages_before,
        pages_after=pages_after,
        status=status,
        rules=rule_hits,
        solver_steps=solver_steps,
        notes=notes,
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stabilize Kami HTML templates.")
    parser.add_argument("targets", nargs="*", default=["all"], help="Target names or 'all'")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes back to source templates (default writes to --out-dir)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory when not using --write",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a JSON report with before/after pages and applied rules",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when constrained targets still exceed page limits",
    )
    return parser


def main(argv: list[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv[1:])

    try:
        targets = resolve_targets(args.targets)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        tokens = load_json(TOKENS_FILE)
        profiles = load_json(PROFILES_FILE)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if not all(isinstance(k, str) and isinstance(v, str) for k, v in tokens.items()):
        print("ERROR: invalid tokens.json, expected string->string map")
        return 2

    defaults = profiles.get("defaults", {})
    target_profiles = profiles.get("targets", {})
    if not isinstance(defaults, dict) or not isinstance(target_profiles, dict):
        print("ERROR: invalid stabilizer_profiles.json, expected keys: defaults(object), targets(object)")
        return 2

    out_dir = Path(args.out_dir)
    all_results: list[TargetResult] = []
    strict_failed = False

    for target in targets:
        source_file, max_pages = HTML_TARGETS[target]
        merged_profile = deep_merge(defaults, target_profiles.get(target, {}))
        try:
            result = run_for_target(
                target=target,
                source_file=source_file,
                max_pages=max_pages,
                profile=merged_profile,
                canonical_tokens=tokens,
                write_in_place=args.write,
                out_dir=out_dir,
                strict=args.strict,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2
        all_results.append(result)
        strict_failed = strict_failed or result.status == "failed-strict"

        page_note = (
            f"{result.pages_after}/{result.max_pages}"
            if result.max_pages > 0 and result.pages_after is not None
            else (str(result.pages_after) if result.pages_after is not None else "n/a")
        )
        print(
            f"{result.status.upper()}: {result.target} | changed={result.changed} "
            f"| pages={page_note} | output={result.output}"
        )
        for note in result.notes:
            print(f"  note: {note}")

    if args.report:
        report_path = out_dir / "stabilize-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "write_mode": bool(args.write),
            "strict": bool(args.strict),
            "targets": [r.to_json() for r in all_results],
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        rel = report_path.relative_to(ROOT) if report_path.is_relative_to(ROOT) else report_path
        print(f"REPORT: {rel}")

    if strict_failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
