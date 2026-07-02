"""
Pane-text menu parsing for session_orchestration (answerable needs-input).

omp emits no lifecycle markers, so when a session sits at ``WAITING_USER`` the
only signal we have is the raw ``tmux capture-pane`` text. This module turns
that text into an *answerable* summary: the question prose plus the ordered
list of selectable option labels.

The heuristics are ported from the (now deprecated) so-MCP orchestrator
(``z-harness/scripts/hermes/mcp_hermes_orchestrator.py``), where they were
hardened against a real 9h-parked retro menu that reached Discord as a
header-less box dump. They are deliberately dependency-free and pure so both
the omp and Claude Code pane paths can call them and so the box-drawing
assumptions stay unit-testable without a live tmux.

A box-drawn omp/Claude selection menu looks like::

     Proposal 1: ... Decision?

    ──────────────────────────────
    │ Accept (Recommended)       │
    │    Apply edits now.        │   <- indented description row, NOT an option
    │ Defer                      │
    ──────────────────────────────
     up/down navigate  enter select  esc cancel

A bare free-form prompt (omp ``❯`` / Claude ``> ``) has no box and yields no
options — callers use that to distinguish a menu from a free-form prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

#: Navigation-footer phrases that mark a selection menu but carry no answer
#: value themselves — dropped from the extracted context and options.
_MENU_FOOTER_HINTS = ("up/down", "enter select", "esc cancel", "tab ")

#: Phrases that specifically mark the *live* selection footer (the box directly
#: above them is the answerable menu, as opposed to a transient render).
_FOOTER_ANCHOR_HINTS = ("enter select", "esc cancel")

#: Leading Nerd-Font / Unicode private-use glyphs omp renders before an option
#: label (e.g. U+F192, U+F10C). Stripped from labels.
_LEADING_PUA = re.compile("^[\ue000-\uf8ff\U000f0000-\U0010fffd]+")

#: Box-drawing prefixes that start a NON-option row (rules, corners, headers).
_BOX_NONOPTION_PREFIXES = ("╭", "╰", "├", "┤", "┌", "└")


def _is_rule(stripped: str) -> bool:
    """True for a pure box-drawing separator rule (─, —, -, =, _)."""
    return bool(stripped) and all(ch in "─—-=_" for ch in stripped)


def extract_menu_options(text: str) -> list[str]:
    """Pull the ordered option labels out of a box-drawn selection menu.

    Each option is a ``│ Label │`` row; some options carry an indented
    ``│    description │`` row directly below them. Only the flush label rows
    are option text — indented rows, separator rules, the nav footer, and any
    question prose above the box are not options. Returns ``[]`` when the pane
    has no box-drawn menu (e.g. a bare ``❯`` prompt).
    """
    options: list[str] = []
    for raw in text.strip("\n").splitlines():
        stripped = raw.strip()
        if not stripped or _is_rule(stripped):
            continue
        low = stripped.lower()
        if any(hint in low for hint in _MENU_FOOTER_HINTS):
            continue
        if len(stripped) < 2 or not (stripped.startswith("│") and stripped.endswith("│")):
            continue
        inner = stripped[1:-1]
        # Strip the single mandatory padding space; anything left starting
        # with whitespace is an indented description row, not a label.
        if inner.startswith(" "):
            inner = inner[1:]
        if inner.strip() and inner[:1].isspace():
            continue
        label = inner.strip()
        if label:
            options.append(label)
    return options


def extract_needs_input_context(text: str, *, limit: int = 1600) -> str:
    """Turn a raw needs_input pane into an answerable question+options summary.

    The captured tmux pane is dominated by box-drawing rules and a navigation
    footer, and a blind ``[-N:]`` tail slice can cut the actual question off the
    top (exactly what happened when a 9h-parked retro menu reached Discord as a
    header-less box dump). This strips separator rules, unwraps ``│ … │`` box
    rows to their inner text, and drops the ``up/down … enter select`` footer,
    leaving the prompt prose and option labels. Best-effort: if nothing
    survives (unknown TUI shape), fall back to the raw tail so no signal is
    ever emitted empty.
    """
    cleaned: list[str] = []
    for raw in text.strip().splitlines():
        stripped = raw.strip()
        if not stripped or _is_rule(stripped):
            continue
        low = stripped.lower()
        if any(hint in low for hint in _MENU_FOOTER_HINTS):
            continue
        # Unwrap box rows: strip leading/trailing vertical borders + padding.
        inner = stripped.strip("│|").strip()
        if inner:
            cleaned.append(inner)
    summary = "\n".join(cleaned).strip()
    if not summary:
        return text.strip()[-limit:]
    # Keep the TAIL of the cleaned block: the live question/options sit at the
    # bottom of the pane, above the footer we already dropped.
    return summary[-limit:]


def _strip_left_border(line: str) -> Optional[str]:
    """Return the inner text of a ``│``-left-bordered row, or None.

    omp's live selection menu renders each row as ``│ <text>`` — and, when the
    pane is wide, an optional matching ``│`` far to the right. The right border
    is frequently pushed past the capture width, so we anchor only on the LEFT
    ``│`` and strip a trailing ``│`` if present.
    """
    stripped = line.rstrip()
    lead = stripped.lstrip()
    if not lead.startswith("│"):
        return None
    inner = lead[1:]
    if inner.rstrip().endswith("│"):
        inner = inner.rstrip()[:-1]
    return inner


def _footer_anchored_menu(text: str) -> Optional[Tuple[str, List[str]]]:
    """Parse omp's live selection menu, anchored on the nav footer.

    A pane can contain several box-drawn regions (a transient "Ask" render, a
    top status bar, the live menu). The authoritative, answerable menu is the
    ``│``-row block directly above the ``enter select / esc cancel`` footer.
    Returns ``(question, options)`` or None when no such footer/menu is found
    (caller falls back to the legacy whole-pane scan).

    Label vs description: option labels sit flush after ``│`` (optionally behind
    a Nerd-Font glyph); omp indents each option's description ~4 spaces. We
    strip a leading private-use glyph, then treat rows whose remaining indent is
    >= 3 as descriptions.
    """
    lines = text.rstrip("\n").splitlines()
    footer_idx = None
    for i, raw in enumerate(lines):
        low = raw.lower()
        if any(hint in low for hint in _FOOTER_ANCHOR_HINTS):
            footer_idx = i
            break
    if footer_idx is None:
        return None

    # Walk up: skip trailing blanks/rules, then collect the contiguous
    # left-bordered option block.
    i = footer_idx - 1
    while i >= 0 and (not lines[i].strip() or _is_rule(lines[i].strip())):
        i -= 1
    options: List[str] = []
    block_rows: List[str] = []
    while i >= 0:
        inner = _strip_left_border(lines[i])
        if inner is None:
            break
        block_rows.append(inner)
        i -= 1
    if not block_rows:
        return None
    block_rows.reverse()  # restore top-to-bottom order

    for inner in block_rows:
        raw_indent = len(inner) - len(inner.lstrip(" "))
        body = inner.strip()
        no_glyph = _LEADING_PUA.sub("", body)
        had_glyph = no_glyph != body
        label = no_glyph.strip()
        if not label or _is_rule(label):
            continue
        low = label.lower()
        if any(hint in low for hint in _MENU_FOOTER_HINTS):
            continue
        # Deeply-indented rows (and ↳-prefixed rows) are descriptions, unless a
        # Nerd-Font glyph marks the row as a selectable label.
        if not had_glyph and (raw_indent >= 3 or label.startswith("↳")):
            continue
        options.append(label)

    if not options:
        return None

    # Question: the nearest non-empty, non-box text line(s) above the block.
    # ``i`` already points at the line just above the option block.
    q_lines: List[str] = []
    while i >= 0:
        s = lines[i].strip()
        if not s or _is_rule(s):
            if q_lines:
                break
            i -= 1
            continue
        if s[:1] in ("│", "|") or s.startswith(_BOX_NONOPTION_PREFIXES):
            break
        q_lines.append(s)
        i -= 1
    question = " ".join(reversed(q_lines)).strip()
    return question, options


#: Lines that are omp chrome / injected noise, never the question itself.
_FREE_FORM_NOISE = (
    "hermes nudge",
    "appears to have stalled",
    "update available",
    "omp update",
    "connected to mcp",
)


def _extract_free_form_question(text: str) -> str:
    """Isolate the most recent question-like line from a free-form omp prompt.

    omp interleaves its reasoning stream, a status bar, update banners, and any
    injected Hermes nudge with the actual question, so the cleaned tail is too
    noisy to post verbatim. Prefer the LAST prose line that ends with ``?``;
    fall back to the last substantive line.
    """
    candidates: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or _is_rule(s):
            continue
        if s[:1] in "╭╰│├┤┌└╮╯|":
            continue  # box-drawing chrome (composer/status bar)
        low = s.lower()
        if any(noise in low for noise in _FREE_FORM_NOISE):
            continue
        candidates.append(s)
    questions = [c for c in candidates if c.rstrip().endswith("?")]
    if questions:
        return questions[-1]
    return candidates[-1] if candidates else ""


def extract(text: str, *, limit: int = 1600) -> tuple[str, list[str], bool]:
    """Extract ``(question, options, is_menu)`` from raw pane text.

    ``options`` is the ordered list of selectable labels (empty for a bare
    prompt). ``is_menu`` is true iff at least one selectable option was found.
    ``question`` is the prompt prose, with option labels removed so a caller can
    render the question and the numbered options without duplication. For a
    free-form prompt (no menu) ``question`` is the cleaned prompt prose.

    Strategy: prefer the footer-anchored parse (omp's live ``enter select``
    menu); fall back to the legacy whole-pane box scan (Claude-style menus and
    older shapes).
    """
    anchored = _footer_anchored_menu(text)
    if anchored is not None:
        question, options = anchored
        return question, options, True

    # Legacy fallback: whole-pane box scan.
    options = extract_menu_options(text)
    if options:
        context = extract_needs_input_context(text, limit=limit)
        opt_set = {opt for opt in options}
        question = "\n".join(
            line for line in context.splitlines() if line.strip() not in opt_set
        ).strip()
        return question, options, True

    # No box menu → free-form prompt: isolate the question from omp's noise.
    question = _extract_free_form_question(text)
    if not question:
        question = extract_needs_input_context(text, limit=limit).strip()
    return question, [], False


def resolve_menu_answer(
    row: Dict[str, Any], reply_text: str
) -> Tuple[str, Optional[list[str]]]:
    """Translate a thread reply into ``(drive_text, pre_keys)``.

    For a ``WAITING_USER`` menu row where the reply is a valid 1-based option
    number, return a natural-language selection naming the chosen label plus
    ``["Escape"]`` — the caller cancels the arrow-key menu and pastes the
    plain-language choice into the freed composer (a raw "2" selects nothing on
    such a menu). For everything else — a free-form ``❯`` prompt, a
    non-numeric reply, an out-of-range number, or missing options — return the
    reply unchanged with no pre-keys, so the existing paste path is used.
    """
    text = (reply_text or "").strip()
    if row.get("state") != "WAITING_USER" or row.get("last_input_kind") != "menu":
        # Not a selection menu (free-form ❯/composer wait): the composer already
        # accepts a pasted reply and there is no arrow-key menu to cancel, so
        # pass the reply through unchanged with no pre-keys.
        return text, None
    raw = row.get("last_options")
    try:
        options = json.loads(raw) if raw else []
    except (ValueError, TypeError):
        options = []
    if not isinstance(options, list) or not options:
        return text, None

    # A selection menu is up: arrow-key navigation PLUS an "Asking…" spinner, so
    # the pane reads as busy and a bare paste is dropped. ANY answer — a valid
    # option number OR free text (the "Other (type your own)" case) OR an
    # out-of-range number — must first send Escape to cancel the menu. Validated
    # live: Escape → "Ask tool was cancelled by the user" → idle composer with no
    # spinner, so the subsequent _wait_for_ready passes and the paste lands.
    choice = text.rstrip(".")
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        idx = int(choice)
        label = str(options[idx - 1])
        drive_text = (
            "[Hermès] You presented a selection menu. The user chose option "
            f"{idx}: «{label}». Please proceed with that choice."
        )
    else:
        drive_text = (
            "[Hermès] You presented a selection menu (now cancelled). "
            f"The user's typed answer: «{text}». Please proceed with that answer."
        )
    return drive_text, ["Escape"]
