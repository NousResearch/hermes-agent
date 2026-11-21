"""decompose_gate.py — free, deterministic pre-gate for the kanban decomposer.

The gateway auto-decomposer (``gateway/kanban_watchers.py::_auto_decompose_tick``
-> ``hermes_cli.kanban_decompose.decompose_task``) used to call an auxiliary LLM
for **every** card in the ``triage`` column, up to ``auto_decompose_per_tick``
per ~60 s tick. On the fleet that produced 3,751 cards (1,687 still open), 33%
of them immediately archived, and ~43% of them read-only recon micro-tasks — a
board flood and aux-LLM burn that fed dispatch starvation.

``should_decompose()`` is the free gate that must pass before any LLM call:

  * in ``triage``;
  * not already specified/decomposed (no ``specified``/``decomposed`` event, no
    children);
  * no explicit ``no-decompose`` marker;
  * if ``opt_in_only``: carries a ``#decompose`` marker **or** comes from an
    allowlisted ``created_by``;
  * if ``require_multi_deliverable``: looks like more than one unit of work.

``filter_children()`` post-processes an LLM fan-out: it drops read-only recon
children beyond ``max_recon_children`` and reports whether what remains is still
worth a fan-out (>= 2 children) or should collapse to a single specification.
"""
from __future__ import annotations

import re

NO_DECOMPOSE_RE = re.compile(
    r"(?i)\b(?:no[-_ ]?decompose|do[-_ ]?not[-_ ]?decompose|"
    r"single[-_ ]?task|keep[-_ ]?as[-_ ]?one)\b"
)
DECOMPOSE_MARKER_RE = re.compile(r"(?i)(?:^|\s)#decompose\b")
RECON_RE = re.compile(
    r"(?i)\b(recon|read[- ]?only|audit|map|trace|grep|sweep|dossier|"
    r"enumerate|inventory|inspect|forensic|verbatim|survey|probe)\b"
)
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S")
_AND_RE = re.compile(r"\band\b", re.IGNORECASE)


def has_no_decompose_marker(title: str, body: str = "") -> bool:
    """True when the card explicitly opts out of decomposition."""
    return bool(NO_DECOMPOSE_RE.search(f"{title or ''} {body or ''}"))


def has_opt_in_marker(title: str, body: str = "") -> bool:
    """True when the card carries an explicit ``#decompose`` opt-in marker."""
    return bool(DECOMPOSE_MARKER_RE.search(f"{title or ''}\n{body or ''}"))


def is_recon(title: str, body: str = "") -> bool:
    """True when a task looks like read-only recon/audit rather than a deliverable."""
    return bool(RECON_RE.search(title or ""))


def looks_multi_deliverable(title: str, body: str = "") -> bool:
    """Cheap heuristic: does the card describe more than one unit of work?"""
    b = body or ""
    if len(_BULLET_RE.findall(b)) >= 2:
        return True
    blob = f"{title or ''} {b}".strip()
    return len(blob) >= 80 and bool(_AND_RE.search(b))


def should_decompose(
    *,
    status: str = "triage",
    has_children: bool = False,
    specified: bool = False,
    decomposed: bool = False,
    title: str = "",
    body: str = "",
    created_by: str = "",
    opt_in_only: bool = False,
    opt_in_created_by=(),
    require_multi_deliverable: bool = False,
) -> tuple[bool, str]:
    """Return ``(eligible, reason)``. ``eligible`` gates a decomposer LLM call."""
    if status != "triage":
        return False, f"task is not in triage (status={status!r})"
    if has_children or specified or decomposed:
        return False, "already specified/decomposed"
    if has_no_decompose_marker(title, body):
        return False, "no-decompose marker"
    if opt_in_only:
        allow = {str(c).strip() for c in (opt_in_created_by or ())}
        opted = has_opt_in_marker(title, body) or (
            bool(created_by) and created_by in allow)
        if not opted:
            return False, "not opted in (#decompose or allowlisted creator)"
    if require_multi_deliverable and not looks_multi_deliverable(title, body):
        return False, "single deliverable (no fan-out shape)"
    return True, "eligible"


def filter_children(children, *, max_recon_children: int = 0):
    """Drop read-only recon children beyond ``max_recon_children``.

    ``children`` is a list of dicts with a ``title``. Returns
    ``(kept, dropped_recon)``. Callers collapse to a single specification when
    fewer than two children remain.
    """
    kept, dropped, recon_seen = [], 0, 0
    for entry in children:
        if is_recon(entry.get("title", "")):
            recon_seen += 1
            if recon_seen > max_recon_children:
                dropped += 1
                continue
        kept.append(entry)
    return kept, dropped
