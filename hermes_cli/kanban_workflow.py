"""Workflow templates: parse, validate, instantiate, gate.

Source of truth: ``hermes_cli/kanban_templates/<template_id>.md`` — a
markdown file with a YAML frontmatter block (machine contract) followed
by prose (human rationale). The CLI implements the spec verbatim; the
spec is the normative contract (t_0ddc29ce ↔ t_548066d6 ↔ the spec).

Public surface:

* ``load_template(template_id, *, templates_dir=None)`` — parse the
  template spec into a ``WorkflowTemplate`` dataclass.
* ``list_templates(*, templates_dir=None)`` — enumerate every template
  id on disk (used by ``workflow list``).
* ``resolve_source_card(card_ref, conn)`` — card-id or VULN-id →
  ``kb.Task`` (with VULN-id disambiguation per spec §7.1).
* ``validate_source(card, template, *, profiles)`` — return
  ``ValidationResult`` (errors + extracted fields); do not write.
* ``instantiate_chain(card, template, *, validation, assignee_overrides,
  dry_run=False, profiles)`` — return a ``ChainPlan`` describing the
  would-be children; if ``dry_run=False`` actually create them
  (ship card sticky-blocked).
* ``cmd_workflow(args)`` — argparse handler.

The ship-card sticky-block is enforced here, not in ``create_task``,
because the spec requires create + block + parent-link in one
transaction. Doing it at the template layer keeps ``create_task``
general (it is also used for triage, swarm, etc.).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from agent.skill_utils import yaml_load

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _default_templates_dir() -> Path:
    """Packaged default: ``<hermes_cli>/kanban_templates``.

    Resolved at import time so tests can monkeypatch via
    ``HERMES_KANBAN_TEMPLATES_DIR`` before this module loads.
    """
    return Path(__file__).resolve().parent / "kanban_templates"


def _templates_dir() -> Path:
    """Return the active templates dir, honouring the env override."""
    override = os.environ.get("HERMES_KANBAN_TEMPLATES_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_templates_dir()


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
# VULN-id format. Spec §3.3 writes ``VULN-[A-Z]{2}-\d{3}`` but the
# spec's own fixture (§10) and live board usage include 3-letter codes
# (``VULN-TST-001``, ``VULN-REM-004``) and digit-bearing family codes
# (``VULN-E2E-001``, the synthetic e2e gate-review fixture). Accept a
# letter-first alphanumeric family of 2-4 chars to match the actual
# data; the spec's prose is tightened in a follow-up edit.
_VULN_ID_RE = re.compile(r"VULN-[A-Z][A-Z0-9]{1,3}-\d{3}")
_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
_DONE_WHEN_RE = re.compile(
    r"(?im)^\s*done when\s*:[ \t]*\n?(.*?)(?=^\s*$\n|^\s*Idempotency\s*:|\Z)",
    re.DOTALL,
)
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*(\d+)\)\s+(.+?)\s*$")
_DASHED_ITEM_RE = re.compile(r"(?m)^\s*-\s+(.+?)\s*$")


@dataclass(frozen=True)
class TemplateStep:
    key: str
    title: str
    assignee: str
    assignee_fallback: str
    gate: str  # "auto" | "approval"


@dataclass(frozen=True)
class WorkflowTemplate:
    template_id: str
    version: str
    spec_file: str
    status: str
    input_role: str
    severity_tag: str
    severities: tuple[str, ...]
    required_fields: tuple[str, ...]
    done_when_section_required: bool
    steps: tuple[TemplateStep, ...]
    placeholders: tuple[str, ...]
    ship_step_key: str
    path: Path

    def step(self, key: str) -> TemplateStep:
        for s in self.steps:
            if s.key == key:
                return s
        raise KeyError(f"template {self.template_id!r} has no step {key!r}")


@dataclass
class ValidationResult:
    """Outcome of validating a source card against a template's input contract.

    ``errors`` is empty on success; ``fields`` carries the extracted values
    for placeholder substitution. The order of errors matches spec §3.5.
    """

    errors: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    done_when_mapped: dict[str, list[str]] = field(default_factory=dict)
    done_when_verbatim: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors


def load_template(template_id: str, *, templates_dir: Optional[Path] = None) -> WorkflowTemplate:
    """Load + parse a single template by id.

    Raises ``WorkflowTemplateError`` for unknown id or malformed spec.
    The filename MUST equal the frontmatter ``template_id`` (spec §7.1).
    """
    base = Path(templates_dir) if templates_dir is not None else _templates_dir()
    path = (base / f"{template_id}.md").resolve()
    # Containment: refuse to follow a symlink that escapes base.
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise WorkflowTemplateError(
            f"template {template_id!r}: path escapes templates dir"
        ) from exc
    if not path.is_file():
        raise WorkflowTemplateError(
            f"template {template_id!r} not found at {path} "
            f"(searched {base})"
        )
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise WorkflowTemplateError(
            f"template {template_id!r}: missing YAML frontmatter (expected "
            f"file to start with '---')"
        )
    try:
        fm = yaml_load(m.group(1))
    except Exception as exc:
        raise WorkflowTemplateError(
            f"template {template_id!r}: invalid YAML frontmatter: {exc}"
        ) from exc
    if not isinstance(fm, dict):
        raise WorkflowTemplateError(
            f"template {template_id!r}: frontmatter must be a mapping"
        )
    fm_template_id = str(fm.get("template_id", "")).strip()
    if fm_template_id != template_id:
        raise WorkflowTemplateError(
            f"template {template_id!r}: frontmatter template_id mismatch "
            f"(got {fm_template_id!r}); filename must equal template_id"
        )
    inp = fm.get("input") or {}
    if not isinstance(inp, dict):
        raise WorkflowTemplateError(
            f"template {template_id!r}: 'input' must be a mapping"
        )
    raw_steps = fm.get("steps") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        raise WorkflowTemplateError(
            f"template {template_id!r}: 'steps' must be a non-empty list"
        )
    steps: list[TemplateStep] = []
    seen_keys: set[str] = set()
    for raw in raw_steps:
        if not isinstance(raw, dict):
            raise WorkflowTemplateError(
                f"template {template_id!r}: each step must be a mapping"
            )
        key = str(raw.get("key", "")).strip()
        if not key or key in seen_keys:
            raise WorkflowTemplateError(
                f"template {template_id!r}: duplicate or empty step key {key!r}"
            )
        seen_keys.add(key)
        steps.append(
            TemplateStep(
                key=key,
                title=str(raw.get("title", "")).strip(),
                assignee=str(raw.get("assignee", "")).strip(),
                assignee_fallback=str(raw.get("assignee_fallback", "")).strip(),
                gate=str(raw.get("gate", "auto")).strip(),
            )
        )
    ship_step = next((s for s in steps if s.gate == "approval"), None)
    if ship_step is None:
        raise WorkflowTemplateError(
            f"template {template_id!r}: no step with gate='approval' "
            f"(spec requires exactly one ship/approval step)"
        )
    return WorkflowTemplate(
        template_id=template_id,
        version=str(fm.get("version", "0.0.0")).strip(),
        spec_file=str(fm.get("spec_file", "")).strip(),
        status=str(fm.get("status", "")).strip(),
        input_role=str(inp.get("role", "")).strip(),
        severity_tag=str(inp.get("severity_tag", "")).strip(),
        severities=tuple(str(s).strip() for s in inp.get("severities") or ()),
        required_fields=tuple(str(s).strip() for s in inp.get("required_fields") or ()),
        done_when_section_required=bool(inp.get("done_when_section_required", False)),
        steps=tuple(steps),
        placeholders=tuple(str(s).strip() for s in fm.get("placeholders") or ()),
        ship_step_key=ship_step.key,
        path=path,
    )


def list_templates(*, templates_dir: Optional[Path] = None) -> list[WorkflowTemplate]:
    """Enumerate every well-formed template in the dir (skip malformed)."""
    base = Path(templates_dir) if templates_dir is not None else _templates_dir()
    if not base.is_dir():
        return []
    out: list[WorkflowTemplate] = []
    for entry in sorted(base.glob("*.md")):
        try:
            out.append(load_template(entry.stem, templates_dir=base))
        except WorkflowTemplateError:
            # Mirror the spec's "skip malformed" behaviour for `workflow list`
            # — load_template's id was the filename stem, so a bad frontmatter
            # must not break listing.
            continue
    return out


# ---------------------------------------------------------------------------
# Source-card resolution + validation
# ---------------------------------------------------------------------------

class WorkflowTemplateError(Exception):
    """Raised when a template spec is missing, malformed, or misused."""


def resolve_source_card(
    card_ref: str, conn, *, allow_archived: bool = False,
) -> kb.Task:
    """Resolve a card reference (id or VULN-id) to a Task.

    Implements spec §7.1 VULN-id resolution:

    1. If ``card_ref`` looks like a task id (``t_<hex>``), return it
       (or raise if unknown).
    2. Otherwise treat as a VULN-id. Find tasks (non-archived unless
       ``allow_archived``) whose TITLE contains the id. Among them,
       prefer scout-created (`created_by=='scout'`) or `[gh]`-prefixed
       titles. If exactly one candidate remains, return it; else raise
       a multi-candidate error.
    """
    card_ref = (card_ref or "").strip()
    if not card_ref:
        raise WorkflowValidationError("source card reference is empty")
    # Task id path
    if re.fullmatch(r"t_[0-9a-f]+", card_ref):
        t = kb.get_task(conn, card_ref)
        if t is None:
            raise WorkflowValidationError(f"unknown card id {card_ref!r}")
        return t
    # VULN-id path
    vuln_match = _VULN_ID_RE.search(card_ref)
    if not vuln_match:
        raise WorkflowValidationError(
            f"card reference {card_ref!r} is neither a task id (t_…) "
            f"nor a VULN-id (VULN-CC-NNN)"
        )
    vuln_id = vuln_match.group(0)
    where = "1=1" if allow_archived else "status != 'archived'"
    rows = conn.execute(
        f"SELECT id, title, created_by FROM tasks WHERE {where}",
    ).fetchall()
    candidates = [r for r in rows if vuln_id in (r["title"] or "")]
    if not candidates:
        raise WorkflowValidationError(
            f"no card with VULN-id {vuln_id!r} in title (non-archived)"
        )
    preferred = [
        r for r in candidates
        if (r["created_by"] == "scout") or (r["title"] or "").lstrip().startswith("[gh]")
    ]
    if not preferred:
        # Fall back to all title-hits — the title match is the primary signal.
        preferred = candidates
    if len(preferred) > 1:
        ids = ", ".join(r["id"] for r in preferred)
        raise WorkflowValidationError(
            f"ambiguous VULN-id {vuln_id!r}: matches multiple non-archived "
            f"cards: {ids}. Re-run with the specific card id."
        )
    t = kb.get_task(conn, preferred[0]["id"])
    if t is None:
        # Should not happen — fall through to the same "unknown" error shape.
        raise WorkflowValidationError(f"unknown card id {preferred[0]['id']!r}")
    return t


def validate_source(
    card: kb.Task,
    template: WorkflowTemplate,
    *,
    profiles: Sequence[str],
    assignees_resolved: dict[str, str],
) -> ValidationResult:
    """Run the spec §3 input-contract checks against the source card.

    Mutates ``assignees_resolved`` in place so the caller (instantiate_chain)
    reuses the resolved assignee map without re-walking the profile list.

    Order of errors matches spec §3.5: unknown card → not a scout issue
    → not SEC-tagged → missing severity → missing required field(s) →
    unknown assignee. We never write here, even on failure.
    """
    res = ValidationResult()
    if not is_scout_card(card):
        res.errors.append(
            f"card {card.id!r} is not a scout issue (need created_by='scout' "
            f"or [gh]-prefixed title or scout:* idempotency key)"
        )
        return res
    if not has_sec_tag(card):
        res.errors.append(
            f"card {card.id!r} is not SEC-tagged "
            f"(expected [SEC] in title or body)"
        )
        return res
    severity = extract_severity(card)
    if severity is None:
        res.errors.append(
            f"card {card.id!r} has no parseable severity "
            f"(expected [SEC][<SEVERITY>] in title or 'severity <SEVERITY>' "
            f"in body; severities ∈ {sorted(_SEVERITIES)})"
        )
        return res
    if template.severities and severity not in template.severities:
        res.errors.append(
            f"card {card.id!r} severity {severity!r} not in template's "
            f"allowed severities {list(template.severities)}"
        )
        return res
    res.fields["SEVERITY"] = severity

    vuln_id = extract_vuln_id(card)
    if vuln_id is None:
        res.errors.append(
            f"card {card.id!r} has no VULN-id (expected 'VULN-id: VULN-XX-NNN' "
            f"block or VULN-[A-Z][A-Z0-9]{{1,3}}-\\d{{3}} in title/body)"
        )
    else:
        res.fields["VULN_ID"] = vuln_id

    component = extract_component(card)
    if component is None:
        res.errors.append(
            f"card {card.id!r} has no Component (expected 'Component: …' "
            f"block or noun phrase after '(VULN-…)' in FULL CONTEXT)"
        )
    else:
        res.fields["COMPONENT"] = component

    evidence = extract_evidence(card)
    if not evidence:
        res.errors.append(
            f"card {card.id!r} has no evidence links (expected 'Evidence: "
            f"<url[, card-id]>' line; using the source card id is acceptable)"
        )
    else:
        res.fields["EVIDENCE_LINKS"] = ", ".join(evidence)

    res.fields["SOURCE_CARD_ID"] = card.id
    res.fields["SOURCE_TITLE"] = card.title or ""

    body = card.body or ""
    if not body.strip():
        res.errors.append(f"card {card.id!r} body is empty")
    else:
        res.fields["ISSUE_BODY"] = body

    if template.required_fields and "done_when" in template.required_fields:
        dw = extract_done_when_section(body)
        if not dw and template.done_when_section_required:
            res.errors.append(
                f"card {card.id!r} missing 'Done when:' section (required by template)"
            )

    # done_when derivation (warn-only if absent per spec §3.3 / §4.4 step 5)
    dw_section = extract_done_when_section(body)
    if dw_section:
        res.fields["DONE_WHEN_VERBATIM"] = dw_section
        mapped = classify_done_when_items(dw_section)
        res.done_when_mapped = mapped
        res.done_when_verbatim = dw_section
    else:
        res.fields["DONE_WHEN_VERBATIM"] = ""
        res.done_when_mapped = {s.key: [] for s in template.steps}

    # Assignee resolution — last so prior errors don't mask an unknown
    # assignee (and so the caller knows the resolved map regardless of
    # required-field errors above).
    profile_set = set(profiles)
    for step in template.steps:
        if step.key in assignees_resolved:
            continue
        primary = step.assignee
        fallback = step.assignee_fallback
        chosen = None
        if primary in profile_set:
            chosen = primary
            note = None
        elif fallback in profile_set:
            chosen = fallback
            note = (
                f"step {step.key!r}: primary assignee {primary!r} not found "
                f"on this host; using fallback {fallback!r}"
            )
        else:
            res.errors.append(
                f"step {step.key!r}: neither primary assignee {primary!r} "
                f"nor fallback {fallback!r} is a known profile (known: "
                f"{sorted(profile_set) or '∅'})"
            )
            continue
        assignees_resolved[step.key] = chosen
        if note:
            print(f"kanban workflow: {note}", file=sys.stderr)

    return res


def is_scout_card(card: kb.Task) -> bool:
    """Spec §3.1: any of (a) created_by='scout', (b) idempotency_key starts
    with 'scout:', (c) title starts with '[gh]'."""
    if (card.created_by or "").strip() == "scout":
        return True
    if (card.idempotency_key or "").startswith("scout:"):
        return True
    if (card.title or "").lstrip().startswith("[gh]"):
        return True
    return False


def has_sec_tag(card: kb.Task) -> bool:
    """Spec §3.2: '[SEC]' in title (case-insensitive) or body."""
    if "[SEC]" in (card.title or "").upper():
        return True
    if "[SEC]" in (card.body or "").upper():
        return True
    return False


def extract_severity(card: kb.Task) -> Optional[str]:
    """Return one of CRITICAL/HIGH/MEDIUM/LOW, or None.

    Spec §3.2:
    - Title: ``[SEC][<SEVERITY>]`` (case-insensitive).
    - Body: ``severity <SEVERITY>`` (case-insensitive) — substring match,
      not a full-line requirement, because scout cards embed the line in
      ``FULL CONTEXT: … severity HIGH. …``.
    Unknown severity values are NOT mapped to None — they fail validation
    against the template's allowed severities list downstream.
    """
    # Title pattern: [SEC][<SEVERITY>]
    m = re.search(r"\[SEC\]\s*\[([A-Za-z]+)\]", card.title or "", re.IGNORECASE)
    if m:
        sev = m.group(1).upper()
        if sev in _SEVERITIES:
            return sev
    # Body substring: "severity <SEVERITY>" surrounded by word/non-word edges
    m = re.search(r"(?i)\bseverity\s+([A-Z]+)\b", card.body or "")
    if m:
        sev = m.group(1).upper()
        if sev in _SEVERITIES:
            return sev
    return None


def extract_vuln_id(card: kb.Task) -> Optional[str]:
    """Spec §3.3: structured block, else regex on title or body."""
    m = re.search(r"(?im)^\s*VULN-id\s*[:=]\s*(VULN-[A-Z][A-Z0-9]{1,3}-\d{3})\s*$",
                  card.body or "")
    if m:
        return m.group(1)
    for source in (card.title, card.body):
        if not source:
            continue
        m = _VULN_ID_RE.search(source)
        if m:
            return m.group(0)
    return None


def extract_component(card: kb.Task) -> Optional[str]:
    """Spec §3.3: structured block, else noun phrase from FULL CONTEXT.

    The FULL CONTEXT clause in scout cards spans multiple lines (the
    title often says "Example vuln in the widget service" while the
    body continues with "in the widget service, severity HIGH."). The
    fallback therefore folds the FULL CONTEXT block (from the line
    starting ``FULL CONTEXT:`` through the next blank line or the
    canonical next section header) before extracting the noun phrase
    after ``(VULN-XX-NNN)``.
    """
    m = re.search(r"(?im)^\s*Component\s*[:=]\s*(.+?)\s*$", card.body or "")
    if m:
        return m.group(1).strip()
    # Fallback: gather the FULL CONTEXT block (single-line OR multi-line)
    block = _extract_full_context_block(card.body or "")
    if block:
        m = re.search(
            r"\(VULN-[A-Z][A-Z0-9]{1,3}-\d{3}\)\s*[—-]?\s*(.+)",
            block,
            re.DOTALL,
        )
        if m:
            phrase = m.group(1).strip()
            # Strip trailing description ("... severity HIGH. Test fixture …")
            # down to a reasonable noun phrase: take everything before the
            # first ", severity" / "." / "—" boundary after the head words.
            phrase = re.split(
                r",\s*severity|\.\s*[A-Z]|\s+—\s+",
                phrase,
                maxsplit=1,
            )[0].strip()
            if phrase:
                return phrase
    return None


def _extract_full_context_block(body: str) -> str:
    """Return the FULL CONTEXT block, folding line continuations.

    A scout card body looks like::

        FULL CONTEXT: veroscale-services#999 (VULN-TST-001) — example
        vulnerability in the widget service, severity HIGH. Test fixture.

    The clause after ``(VULN-…)`` is the affected service/path. We collapse
    the whole block into one line so the regex in ``extract_component`` can
    see it.
    """
    m = re.search(
        r"(?im)^\s*FULL CONTEXT\s*:?\s*(.+?)(?=^\s*\n|\Z)",
        body,
        re.DOTALL,
    )
    if not m:
        return ""
    # Collapse whitespace (including embedded newlines) into single spaces.
    return re.sub(r"\s+", " ", m.group(1)).strip()


def extract_evidence(card: kb.Task) -> list[str]:
    """Spec §3.3: structured block, else Evidence: <url> line, else source id."""
    m = re.search(r"(?im)^\s*Evidence\s*:\s*(.+?)\s*$", card.body or "")
    if m:
        raw = m.group(1)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts if parts else [card.id]
    # Fall back to any URL in body or title
    urls = re.findall(r"https?://\S+", (card.body or "") + "\n" + (card.title or ""))
    if urls:
        return urls
    return [card.id]


def extract_done_when_section(body: str) -> str:
    """Spec §4.4 step 1: from 'Done when:' to next blank line or Idempotency:."""
    m = _DONE_WHEN_RE.search(body or "")
    if not m:
        return ""
    return m.group(1).rstrip()


# done_when classification rules per spec §4.4 step 2.
_DONE_WHEN_RULES: tuple[tuple[str, str], ...] = (
    (
        "ship-pr",
        r"\b(PR|pull request)\b.*\b(open|merged|merge)\b|"
        r"\bmerged? to (main|master)\b|"
        r"\b(coord|coordinate).*\bSab\b|"
        r"\birreversible\b|\bdeploy\b",
    ),
    (
        "regression-test",
        r"\bregression test\b|\btest asserts\b|"
        r"\btests? (pass|fail|added|written)\b|"
        r"\bfailure mode.*cannot be reproduced\b|"
        r"\b(reproduce|reproduction).*\bfixed\b",
    ),
    (
        "corpus-recon",
        r"\b(recon|scoping|corpus|survey|investigate|map)\b|"
        r"\bcite\b|\bscope check\b|\bconfirm which\b",
    ),
)


def classify_done_when_items(section: str) -> dict[str, list[str]]:
    """Return {step_key: [item, ...]} with the default bucket = repro-patch.

    The spec writes ``(1) …, (2) …, (3) …`` on one logical line, but scout
    bodies often wrap an item onto a continuation line, AND items end with
    a trailing comma (the next ``(N)`` marker starts the next item). We
    split on top-level ``(N)`` markers first (the canonical numbering),
    then drop stray embedded newlines and trailing commas.
    """
    # Split on top-level numbered markers. The lookahead keeps the
    # marker with the item that follows it.
    parts = re.split(r"(?=\(\d+\)\s)", section)
    out: dict[str, list[str]] = {
        "ship-pr": [], "regression-test": [], "corpus-recon": [], "repro-patch": [],
    }
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # The split above keeps the ``(N)`` marker with each item, so the
        # part starts with ``(``. ``\(?`` makes the leading paren optional
        # so the same regex matches both ``(1) foo`` and ``1) foo``.
        m = re.match(r"\s*\(?(\d+)\)\s+(.+)", part, re.DOTALL)
        if m:
            num, text = m.group(1), m.group(2)
            # Collapse embedded newlines + drop trailing comma/period so a
            # wrapped item still classifies. The trailing punctuation is
            # cosmetic only.
            text = re.sub(r"\s+", " ", text).strip().rstrip(",.;")
            line = f"({num}) {text}"
        else:
            m = _DASHED_ITEM_RE.match(part)
            if not m:
                continue
            text = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(",.;")
            line = f"- {text}"
        assigned = False
        for key, pattern in _DONE_WHEN_RULES:
            if re.search(pattern, text, re.IGNORECASE):
                out[key].append(line)
                assigned = True
                break
        if not assigned:
            out["repro-patch"].append(line)
    return out


# ---------------------------------------------------------------------------
# Chain instantiation
# ---------------------------------------------------------------------------

@dataclass
class PlannedChild:
    step_key: str
    title: str
    body: str
    assignee: str
    parents: list[str]
    priority: int
    gate: str  # "auto" | "approval"
    idempotency_key: str
    workflow_template_id: str
    initial_status: str
    ship_block_kind: Optional[str] = None
    ship_block_reason: Optional[str] = None


@dataclass
class ChainPlan:
    template: WorkflowTemplate
    source_card_id: str
    children: list[PlannedChild]
    existing_chain: list[str] = field(default_factory=list)

    @property
    def ship_child(self) -> Optional[PlannedChild]:
        for c in self.children:
            if c.gate == "approval":
                return c
        return None


_SEVERITY_PRIORITY = {"CRITICAL": 100, "HIGH": 80, "MEDIUM": 60, "LOW": 40}


def build_chain_plan(
    card: kb.Task,
    template: WorkflowTemplate,
    validation: ValidationResult,
    *,
    assignee_overrides: dict[str, str],
    assignees_resolved: dict[str, str],
) -> ChainPlan:
    """Pure plan builder — no DB writes. Caller runs the writes."""
    fields = dict(validation.fields)
    fields.setdefault("TEMPLATE_ID", template.template_id)
    priority = _SEVERITY_PRIORITY.get(fields.get("SEVERITY", ""), 0)
    # For the ship card's CHAIN_IDS we need step keys; we substitute the
    # previous step's *real* id after each create, so the placeholder is
    # resolved below in instantiate_chain.
    children: list[PlannedChild] = []
    prev_id_holder: list[str] = [card.id]  # step 1's previous = source
    for idx, step in enumerate(template.steps, start=1):
        assignee = assignee_overrides.get(step.key) or assignees_resolved.get(step.key, "")
        if not assignee:
            raise WorkflowValidationError(
                f"step {step.key!r}: no assignee resolved "
                f"(override or template profile missing)"
            )
        # Title
        title = step.title.replace("{{VULN_ID}}", fields.get("VULN_ID", "?"))
        # Body — start with provenance block
        prev_id = prev_id_holder[-1] if prev_id_holder else card.id
        chain_ids_placeholder = (
            "\n".join(
                f"- step `{s.key}`: {prev_id_holder[i] if i < len(prev_id_holder) else '?'}"
                for i, s in enumerate(template.steps[:idx])
            )
            if step.gate == "approval"
            else ""
        )
        done_when_mapped_lines = _format_done_when_mapped(
            validation.done_when_mapped.get(step.key, []),
        )
        body = _render_step_body(
            step,
            fields={
                **fields,
                "STEP_KEY": step.key,
                "STEP_N": str(idx),
                "PREV_STEP_ID": prev_id,
                "CHAIN_IDS": chain_ids_placeholder,
                "DONE_WHEN_MAPPED": done_when_mapped_lines,
            },
        )
        idem = (
            f"sec-vuln-remediation:{fields.get('VULN_ID', '?')}:{step.key}"
            if template.template_id == "sec-vuln-remediation"
            else f"{template.template_id}:{fields.get('VULN_ID', '?')}:{step.key}"
        )
        # Approval-gated ship card is sticky-blocked; created in `running` so
        # block_task can fire (block_task requires running/ready, spec §5.2).
        is_ship = step.gate == "approval"
        initial_status = "running" if is_ship else "running"
        planned = PlannedChild(
            step_key=step.key,
            title=title,
            body=body,
            assignee=assignee,
            parents=[card.id] + ([prev_id] if idx > 1 else []),
            priority=priority,
            gate=step.gate,
            idempotency_key=idem,
            workflow_template_id=template.template_id,
            initial_status=initial_status,
            ship_block_kind=("needs_input" if is_ship else None),
            ship_block_reason=(
                _ship_gate_text(fields.get("VULN_ID", "?")) if is_ship else None
            ),
        )
        children.append(planned)
        prev_id_holder.append("__PENDING__")  # real id wired in instantiate_chain
    return ChainPlan(
        template=template,
        source_card_id=card.id,
        children=children,
    )


def _ship_gate_text(vuln_id: str) -> str:
    return (
        f"APPROVAL GATE (sec-vuln-remediation) — {vuln_id}. Steps 1-3 of the "
        f"chain must complete before this card may run. Approving "
        f"(hermes kanban unblock <id>) authorizes ONLY: opening a scoped PR "
        f"against origin/main with evidence (gates 1-4 in the body). "
        f"Auto-merge and auto-deploy are FORBIDDEN; merge and deploy remain "
        f"Sab actions outside this card."
    )


_STEP_BODIES: dict[str, str] = {
    "corpus-recon": """## Scope
CORPUS-FIRST: search the corpus (cfd signals, repo tests/specs/docs, sibling
worktrees/branches) BEFORE any code. Survey the affected component named in
the issue. Extend existing patterns — never introduce a parallel layer.

## Done when
1. Recon report published as a comment on THIS card (or attached artifact):
   file paths + line refs, trust model, exposure verdict
   (confirmed / not-reproducible / already-fixed), patterns to extend,
   signals cited.
2. No code changes in this step (report only).
3. Any human-only actions discovered (DNS, Workspace/admin UI, irreversible
   ops) are listed for the patch step — they become their own blocked
   [Sab action] cards per gate G2, never folded into an auto step.
4. {{DONE_WHEN_MAPPED}}

## Issue body (verbatim)
{{ISSUE_BODY}}

## Issue done-when (verbatim)
{{DONE_WHEN_VERBATIM}}
""",
    "repro-patch": """## Scope
Reproduce the failure mode from the issue (or cite recon's repro evidence),
then apply the minimal fix. Branch from origin/main; the diff must stay
inside this VULN-id's scope (gate G4). Extend existing helpers — do not bolt
on parallel machinery. If the fix requires a design decision: STOP, ship a
design-doc PR with numbered Sab gates, and block for approval (gate G1).

## Done when
1. Failure mode reproduced against the fixed code, or recon's repro
   evidence cited.
2. Minimal-diff fix applied; tests added/extended (red-green proof lands in
   step 3).
3. No human-only action performed here — any discovered [Sab action] was
   split to its own blocked card (gate G2).
4. {{DONE_WHEN_MAPPED}}

## Issue body (verbatim)
{{ISSUE_BODY}}

## Issue done-when (verbatim)
{{DONE_WHEN_VERBATIM}}
""",
    "regression-test": """## Scope
Regression tests for the fix. Corpus-first: extend the existing test surface
(no new test framework, no parallel suite). Prove the tests are load-bearing.

## Done when
1. Tests added to the existing suite; actual output pasted in the card.
2. Red-green proven: reverting the fix fails the new tests; fix applied →
   full suite passes.
3. {{DONE_WHEN_MAPPED}}

## Issue body (verbatim)
{{ISSUE_BODY}}

## Issue done-when (verbatim)
{{DONE_WHEN_VERBATIM}}
""",
    "ship-pr": """## Scope
APPROVAL-GATED STEP. You are dispatched only after Sab approved via
`hermes kanban unblock <this-card>`. Open ONE PR against origin/main with
the fix + regression tests; verify mergeability and CI; post evidence.
Do NOT merge. Do NOT deploy. Merge and deploy are Sab actions.

## Sab gates (all must hold; post evidence for each)
- gate_1: PR open against origin/main, NOT merged, NOT auto-merged
- gate_2: PR diff scoped to {{VULN_ID}} only — no sibling/unrelated work
- gate_3: test output pasted verbatim; typecheck/lint clean
- gate_4: no auto-deploy artifact (no deploy hook fired, no Kamal/CF push)

## Done when
1. Gates 1–4 verified and evidenced as comments on this card.
2. {{DONE_WHEN_MAPPED}} (items mentioning PR-merged/coord-with-Sab are
   verified AFTER Sab merges — see below).
3. After Sab merges: re-verify merged state via the gh API, comment the
   merge commit + close/comment the issue, then complete this card.

## Chain
{{CHAIN_IDS}}

## Issue body (verbatim)
{{ISSUE_BODY}}

## Issue done-when (verbatim)
{{DONE_WHEN_VERBATIM}}
""",
}

_PROVENANCE_BLOCK = """<!-- AUTO-SPAWNED by hermes kanban workflow {{TEMPLATE_ID}} v{{TEMPLATE_VERSION}} — do not edit the provenance block -->
## Source issue
- VULN-id: {{VULN_ID}}
- Severity: {{SEVERITY}}
- Affected component: {{COMPONENT}}
- Source card: {{SOURCE_CARD_ID}} — "{{SOURCE_TITLE}}"
- Evidence: {{EVIDENCE_LINKS}}
- Chain step: {{STEP_KEY}} ({{STEP_N}}/4); previous step: {{PREV_STEP_ID}}
"""


def _render_step_body(step: TemplateStep, *, fields: dict[str, str]) -> str:
    template_body = _STEP_BODIES.get(step.key)
    if template_body is None:
        # Unknown step — fall back to the generic ship-pr body so a
        # non-canonical template still produces a usable card.
        template_body = _STEP_BODIES["ship-pr"]
    # Inject version + template_id defaults that callers may have omitted.
    fields = dict(fields)
    fields.setdefault("TEMPLATE_VERSION", "1.0.0")
    provenance = _render_template(_PROVENANCE_BLOCK, fields)
    return provenance + "\n" + _render_template(template_body, fields)


def _render_template(text: str, fields: dict[str, str]) -> str:
    out = text
    for key, value in fields.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def _render_child_body(
    planned: PlannedChild,
    template: WorkflowTemplate,
    validation: ValidationResult,
    step_idx: int,
    prev_ids: Sequence[str],
) -> str:
    """Render a child body with the given previous-step ids.

    Used by BOTH the dry-run pass (sentinel ids) and the real-write
    loop (real sibling ids) so the two paths produce identical body
    shapes. The real-write loop MUST call this after each create —
    otherwise children 2-4 leak ``<id-of-step-N-would-be-created>``
    into the provenance ``previous step:`` line and the ``## Chain``
    section (the JSON-output placeholder fix missed the body path).
    """
    chain_ids = ""
    if planned.gate == "approval":
        chain_ids = "\n".join(
            f"- step `{template.steps[i].key}`: {prev_ids[i]}"
            for i in range(len(prev_ids))
        )
    return _render_step_body(
        template.step(planned.step_key),
        fields={
            **validation.fields,
            "TEMPLATE_ID": template.template_id,
            "STEP_KEY": planned.step_key,
            "STEP_N": str(step_idx),
            "PREV_STEP_ID": prev_ids[-1] if prev_ids else "",
            "CHAIN_IDS": chain_ids,
            "DONE_WHEN_MAPPED": _format_done_when_mapped(
                validation.done_when_mapped.get(planned.step_key, []),
            ),
        },
    )


def _format_done_when_mapped(items: list[str]) -> str:
    if not items:
        return (
            "No issue done-when items map to this step; static acceptance above applies."
        )
    return "\n".join(f"- {it}" for it in items)


# ---------------------------------------------------------------------------
# Pre-flight (chain already exists?)
# ---------------------------------------------------------------------------

def find_existing_chain_ids(
    conn, *, source_card_id: str, template_id: str,
) -> list[str]:
    """Spec §6.1: tasks linked from the source with the template set."""
    rows = conn.execute(
        """
        SELECT t.id
          FROM tasks t
          JOIN task_links l ON l.child_id = t.id
         WHERE l.parent_id = ?
           AND t.workflow_template_id = ?
           AND t.status != 'archived'
         ORDER BY t.created_at
        """,
        (source_card_id, template_id),
    ).fetchall()
    return [r["id"] for r in rows]


# ---------------------------------------------------------------------------
# Sticky-block writer (create + block + parent link in one txn)
# ---------------------------------------------------------------------------

def _append_event(
    conn, task_id: str, kind: str, payload: dict[str, Any], *,
    run_id: Optional[int] = None,
) -> None:
    """Thin wrapper around kanban_db._append_event, exposed for the
    workflow writer so it can emit a 'workflow_spawned' event with the
    chain manifest. Falls back gracefully if the private helper changes.
    """
    import json as _json
    fn = getattr(kb, "_append_event", None)
    if fn is None:
        return
    try:
        fn(conn, task_id, kind, payload, run_id=run_id)
    except TypeError:
        # Older signature (no run_id)
        fn(conn, task_id, kind, payload)


def _fire_kanban_lifecycle_hook(event: str, task_id: str, **kwargs) -> None:
    fn = getattr(kb, "_fire_kanban_lifecycle_hook", None)
    if fn is None:
        return
    try:
        fn(event, task_id, **kwargs)
    except Exception:
        # Hooks are advisory; never fail the writer on a hook error.
        pass


def _sticky_block_in_txn(
    conn,
    *,
    task_id: str,
    kind: str,
    reason: str,
) -> None:
    """Apply a sticky block INSIDE an already-open write_txn.

    Mirrors the SQL path of :func:`kanban_db.block_task` (running/ready ->
    blocked + blocked event row) but without opening a nested transaction
    — the caller already holds ``write_txn(conn, allow_nested=True)`` and
    ``block_task`` would raise on its inner ``write_txn`` boundary. We do
    NOT call ``block_task`` so the spec's "create + block + link in one
    atomic txn" guarantee holds end-to-end (spec §5.2).
    """
    cur = conn.execute(
        "SELECT status, block_kind, block_recurrences FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if cur is None:
        raise WorkflowValidationError(
            f"sticky_block: task {task_id} vanished between create and block"
        )
    if cur["status"] not in ("running", "ready"):
        raise WorkflowValidationError(
            f"sticky_block: task {task_id} not in running/ready "
            f"(status={cur['status']!r}) — spec §5.2 requires create+block"
        )
    prev_recurrences = (
        int(cur["block_recurrences"]) if cur["block_recurrences"] is not None else 0
    )
    # First-time sticky block; a re-fire would trip block-recurrence accounting.
    recurrences = 1
    conn.execute(
        """
        UPDATE tasks
           SET status            = 'blocked',
               claim_lock        = NULL,
               claim_expires     = NULL,
               worker_pid        = NULL,
               block_kind        = ?,
               block_recurrences = ?
         WHERE id = ?
           AND status IN ('running', 'ready')
        """,
        (kind, recurrences, task_id),
    )
    # Synthesize an ended run so the reason is preserved in attempt history.
    kb._synthesize_ended_run(
        conn, task_id, outcome="blocked", summary=reason,
    )
    kb._append_event(
        conn, task_id, "blocked",
        {"reason": reason, "kind": kind, "recurrences": recurrences,
         "source_status": "ready"},
    )


def create_child_with_sticky_block(
    conn,
    *,
    planned: PlannedChild,
    real_parent_ids: list[str],
) -> str:
    """Spec §5.2: create + sticky-block + parent-link in one atomic txn.

    Returns the new task id. Auto-dispatch steps use the regular
    ``create_task`` path (they don't need the sticky block).
    """
    from hermes_cli.kanban_db import create_task

    # Spec §5.2 requires create + block + parent-link in ONE transaction.
    # ``create_task`` opens its own write_txn(allow_nested=True) when called
    # inside an outer scope, and our sticky-block helper operates on the
    # already-open connection. The outer txn is what makes the whole
    # chain-creation atomic against a mid-write crash.
    #
    # IMPORTANT: when sticky-blocking, pass parents=() to create_task.
    # create_task auto-demotes a "running" / "ready" card to "todo" when
    # ANY parent isn't done yet, which would break block_task()'s precondition
    # (it only fires from running/ready). Spec §5.2 step 1 explicitly
    # specifies parents=[] for this reason; we then INSERT the link rows
    # ourselves below (spec §5.2 step 3).
    #
    # Idempotency: create_task returns the existing id when idempotency_key
    # matches a non-archived row. Under --force re-runs, the existing ship
    # card is already sticky-blocked — do NOT re-fire the block (spec §6.2:
    # "the ship card's sticky block is applied ONLY if the reused ship card
    # is not already blocked"). Skip the block in that case.
    parents_for_create: tuple = tuple(real_parent_ids) if not planned.ship_block_kind else ()
    with kb.write_txn(conn):
        task_id = create_task(
            conn,
            title=planned.title,
            body=planned.body,
            assignee=planned.assignee,
            created_by="workflow:" + planned.workflow_template_id,
            workspace_kind="scratch",
            priority=planned.priority,
            parents=parents_for_create,
            idempotency_key=planned.idempotency_key,
            initial_status=planned.initial_status,
            workflow_template_id=planned.workflow_template_id,
            current_step_key=planned.step_key,
        )
        if planned.ship_block_kind:
            # Look up the existing card's status. If it's already sticky-blocked
            # (a --force re-run finding the ship card from the prior chain),
            # skip the block — never re-fire sticky_block on an already-stuck
            # card (would trip block-recurrence accounting).
            existing = conn.execute(
                "SELECT status, block_kind FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if not (existing and existing["status"] == "blocked"
                    and existing["block_kind"] == planned.ship_block_kind):
                _sticky_block_in_txn(
                    conn,
                    task_id=task_id,
                    kind=planned.ship_block_kind,
                    reason=planned.ship_block_reason or "",
                )
        # Wire parent links explicitly so the order is identical to spec §5.2
        # ("insert task_links rows … directly in the same transaction").
        # create_task already inserted links via parents=, but the spec is
        # explicit about doing it after the block — belt + braces.
        for pid in real_parent_ids:
            conn.execute(
                "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
                (pid, task_id),
            )
        _append_event(
            conn, task_id, "workflow_spawned",
            {
                "template_id": planned.workflow_template_id,
                "step_key": planned.step_key,
                "gate": planned.gate,
                "parents": real_parent_ids,
                "idempotency_key": planned.idempotency_key,
            },
        )
    _fire_kanban_lifecycle_hook(
        "kanban_task_created", task_id,
        board=kb.get_current_board(),
        assignee=planned.assignee,
        reason=f"workflow:{planned.workflow_template_id}:{planned.step_key}",
    )
    return task_id


# ---------------------------------------------------------------------------
# Validation-error class (CLI surface)
# ---------------------------------------------------------------------------

class WorkflowValidationError(Exception):
    """Raised when validation fails before any write occurs."""


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def _parse_assignee_overrides(raw: Optional[Iterable[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw or ():
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--assignee expects KEY=PROFILE (got {item!r})"
            )
        key, _, profile = item.partition("=")
        key = key.strip()
        profile = profile.strip()
        if not key or not profile:
            raise argparse.ArgumentTypeError(
                f"--assignee expects KEY=PROFILE (got {item!r})"
            )
        out[key] = profile
    return out


def _known_assignees(conn) -> list[str]:
    """Union of on-disk profiles + every assignee ever seen on this board."""
    disk = set(kb.list_profiles_on_disk())
    rows = conn.execute(
        "SELECT DISTINCT assignee FROM tasks WHERE assignee IS NOT NULL",
    ).fetchall()
    seen = {r["assignee"] for r in rows if r["assignee"]}
    return sorted(disk | seen)


def cmd_workflow(args: argparse.Namespace) -> int:
    """``hermes kanban workflow …`` argparse handler.

    Dispatch (spec §7.1):

    * positional[0] == ``list`` → enumerate templates
    * positional[0] == ``show`` → print a single template (positional[1])
    * otherwise → instantiate ``positional[0]`` against ``positional[1]``
      (card ref), with ``--dry-run`` / ``--force`` / ``--assignee`` /
      ``--json`` honoured.
    """
    positionals = list(getattr(args, "positional", None) or [])
    action = positionals[0] if positionals else None
    # Backwards compat: legacy ``workflow_action`` (subparsers) is no longer
    # attached, but if an older caller routed here via the subparsers path
    # honour it.
    legacy_action = getattr(args, "workflow_action", None)
    if action is None:
        action = legacy_action
    if action in (None, "list"):
        # If the caller used ``workflow list`` with no other args, list.
        # If they passed ``list`` AND something else, it's a usage error.
        if action == "list" and len(positionals) > 1:
            print(
                "kanban workflow: `list` takes no arguments "
                f"(got: {' '.join(positionals[1:])})",
                file=sys.stderr,
            )
            return 2
        return _cmd_workflow_list(args)
    if action == "show":
        if len(positionals) < 2:
            print(
                "kanban workflow: `show` requires a template id "
                "(e.g. `hermes kanban workflow show sec-vuln-remediation`)",
                file=sys.stderr,
            )
            return 2
        if len(positionals) > 2:
            print(
                "kanban workflow: `show` takes exactly one template id "
                f"(got: {' '.join(positionals[1:])})",
                file=sys.stderr,
            )
            return 2
        args.template = positionals[1]
        return _cmd_workflow_show(args)
    # Otherwise: positional[0] is the template id, positional[1] (optional)
    # is the card ref. Empty card_ref + non-list/show → usage error.
    if len(positionals) == 1:
        print(
            "kanban workflow: usage: hermes kanban workflow <template> "
            "<card-ref> [--dry-run] [--force] [--assignee KEY=PROFILE]... "
            "[--json]\n"
            "             (no card ref supplied; for `list` / `show` "
            "use the matching keyword)",
            file=sys.stderr,
        )
        return 2
    if len(positionals) > 2:
        print(
            "kanban workflow: too many positional arguments: "
            f"{' '.join(positionals)}",
            file=sys.stderr,
        )
        return 2
    args.template = positionals[0]
    args.card_ref = positionals[1]
    return _cmd_workflow_run(args)


def _cmd_workflow_list(args: argparse.Namespace) -> int:
    templates = list_templates()
    if getattr(args, "json", False):
        print(json.dumps([
            {
                "template_id": t.template_id,
                "version": t.version,
                "status": t.status,
                "steps": [
                    {"key": s.key, "gate": s.gate, "assignee": s.assignee,
                     "fallback": s.assignee_fallback}
                    for s in t.steps
                ],
                "path": str(t.path),
            }
            for t in templates
        ], indent=2, ensure_ascii=False))
        return 0
    if not templates:
        print("(no workflow templates found in {})".format(_templates_dir()),
              file=sys.stderr)
        return 0
    print(f"{'TEMPLATE_ID':<32}  {'VERSION':<10}  STEPS")
    for t in templates:
        steps = ", ".join(
            f"{s.key}({s.gate})" for s in t.steps
        )
        print(f"{t.template_id:<32}  {t.version:<10}  {steps}")
    return 0


def _cmd_workflow_show(args: argparse.Namespace) -> int:
    template_id = args.template
    try:
        template = load_template(template_id)
    except WorkflowTemplateError as exc:
        print(f"kanban workflow: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps({
            "template_id": template.template_id,
            "version": template.version,
            "status": template.status,
            "spec_file": template.spec_file,
            "input": {
                "role": template.input_role,
                "severity_tag": template.severity_tag,
                "severities": list(template.severities),
                "required_fields": list(template.required_fields),
                "done_when_section_required": template.done_when_section_required,
            },
            "steps": [
                {"key": s.key, "title": s.title, "assignee": s.assignee,
                 "fallback": s.assignee_fallback, "gate": s.gate}
                for s in template.steps
            ],
            "placeholders": list(template.placeholders),
            "ship_step_key": template.ship_step_key,
            "path": str(template.path),
        }, indent=2, ensure_ascii=False))
        return 0
    print(f"Template:    {template.template_id}  (version {template.version})")
    print(f"Status:      {template.status}")
    print(f"Spec file:   {template.spec_file or template.path}")
    print(f"Input role:  {template.input_role}  | severity tag: {template.severity_tag}")
    print(f"Severities:  {', '.join(template.severities) or '(none)'}")
    print(f"Required:    {', '.join(template.required_fields) or '(none)'}")
    print(f"Ship step:   {template.ship_step_key}")
    print(f"Steps:")
    for s in template.steps:
        print(f"  - {s.key:<18} gate={s.gate:<9} assignee={s.assignee}"
              f"{'' if not s.assignee_fallback else f' (fallback: {s.assignee_fallback})'}")
    return 0


def _cmd_workflow_run(args: argparse.Namespace) -> int:
    template_id = getattr(args, "template", None)
    card_ref = getattr(args, "card_ref", None)
    if not template_id or not card_ref:
        print(
            "kanban workflow: usage: hermes kanban workflow <template> <card-ref> "
            "[--dry-run] [--force] [--assignee KEY=PROFILE]... [--json]",
            file=sys.stderr,
        )
        return 2
    try:
        template = load_template(template_id)
    except WorkflowTemplateError as exc:
        print(f"kanban workflow: {exc}", file=sys.stderr)
        return 2
    try:
        overrides = _parse_assignee_overrides(getattr(args, "assignee", None))
    except argparse.ArgumentTypeError as exc:
        print(f"kanban workflow: {exc}", file=sys.stderr)
        return 2
    dry_run = bool(getattr(args, "dry_run", False))
    force = bool(getattr(args, "force", False))

    with kb.connect_closing() as conn:
        profiles = _known_assignees(conn)
        # Source card
        try:
            source_card = resolve_source_card(card_ref, conn)
        except WorkflowValidationError as exc:
            print(f"kanban workflow: {exc}", file=sys.stderr)
            return 2

        # Pre-flight: existing chain?
        existing_ids = find_existing_chain_ids(
            conn,
            source_card_id=source_card.id,
            template_id=template.template_id,
        )
        if existing_ids and not force:
            if getattr(args, "json", False):
                print(json.dumps({
                    "template_id": template.template_id,
                    "source_card": source_card.id,
                    "existing_chain": existing_ids,
                    "no_op": True,
                }, indent=2, ensure_ascii=False))
            else:
                print(f"chain already exists: {', '.join(existing_ids)}")
                print(
                    "(re-run with --force to bypass pre-flight; not a gate bypass)",
                    file=sys.stderr,
                )
            return 0

        # Validate assignee overrides against known profiles
        for key, profile in overrides.items():
            if profile not in set(profiles):
                print(
                    f"kanban workflow: --assignee {key}={profile!r}: unknown "
                    f"profile (known: {profiles or '∅'})",
                    file=sys.stderr,
                )
                return 2

        # Run validation (writes nothing)
        assignees_resolved: dict[str, str] = {}
        validation = validate_source(
            source_card, template,
            profiles=profiles,
            assignees_resolved=assignees_resolved,
        )
        # Apply overrides on top of resolution
        assignees_resolved.update(
            {k: v for k, v in overrides.items() if k in {s.key for s in template.steps}}
        )
        if not validation.ok:
            for err in validation.errors:
                print(f"kanban workflow: {err}", file=sys.stderr)
            return 2

        # Build the plan (still pure)
        try:
            plan = build_chain_plan(
                source_card, template, validation,
                assignee_overrides=overrides,
                assignees_resolved=assignees_resolved,
            )
        except WorkflowValidationError as exc:
            print(f"kanban workflow: {exc}", file=sys.stderr)
            return 2

        # Substitutes real previous-step ids in each subsequent step's body
        # AND in planned.parents (so dry-run output shows real parent ids,
        # not the "__PENDING__" placeholders build_chain_plan filled in).
        # Without this fix, both `--dry-run` (human + JSON) and the real-run
        # JSON output report __PENDING__ for steps 2-4.
        #
        # We also grow prev_ids with each step in the dry-run path so the
        # PREV_STEP_ID + CHAIN_IDS substitutions in later steps reflect the
        # would-be ids in order. The real-write loop below does the same
        # thing with actual ids — so the two paths produce identical
        # parent/body shapes, which is what the spec promises.
        prev_ids: list[str] = [source_card.id]
        for planned in plan.children:
            step_idx = next(
                i for i, s in enumerate(template.steps, start=1)
                if s.key == planned.step_key
            )
            # Rebuild body with substituted prev ids (sentinels in the
            # dry-run pass; the real-write loop re-renders with real ids).
            planned.body = _render_child_body(
                planned, template, validation, step_idx, prev_ids,
            )
            # Patch planned.parents: step 1 = [source]; step N (N>=2)
            # = [source, prev_step]. In the dry-run path we don't have the
            # real previous-step ids yet, so emit a clearly-placeholder
            # marker that an operator can recognize. The real-write loop
            # below overwrites planned.parents with the actual ids before
            # each create call so production cards always get real ids.
            if step_idx == 1:
                planned.parents = [source_card.id]
            else:
                planned.parents = [
                    source_card.id,
                    f"<id-of-step-{step_idx - 1}-would-be-created>",
                ]
            # Track the would-be prev-ids locally so the body substitutions
            # below stay consistent with planned.parents. The real-write
            # loop replaces these with actual ids as it goes.
            while len(prev_ids) <= step_idx:
                prev_ids.append(f"<id-of-step-{len(prev_ids)}-would-be-created>")

        if dry_run:
            return _render_plan(plan, validation, source_card,
                                profiles=profiles, dry_run=True,
                                json_out=getattr(args, "json", False))

        # Real run — create children in order. Each previous step's REAL id
        # is recorded so step N+1's parents and CHAIN_IDS use the true ids.
        # We also patch planned.parents with real ids so the JSON/human
        # output at the end reports real parents (not the dry-run sentinels).
        created: list[tuple[PlannedChild, str]] = []
        for planned in plan.children:
            step_idx = next(
                i for i, s in enumerate(template.steps, start=1)
                if s.key == planned.step_key
            )
            parent_ids = [source_card.id]
            if step_idx > 1:
                parent_ids.append(prev_ids[step_idx - 1])  # previous child id
            # Patch planned.parents + prev_ids bookkeeping so the rendered
            # plan at the bottom reports real ids (not dry-run sentinels).
            planned.parents = list(parent_ids)
            # Re-render the body with REAL previous-step ids — the dry-run
            # pass above filled sentinels, and production bodies must not
            # leak <id-of-step-N-would-be-created> into the provenance
            # "previous step:" line or the "## Chain" section.
            planned.body = _render_child_body(
                planned, template, validation, step_idx, prev_ids[:step_idx],
            )
            try:
                tid = create_child_with_sticky_block(
                    conn,
                    planned=planned,
                    real_parent_ids=parent_ids,
                )
            except Exception as exc:
                # Partial chain: report what landed + what didn't, exit 1
                print(
                    f"kanban workflow: step {planned.step_key!r} failed: {exc}",
                    file=sys.stderr,
                )
                if created:
                    print(
                        "kanban workflow: partial chain (created so far):",
                        file=sys.stderr,
                    )
                    for p, cid in created:
                        print(
                            f"  {cid}  {p.title}  assignee={p.assignee}  gate={p.gate}",
                            file=sys.stderr,
                        )
                return 1
            created.append((planned, tid))
            # The list grows so subsequent steps can wire prev_id by index.
            while len(prev_ids) < step_idx + 1:
                prev_ids.append(tid)
            prev_ids[step_idx] = tid
        # Recompute once after the final commit so any to-do children that
        # are now parent-gated can promote. Per spec §5.2 this is what
        # _landing_status_after_parents would do; recompute_ready is the
        # canonical entry point.
        kb.recompute_ready(conn)

    # Final rendering
    if getattr(args, "json", False):
        return _render_plan(plan, validation, source_card,
                            profiles=profiles, dry_run=False,
                            json_out=True, created=created)
    return _render_plan(plan, validation, source_card,
                        profiles=profiles, dry_run=False,
                        json_out=False, created=created)


def _render_plan(
    plan: ChainPlan,
    validation: ValidationResult,
    source_card: kb.Task,
    *,
    profiles: Sequence[str],
    dry_run: bool,
    json_out: bool,
    created: Optional[list[tuple[PlannedChild, str]]] = None,
) -> int:
    ship = plan.ship_child
    ship_id = (
        next((cid for p, cid in (created or []) if p.gate == "approval"), None)
        if created else None
    )
    if json_out:
        payload: dict[str, Any] = {
            "template_id": plan.template.template_id,
            "version": plan.template.version,
            "source_card": source_card.id,
            "source_title": source_card.title,
            "validated": {
                "vuln_id": validation.fields.get("VULN_ID"),
                "severity": validation.fields.get("SEVERITY"),
                "component": validation.fields.get("COMPONENT"),
                "evidence_links": validation.fields.get("EVIDENCE_LINKS"),
                "assignees_resolved": {
                    s.key: (next((cid for p, cid in (created or []) if p.step_key == s.key), None) or s.assignee)
                    for s in plan.template.steps
                },
            },
            "children": [
                {
                    "step": p.step_key,
                    "id": (next((cid for cp, cid in (created or []) if cp.step_key == p.step_key), None)
                           if created else None),
                    "title": p.title,
                    "assignee": p.assignee,
                    "parents": p.parents,
                    "gate": p.gate,
                    "idempotency_key": p.idempotency_key,
                }
                for p in plan.children
            ],
            "dry_run": dry_run,
        }
        if ship is not None:
            payload["ship_gate"] = {
                "step": ship.step_key,
                "card_id": ship_id,
                "unblock": f"hermes kanban unblock {ship_id}" if ship_id else None,
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    # Human output
    header = (
        f"Created chain for {validation.fields.get('VULN_ID', '?')} "
        f"({validation.fields.get('SEVERITY', '?')}):"
        if not dry_run else
        f"Dry-run for {validation.fields.get('VULN_ID', '?')} "
        f"({validation.fields.get('SEVERITY', '?')}):"
    )
    print(header)
    for p in plan.children:
        cid = (
            next((cid for cp, cid in (created or []) if cp.step_key == p.step_key), None)
            if created else None
        )
        marker = cid or "(would create)"
        parents = ", ".join(p.parents)
        gate = p.gate if p.gate == "approval" else "auto"
        print(f"  {marker}  {p.title}  assignee={p.assignee}  "
              f"parents={parents}  gate={gate}")
    if ship is not None and ship_id:
        print()
        print(f"Ship card parked at approval gate: {ship_id}")
        print(f"Unblock with:  hermes kanban unblock {ship_id}")
    return 0


# JSON is imported lazily inside the json-only branch so the module loads
# without the stdlib cost on the hot path of other commands.
import json  # noqa: E402  (placed last on purpose)


# ---------------------------------------------------------------------------
# build_parser (used by hermes_cli.kanban.build_parser)
# ---------------------------------------------------------------------------

def build_parser(sub) -> None:
    """Attach the ``workflow`` subcommand tree to ``sub`` (the kanban
    subparsers). Idempotent: re-calling on the same parser is a no-op.

    The CLI shape (spec §7.1) is::

        hermes kanban workflow list
        hermes kanban workflow show <template>
        hermes kanban workflow <template> <card-ref> [--dry-run] [--force]
            [--assignee KEY=PROFILE]... [--json]

    Action keywords (``list`` / ``show``) are disambiguated from template
    ids in ``cmd_workflow``: if the first positional is one of the reserved
    keywords we dispatch to the matching sub-handler, otherwise we treat
    it as a template id. This keeps ``hermes kanban workflow
    sec-vuln-remediation <card-ref>`` reachable without an extra ``run``
    keyword (spec §7.1) while still giving ``list`` / ``show`` discoverable
    subcommand help.
    """
    # Idempotency guard: ``sub`` is a ``_SubParsersAction`` whose already-
    # registered child parsers live on ``choices``. Bail if a ``workflow``
    # child parser is already attached.
    if isinstance(getattr(sub, "choices", None), dict) and "workflow" in sub.choices:
        return
    p = sub.add_parser(
        "workflow",
        help=(
            "Run a workflow template against a source card. "
            "Templates are defined in hermes_cli/kanban_templates/<id>.md. "
            "Built-ins: `hermes kanban workflow list` to enumerate, "
            "`hermes kanban workflow <template> <card-ref>` to instantiate "
            "(use --dry-run to preview)."
        ),
        description=(
            "Workflow templates codify multi-step remediation chains "
            "(scout issue → recon → reproduce+patch → regression test → "
            "approval-gated ship). Running a template against a card "
            "instantiates the child chain with parent-gated promotion; "
            "the ship card is parked at a sticky needs_input block until "
            "Sab approves via `hermes kanban unblock <id>`. "
            "Auto-merge and auto-deploy are FORBIDDEN.\n\n"
            "Subcommands:\n"
            "  list                    Enumerate every template on disk\n"
            "  show <template>         Print a template's parsed contract\n"
            "  <template> <card-ref>   Instantiate against a source card\n"
            "                          (--dry-run prints the would-be chain)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Reserved first-positionals (dispatched as subcommands in cmd_workflow).
    # We don't use argparse subparsers here because the spec's run form
    # (``hermes kanban workflow <template> <card-ref>``) is positional-only
    # and we want ``list`` / ``show`` to be reserved words without making
    # them mandatory keywords.
    p.add_argument(
        "positional", nargs="*", metavar="[template | list | show ...]",
        help=(
            "Either an action keyword (list, show <template>) or a "
            "template id plus optional card reference. "
            "Examples: `list`, `show sec-vuln-remediation`, "
            "`sec-vuln-remediation t_abc123`."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate + print would-be children without writing. Exit 0 on "
             "success, 2 on validation failure.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Bypass pre-flight stop when an existing chain is found. "
             "NOT a gate bypass: ship card still created sticky-blocked.",
    )
    p.add_argument(
        "--assignee", action="append", default=[], metavar="KEY=PROFILE",
        help="Override the resolved assignee for a single step. Repeatable. "
             "Example: --assignee ship-pr=default",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON")
