"""Testable agent constitutions (SOUL.md).

A project ``SOUL.md`` may declare a structured constitution: YAML frontmatter
(``soul_version``, ``eval_suite``) plus ``## 1. Axioms`` / ``## 2. Values`` /
``## 3. Dispositions`` sections, paired with a co-located ``SOUL.suite.yaml``
probe suite. The constitution compiles into the system prompt AFTER project
instructions; a soul that fails the entry lint — every axiom needs at least
one ``must_refuse`` and one ``must_not_refuse`` probe — is rejected and never
injected, so untested axioms never reach the model.

This coexists with Hermes's identity SOUL.md: a free-form SOUL.md without the
constitution structure is left alone and keeps loading through the normal
identity slot (``load_soul_md``); only structured files become constitutions.

Formation security boundary: agent writes to the soul eval artifacts
(``SOUL.suite.yaml``, ``SOUL.baseline.json``) are hard-denied in
``tools/file_tools_write_guards.py`` before any configurable rule runs, so the
agent can never rewrite its own tests. ``SOUL.md`` itself stays on the
existing always-ask protected-instruction gate (human approval required, even
under ``--yolo``).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from agent.soul_eval import lint_suite

logger = logging.getLogger(__name__)

SOUL_FILENAME = "SOUL.md"
DEFAULT_SUITE_FILENAME = "SOUL.suite.yaml"

# Eval artifacts the agent may never write: its own test probes and baselines.
# Names are conventional and fixed so the guard stays a pure basename check
# with no config or I/O — there is nothing to misconfigure into an open gate.
SOUL_EVAL_ARTIFACT_BASENAMES = frozenset({"soul.suite.yaml", "soul.baseline.json"})


def is_soul_eval_artifact(name: str) -> bool:
    """True when *name* is a soul eval artifact basename (case-insensitive).

    The basename split handles both POSIX and Windows separators so the guard
    holds regardless of the platform the path string came from."""
    base = re.split(r"[\\/]", name)[-1]
    return base.lower() in SOUL_EVAL_ARTIFACT_BASENAMES


def soul_formation_denial(target: str) -> str:
    """Message shown to the agent when the formation guard fires. It names the
    rule and the legitimate path: the human edits the file in their own editor."""
    return (
        f"Denied by the soul formation guard: agents may never edit {target}. "
        "Axioms change only by human edit plus a full eval re-run. "
        "Ask the user to make this change in their own editor."
    )


class SoulLoadError(Exception):
    """A soul file (or its suite) failed to load. ``messages`` carries every
    problem; the soul is rejected and nothing is injected."""

    def __init__(self, messages: List[str]):
        super().__init__("; ".join(messages))
        self.messages = messages


@dataclass
class Axiom:
    id: str
    statement: str
    enforced_by: str = ""


@dataclass
class SoulFile:
    path: str
    version: str
    agent: str = ""
    owner: str = ""
    suite_path: str = ""
    purpose: str = ""
    axioms: List[Axiom] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    dispositions: str = ""


def _split_frontmatter(content: str) -> tuple:
    """Return (frontmatter_dict, body). Frontmatter must start at offset 0."""
    if not content.startswith("---\n") and not content.startswith("---\r\n"):
        return {}, content
    lines = content.split("\n")
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end is None:
        return {}, content
    try:
        fm = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError:
        return {}, content
    if not isinstance(fm, dict):
        return {}, content
    return fm, "\n".join(lines[end + 1:])


_AXIOMS_HEADING = re.compile(r"^##\s+1\.\s*axioms\b", re.IGNORECASE | re.MULTILINE)


def is_constitution_content(content: str) -> bool:
    """True when the text is a structured soul constitution (not a free-form
    identity SOUL.md): YAML frontmatter carrying ``soul_version`` plus a
    ``## 1. Axioms`` section."""
    fm, _body = _split_frontmatter(content)
    if not fm.get("soul_version"):
        return False
    return bool(_AXIOMS_HEADING.search(content))


def _section_text(body: str, heading_re: str) -> str:
    """Raw markdown under the first heading matching *heading_re*, up to the
    next ``## `` heading or EOF."""
    m = re.search(heading_re, body, re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""
    rest = body[m.end():]
    nxt = re.search(r"^##\s+", rest, re.MULTILINE)
    if nxt:
        rest = rest[:nxt.start()]
    return rest.strip()


def _parse_markdown_table(section: str) -> List[List[str]]:
    """Parse the first markdown table in *section* into data rows (lists of
    stripped cells, backticks removed from the first column)."""
    lines = [ln.strip() for ln in section.split("\n")]
    # Find header row followed by a separator row (|---|---|).
    start = None
    for i in range(len(lines) - 1):
        if lines[i].startswith("|") and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1]):
            start = i + 2
            break
    if start is None:
        return []
    rows: List[List[str]] = []
    for ln in lines[start:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


def _parse_axioms(body: str) -> List[Axiom]:
    section = _section_text(body, r"^##\s+1\.\s*axioms\b.*$")
    axioms: List[Axiom] = []
    for cells in _parse_markdown_table(section):
        if len(cells) < 2:
            continue
        axiom_id = cells[0].strip().strip("`").strip()
        if not axiom_id:
            continue
        axioms.append(Axiom(
            id=axiom_id,
            statement=cells[1].strip(),
            enforced_by=cells[2].strip().strip("`") if len(cells) > 2 else "",
        ))
    return axioms


def _parse_values(body: str) -> List[str]:
    section = _section_text(body, r"^##\s+2\.\s*values\b.*$")
    values: List[str] = []
    for cells in _parse_markdown_table(section):
        if len(cells) >= 2 and cells[1].strip():
            values.append(cells[1].strip().strip("`"))
    return values


def parse_constitution(content: str, path: str) -> SoulFile:
    """Parse a structured SOUL.md constitution. Raises SoulLoadError."""
    fm, body = _split_frontmatter(content)
    version = str(fm.get("soul_version") or "").strip()
    if not version:
        raise SoulLoadError([f"soul: {path} has no soul_version in its frontmatter"])
    axioms = _parse_axioms(body)
    if not axioms:
        raise SoulLoadError([f"soul: {path} defines no axioms in §1"])
    values = _parse_values(body)
    purpose = _section_text(body, r"^##\s+0\.\s*purpose\b.*$")
    dispositions = _section_text(body, r"^##\s+3\.\s*dispositions\b.*$")
    suite_name = str(fm.get("eval_suite") or DEFAULT_SUITE_FILENAME).strip()
    suite_path = str(Path(path).parent / suite_name)
    return SoulFile(
        path=path,
        version=version,
        agent=str(fm.get("agent") or ""),
        owner=str(fm.get("owner") or ""),
        suite_path=suite_path,
        purpose=purpose,
        axioms=axioms,
        values=values,
        dispositions=dispositions,
    )


def entry_lint(soul: SoulFile, suite: Dict) -> List[str]:
    """Runtime entry lint: nothing enters the soul that cannot be tested. Every
    axiom id in SOUL.md §1 must have at least one must_refuse and one
    must_not_refuse probe in the co-located suite, or the soul is rejected."""
    errs: List[str] = []
    if not soul.axioms:
        errs.append(f"soul: {soul.path} defines no axioms in §1")
    linted = lint_suite({**suite, "axioms": [a.id for a in soul.axioms]})
    for e in linted:
        if e.startswith("ORPHAN"):
            errs.append(f"soul: axiom {e.split(' ')[2]} has no paired probes in {soul.suite_path}")
        else:
            errs.append(f"soul: suite {soul.suite_path}: {e}")
    return errs


def compile_system_section(soul: SoulFile) -> str:
    """Compile the constitution into its system-prompt section. Axioms are
    absolute; values are ranked; a disposition may never soften an axiom."""
    lines = [
        f'<soul version="{soul.version}">',
        "The following constitution governs this session. Axioms are absolute and never traded off;",
        "values below are ranked in strict precedence order. A disposition may never soften an axiom.",
    ]
    if soul.purpose:
        lines.append(f"Purpose: {soul.purpose}")
    lines.append("## Axioms")
    lines.extend(f"- [{a.id}] {a.statement}" for a in soul.axioms)
    lines.append("## Values (ranked)")
    lines.extend(f"{i + 1}. {v}" for i, v in enumerate(soul.values))
    if soul.dispositions:
        lines.extend(["## Dispositions", soul.dispositions])
    lines.extend([
        "## Formation",
        "You may never edit SOUL.suite.yaml or SOUL.baseline.json. Edits to SOUL.md always "
        "require human approval. Axioms change only by human edit plus a full eval re-run "
        "(`hermes soul eval`). If the user asks you to change the soul, ask them to make the "
        "change in their own editor and re-run the eval.",
        "</soul>",
    ])
    return "\n".join(lines)


def _default_home() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        env = os.environ.get("HERMES_HOME")
        return Path(env).expanduser() if env else None


def find_constitution_file(
    start_dir: Optional[str] = None, home_dir: Optional[str] = None
) -> Optional[str]:
    """Walk up from *start_dir* looking for SOUL.md; fall back to the Hermes
    home SOUL.md. Returns the path whether or not it is a structured
    constitution — callers decide with :func:`is_constitution_content`."""
    start = Path(start_dir or os.getcwd()).resolve()
    d = start
    while True:
        candidate = d / SOUL_FILENAME
        if candidate.is_file():
            return str(candidate)
        parent = d.parent
        if parent == d:
            break
        d = parent
    home = Path(home_dir).expanduser() if home_dir else _default_home()
    if home is not None:
        candidate = home / SOUL_FILENAME
        if candidate.is_file():
            return str(candidate)
    return None


def load_constitution(path: str) -> SoulFile:
    """Read and parse the constitution at *path*. Raises SoulLoadError."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise SoulLoadError([f"soul: cannot read {path}: {e}"])
    if not is_constitution_content(content):
        raise SoulLoadError([f"soul: {path} is not a structured constitution"])
    return parse_constitution(content, path)


def load_constitution_section(
    cwd: Optional[str] = None, home_dir: Optional[str] = None
) -> Optional[str]:
    """Compiled ``<soul>`` system-prompt section, or None when no constitution
    exists. A soul that fails the entry lint is rejected: it is logged and
    nothing is injected, so untested axioms never reach the model."""
    from agent.soul_eval import parse_suite

    path = find_constitution_file(cwd, home_dir)
    if not path:
        return None
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("soul rejected at load; not injected into system prompt: cannot read %s: %s", path, e)
        return None
    if not is_constitution_content(content):
        # Free-form identity SOUL.md — not a constitution, left to the identity slot.
        return None
    try:
        soul = parse_constitution(content, path)
    except SoulLoadError as e:
        logger.warning("soul rejected at load; not injected into system prompt: %s", e.messages)
        return None
    try:
        suite_text = Path(soul.suite_path).read_text(encoding="utf-8")
    except OSError:
        logger.warning(
            "soul rejected at load; not injected into system prompt: "
            "eval suite not found at %s; every soul needs a paired probe suite",
            soul.suite_path,
        )
        return None
    try:
        suite = parse_suite(suite_text)
    except Exception as e:
        logger.warning("soul rejected at load; not injected into system prompt: cannot parse suite: %s", e)
        return None
    errs = entry_lint(soul, suite)
    if errs:
        logger.warning("soul rejected at load; not injected into system prompt: %s", errs)
        return None
    return compile_system_section(soul)
