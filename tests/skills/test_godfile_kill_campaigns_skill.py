"""Tests for the godfile-kill-campaigns skill's w3-verbatim-audit template.

Covers the authoring standards (frontmatter shape, ≤60-char description)
and the behavior of ``templates/w3-verbatim-audit.py``'s def_spans() —
the AST-anchored finder that must resolve BOTH plain assignments and the
two shapes that break top-level-only finders (PR #79609):

  1. annotated module constants (ast.AnnAssign), e.g. ``_AUX_TASKS: list[int]``
  2. constants assigned inside a module-level try/except (or if) block

Both shapes never appear in ``tree.body``, so a tree.body-only finder
reports them NOT IN LIVE even when they exist — the SKILL.md's own
documented traps ("ast.walk for Assign/AnnAssign targets anywhere").
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "software-development" / "godfile-kill-campaigns"
TEMPLATE = SKILL_DIR / "templates" / "w3-verbatim-audit.py"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def w3v():
    spec = importlib.util.spec_from_file_location("w3_verbatim_audit", TEMPLATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Authoring standards
# ---------------------------------------------------------------------------


def test_skill_files_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert (SKILL_DIR / "templates" / "w3-verbatim-audit.py").is_file()


def test_description_within_limit(frontmatter: dict) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit 60): {desc!r}"


# ---------------------------------------------------------------------------
# def_spans() — every shape the checker must resolve
# ---------------------------------------------------------------------------


def test_plain_assign_found(w3v) -> None:
    src = 'PLAIN = 1\n\n_X = "a"\n'
    spans = w3v.def_spans(src)
    assert set(spans) == {"PLAIN", "_X"}
    assert w3v.span_text(src, spans["PLAIN"]) == "PLAIN = 1\n"


def test_annotated_assign_found(w3v) -> None:
    src = "_AUX_TASKS: list[int] = [1, 2, 3]\n"
    spans = w3v.def_spans(src)
    assert "_AUX_TASKS" in spans, "AnnAssign constant must resolve (PR #79609)"
    assert w3v.span_text(src, spans["_AUX_TASKS"]) == "_AUX_TASKS: list[int] = [1, 2, 3]\n"


def test_try_except_nested_constant_maps_to_enclosing_statement(w3v) -> None:
    src = (
        "try:\n"
        "    _PET_REFERENCE_MAX_BYTES = 512 * 1024\n"
        "except ImportError:\n"
        "    _PET_REFERENCE_MAX_BYTES = 0\n"
    )
    spans = w3v.def_spans(src)
    assert "_PET_REFERENCE_MAX_BYTES" in spans, "try/except-nested constant must resolve (PR #79609)"
    node = spans["_PET_REFERENCE_MAX_BYTES"]
    assert node.lineno == 1 and node.end_lineno == 4, "span must be the enclosing top-level try statement"
    assert w3v.span_text(src, node) == src


def test_if_nested_constant_maps_to_enclosing_statement(w3v) -> None:
    src = "if True:\n    _Y = 2\n"
    spans = w3v.def_spans(src)
    assert "_Y" in spans
    node = spans["_Y"]
    assert node.lineno == 1 and node.end_lineno == 2


def test_function_def_still_found(w3v) -> None:
    src = "def _run():\n    return 1\n"
    spans = w3v.def_spans(src)
    assert "_run" in spans
    assert w3v.span_text(src, spans["_run"]) == src


def test_annotation_on_attribute_target_is_not_a_name_def(w3v) -> None:
    src = "obj.attr: int = 1\n"
    assert w3v.def_spans(src) == {}


def test_comment_block_above_annotated_constant(w3v) -> None:
    src = "# Task list for auxiliary work.\n_AUX_TASKS: list[int] = [1, 2, 3]\n"
    node = w3v.def_spans(src).get("_AUX_TASKS")
    assert node is not None
    # comment_block_above() scans upward from the defining node's lineno
    block = w3v.comment_block_above(src, "_AUX_TASKS")
    assert block == ["# Task list for auxiliary work."]
