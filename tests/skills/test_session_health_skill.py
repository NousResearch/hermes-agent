"""Tests for the session-health bundled skill pair.

Covers SKILL.md frontmatter invariants (AGENTS.md skill authoring
standards), validates every ``session_search(...)`` example documented in
the two SKILL.md files against the tool's real schema and calling shapes,
and grounds the workflows' core assumptions against the actual tool:
browse excludes the current session, the read shape ignores
role_filter/limit, unsupported sort values are dropped, and scroll rejects
the active session lineage. No live network calls — everything runs on a
temp SQLite session DB.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import SESSION_SEARCH_SCHEMA, session_search

CATEGORY_DIR = Path(__file__).resolve().parents[2] / "skills" / "session-health"
SKILL_NAMES = ["session-health-check", "session-loop-detection"]
SCHEMA_PARAMS = set(SESSION_SEARCH_SCHEMA["parameters"]["properties"])
SORT_ENUM = set(SESSION_SEARCH_SCHEMA["parameters"]["properties"]["sort"]["enum"])

_CALL_RE = re.compile(r"session_search\(([^()]*)\)")


def _read_skill(name: str) -> str:
    return (CATEGORY_DIR / name / "SKILL.md").read_text(encoding="utf-8")


def _frontmatter(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    content = _read_skill(name)
    assert content.startswith("---\n"), "frontmatter must start at byte 0"
    match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    assert match, "frontmatter must close with ---"
    return yaml.safe_load(match.group(1))


def _documented_calls(name: str) -> list[dict]:
    """Extract every session_search(...) example as a kwarg dict.

    Angle-bracket placeholders (``<id>``, ``"<task keywords>"``) are
    replaced with ``0`` / ``"0"`` so each example parses as real Python.
    """
    calls = []
    for raw_args in _CALL_RE.findall(_read_skill(name)):
        cleaned = re.sub(r"<[^<>]+>", "0", raw_args)
        node = ast.parse(f"session_search({cleaned})", mode="eval").body
        assert not node.args, f"example must be keyword-only: ({raw_args})"
        kwargs = {}
        for kw in node.keywords:
            assert kw.arg is not None, f"no **kwargs in examples: ({raw_args})"
            kwargs[kw.arg] = (
                kw.value.value if isinstance(kw.value, ast.Constant) else None
            )
        calls.append(kwargs)
    assert len(calls) >= 4, f"{name}: expected several documented calls"
    return calls


# =========================================================================
# Frontmatter invariants (AGENTS.md skill authoring standards)
# =========================================================================


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_frontmatter_required_fields(name):
    fm = _frontmatter(name)
    for field in ("name", "description", "version", "author", "license"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == name  # matches directory name


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_description_within_hardline_limit(name):
    desc = _frontmatter(name)["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit 60)"
    assert desc.endswith("."), "description must end with a period"
    assert ". " not in desc, "description must be a single sentence"


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_author_credits_contributor_first(name):
    author = _frontmatter(name)["author"]
    first = re.split(r"[,+]", author)[0].strip()
    assert first != "Hermes Agent", "human contributor must be credited first"
    assert "Hermes Agent" in author  # secondary collaborator


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_platforms_declared(name):
    fm = _frontmatter(name)
    assert "platforms" in fm and len(fm["platforms"]) >= 1


def test_category_description_exists():
    content = (CATEGORY_DIR / "DESCRIPTION.md").read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "description:" in content


# =========================================================================
# Documented session_search examples match the real tool contract
# =========================================================================


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_examples_use_only_schema_parameters(name):
    for kwargs in _documented_calls(name):
        unknown = set(kwargs) - SCHEMA_PARAMS
        assert not unknown, f"undocumented parameters {unknown} in {kwargs}"


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_examples_use_supported_sort_values(name):
    for kwargs in _documented_calls(name):
        if "sort" in kwargs:
            assert kwargs["sort"] in SORT_ENUM, f"unsupported sort in {kwargs}"


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_examples_use_supported_role_filter_values(name):
    for kwargs in _documented_calls(name):
        if "role_filter" in kwargs:
            roles = {r.strip() for r in kwargs["role_filter"].split(",")}
            assert roles <= {"user", "assistant", "tool"}, kwargs


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_examples_match_a_real_calling_shape(name):
    """Each example must be exactly one of browse/discovery/read/scroll.

    Guards the failure mode from review: role_filter/limit on the read
    shape or browse shape are silently ignored by the tool, so documenting
    them there teaches a call that does not do what the prose claims.
    """
    for kwargs in _documented_calls(name):
        keys = set(kwargs) - {"profile"}
        if "around_message_id" in keys:  # scroll
            assert "session_id" in keys, f"scroll needs session_id: {kwargs}"
            assert keys <= {"session_id", "around_message_id", "window"}, kwargs
        elif "session_id" in keys:  # read — role_filter/limit/sort ignored
            assert keys == {"session_id"}, (
                f"read shape ignores every other parameter: {kwargs}"
            )
        elif "query" in keys:  # discovery
            assert keys <= {"query", "role_filter", "limit", "sort"}, kwargs
        else:  # browse — only limit applies
            assert keys <= {"limit"}, f"browse ignores {keys - {'limit'}}"


# =========================================================================
# Workflow assumptions grounded against the actual tool behavior
# =========================================================================


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    yield d
    d.close()


@pytest.fixture
def seeded(db):
    """One live session + one past session that looped on a failing tool."""
    db.create_session("s_current", source="cli")
    db.append_message("s_current", role="user", content="live task")
    live_mid = db.append_message("s_current", role="assistant", content="working")

    db.create_session("s_past", source="cli")
    db.append_message("s_past", role="user", content="fix the deploy")
    first_loop_mid = None
    for _ in range(3):
        mid = db.append_message(
            "s_past",
            role="assistant",
            content="",
            tool_calls=[{"name": "terminal", "arguments": {"cmd": "deploy"}}],
        )
        first_loop_mid = first_loop_mid or mid
        db.append_message(
            "s_past", role="tool", tool_name="terminal",
            content="error: deploy failed",
        )
    db._conn.commit()
    return {"db": db, "live_mid": live_mid, "loop_mid": first_loop_mid}


def test_browse_excludes_current_session(seeded):
    """Step 'pick target sessions': browse lists PAST sessions only."""
    out = json.loads(
        session_search(db=seeded["db"], current_session_id="s_current", limit=10)
    )
    assert out["success"] and out["mode"] == "browse"
    ids = [r["session_id"] for r in out["results"]]
    assert "s_past" in ids
    assert "s_current" not in ids


def test_browse_returns_triage_metadata(seeded):
    """The health-check triage relies on message_count/preview metadata."""
    out = json.loads(
        session_search(db=seeded["db"], current_session_id="s_current", limit=10)
    )
    row = next(r for r in out["results"] if r["session_id"] == "s_past")
    assert row["message_count"] == 7
    for key in ("title", "last_active", "preview"):
        assert key in row


def test_read_shape_ignores_role_filter_and_limit(seeded):
    """The tool has no per-turn filter shape — the skills must not claim one."""
    out = json.loads(
        session_search(db=seeded["db"], session_id="s_past",
                       role_filter="assistant", limit=1)
    )
    assert out["success"] and out["mode"] == "read"
    assert out["message_count"] == 7  # limit=1 had no effect
    assert {m["role"] for m in out["messages"]} == {"user", "assistant", "tool"}


def test_read_shape_exposes_tool_calls_for_loop_detection(seeded):
    """Loop detection compares tool_calls across consecutive assistant turns."""
    out = json.loads(session_search(db=seeded["db"], session_id="s_past"))
    repeated = [
        m for m in out["messages"]
        if m["role"] == "assistant" and m.get("tool_calls")
    ]
    assert len(repeated) == 3
    names = {c["name"] for m in repeated for c in m["tool_calls"]}
    assert names == {"terminal"}


def test_unsupported_sort_value_is_dropped(seeded):
    """`sort="recent"` (rejected in review) silently degrades to relevance."""
    db = seeded["db"]
    with patch.object(db, "search_messages", wraps=db.search_messages) as spy:
        session_search(db=db, query="deploy", sort="recent")
        assert spy.call_args.kwargs["sort"] is None
        session_search(db=db, query="deploy", sort="newest")
        assert spy.call_args.kwargs["sort"] == "newest"


def test_discovery_finds_tool_errors_in_past_sessions(seeded):
    """Step 'scan for failure signatures': discovery scoped to tool output."""
    out = json.loads(
        session_search(db=seeded["db"], query="error OR failed",
                       role_filter="tool", limit=5, sort="newest",
                       current_session_id="s_current")
    )
    assert out["success"] and out["mode"] == "discover"
    assert out["count"] >= 1
    hit = out["results"][0]
    assert hit["session_id"] == "s_past"
    assert isinstance(hit["match_message_id"], int)


def test_scroll_drills_into_past_session(seeded):
    """Step 'drill into hits': anchor on match_message_id in a past session."""
    out = json.loads(
        session_search(db=seeded["db"], session_id="s_past",
                       around_message_id=seeded["loop_mid"], window=10,
                       current_session_id="s_current")
    )
    assert out["success"] and out["mode"] == "scroll"
    assert any(m.get("anchor") for m in out["messages"])


def test_scroll_rejects_current_session_lineage(seeded):
    """Grounds the 'past sessions only' pitfall both skills document."""
    out = json.loads(
        session_search(db=seeded["db"], session_id="s_current",
                       around_message_id=seeded["live_mid"],
                       current_session_id="s_current")
    )
    assert out.get("success") is False
    assert "scroll rejected" in out["error"]
