"""Offline validation for the jMunch optional skill.

Covers the two things a Hermes user relies on the skill getting right:
  1. Frontmatter meets the skill-authoring rules (AGENTS.md): a <=60-char,
     single-sentence description; required fields present.
  2. Every MCP tool referenced in the prose/examples uses Hermes's real
     ``mcp__<server>__<tool>`` registration convention -- not a bare name or
     the single-underscore ``mcp_<server>_`` form.

Pure stdlib + pytest, no network, no imports from the skill.
"""

import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "mcp" / "jmunch"
SKILL_MD = SKILL_DIR / "SKILL.md"
TEMPLATE_MD = SKILL_DIR / "references" / "hermes-context-template.md"

SERVERS = ("jcodemunch", "jdocmunch", "jdatamunch")

# Representative jMunch tool names spanning all three servers. Every one of
# these must appear only in prefixed form; a bare ``tool(`` call is a bug.
JMUNCH_TOOLS = (
    # jcodemunch
    "index_folder", "index_repo", "list_repos", "search_symbols",
    "get_symbol_source", "get_file_outline", "get_file_tree", "find_importers",
    "find_references", "get_class_hierarchy", "get_call_hierarchy",
    "get_blast_radius", "check_rename_safe", "get_impact_preview",
    "find_dead_code", "get_symbol_complexity", "get_hotspots",
    "get_dependency_cycles", "get_untested_symbols", "get_ranked_context",
    "get_context_bundle",
    # jdocmunch
    "index_local", "doc_index_repo", "doc_list_repos", "search_sections",
    "get_section", "get_sections", "get_section_context", "get_toc",
    "get_toc_tree", "get_document_outline", "get_broken_links",
    "get_doc_coverage",
    # jdatamunch
    "list_datasets", "describe_dataset", "describe_column", "sample_rows",
    "get_rows", "search_data", "aggregate", "join_datasets",
    "get_data_hotspots", "get_correlations", "get_schema_drift",
)


def _frontmatter(text: str) -> dict:
    """Parse the leading ``---`` YAML block into flat top-level key/value pairs.

    Deliberately regex-based (stdlib only, no PyYAML dependency); we only need
    the scalar top-level fields.
    """
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md must start with a --- frontmatter block"
    fields = {}
    for line in m.group(1).splitlines():
        fm = re.match(r"^([A-Za-z_]+):\s?(.*)$", line)
        if fm:
            fields[fm.group(1)] = fm.group(2).strip()
    return fields


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text) -> dict:
    return _frontmatter(skill_text)


# ── Frontmatter ──────────────────────────────────────────────────────────────

def test_skill_files_exist():
    assert SKILL_MD.is_file()
    assert TEMPLATE_MD.is_file()


def test_required_frontmatter_fields(frontmatter):
    for key in ("name", "description", "version", "author", "license"):
        assert frontmatter.get(key), f"frontmatter missing '{key}'"
    assert frontmatter["name"] == "jmunch"


def test_description_within_60_chars(frontmatter):
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit 60): {desc!r}"


def test_description_is_one_sentence_ending_in_period(frontmatter):
    desc = frontmatter["description"]
    assert desc.endswith("."), "description must end with a period"
    # exactly one sentence: no interior sentence break
    assert ". " not in desc, "description must be a single sentence"


def test_description_has_no_marketing_words(frontmatter):
    banned = ("powerful", "comprehensive", "seamless", "advanced")
    low = frontmatter["description"].lower()
    hits = [w for w in banned if w in low]
    assert not hits, f"description contains banned marketing words: {hits}"


# ── Tool naming convention ───────────────────────────────────────────────────

@pytest.mark.parametrize("path", [SKILL_MD, TEMPLATE_MD])
def test_no_single_underscore_mcp_prefix(path):
    """The wrong ``mcp_jcodemunch_*`` (single underscore) form must not appear;
    Hermes registers ``mcp__<server>__<tool>``."""
    text = path.read_text(encoding="utf-8")
    bad = re.findall(r"mcp_j(?:codemunch|docmunch|datamunch)", text)
    assert not bad, f"{path.name} uses single-underscore mcp prefix {set(bad)}"


@pytest.mark.parametrize("path", [SKILL_MD, TEMPLATE_MD])
def test_every_mcp_prefix_names_a_real_server(path):
    text = path.read_text(encoding="utf-8")
    prefixes = set(re.findall(r"mcp__([a-z0-9]+)__", text))
    unknown = prefixes - set(SERVERS)
    assert not unknown, f"{path.name} references unknown MCP servers: {unknown}"


@pytest.mark.parametrize("path", [SKILL_MD, TEMPLATE_MD])
def test_no_bare_jmunch_tool_calls(path):
    """A jMunch tool invoked as a bare ``tool(`` (no ``mcp__server__`` prefix)
    would fail against Hermes's registered names. Prefixed forms are preceded
    by ``__`` so the word-boundary lookbehind excludes them."""
    text = path.read_text(encoding="utf-8")
    alternation = "|".join(re.escape(t) for t in JMUNCH_TOOLS)
    bare = re.findall(rf"(?<!\w)(?:{alternation})\(", text)
    assert not bare, f"{path.name} has un-prefixed jMunch tool calls: {sorted(set(bare))}"


@pytest.mark.parametrize("path", [SKILL_MD, TEMPLATE_MD])
def test_prefixed_examples_are_present(path):
    """Sanity check the fix landed: each server's prefix actually appears."""
    text = path.read_text(encoding="utf-8")
    for server in SERVERS:
        assert f"mcp__{server}__" in text, f"{path.name} missing mcp__{server}__ examples"
