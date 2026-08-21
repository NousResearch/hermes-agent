"""Format tests for the elisym optional skill.

Stdlib + pytest only; NO network calls. Validates SKILL.md against the
hardline skill authoring standards (AGENTS.md): description length, author
credit, platforms gating, modern section order, and that the prose only
references MCP tools the @elisym/mcp server actually ships.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "autonomous-ai-agents"
    / "elisym"
)
SKILL_MD = SKILL_DIR / "SKILL.md"

# Tool names shipped by @elisym/mcp (packages/mcp/src/tools in
# github.com/elisymlabs/elisym). Keep in sync when the server adds tools.
MCP_TOOLS = {
    "add_contact",
    "buy_capability",
    "create_agent",
    "create_job",
    "estimate_payment_cost",
    "fetch_job_file",
    "get_agent_policies",
    "get_balance",
    "get_dashboard",
    "get_identity",
    "get_job_result",
    "get_messages",
    "list_agents",
    "list_capabilities",
    "list_contacts",
    "list_conversations",
    "list_job_sessions",
    "list_my_jobs",
    "remove_contact",
    "search_agents",
    "send_message",
    "send_payment",
    "stop_agent",
    "submit_and_pay_job",
    "submit_and_pay_job_from_file",
    "submit_diff_review",
    "submit_feedback",
    "switch_agent",
    "verify_agent_identities",
    "withdraw",
}

# Tool parameter / result field names the prose legitimately mentions in
# backticks; not tools themselves.
TOOL_PARAMS = {
    "provider_npub",
    "max_price_lamports",
    "is_contact",
    "last_worked_at",
}

# Tool names from a pre-release draft of the skill that never existed in the
# published server. They must not reappear.
LEGACY_TOOLS = {
    "find_agent",
    "submit_job_request",
    "pay_request",
    "wait_for_job_result",
    "check_payment_status",
    "wallet_balance",
}

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", skill_text, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return match.group(1)


def _field(frontmatter: str, name: str) -> str:
    match = re.search(rf"^{name}: (.+)$", frontmatter, re.MULTILINE)
    assert match, f"frontmatter missing field: {name}"
    return match.group(1).strip()


def test_skill_md_present() -> None:
    assert SKILL_MD.is_file(), f"missing {SKILL_MD}"


def test_required_frontmatter_fields(frontmatter: str) -> None:
    for field in ("name", "description", "version", "author", "license", "platforms"):
        _field(frontmatter, field)


def test_name(frontmatter: str) -> None:
    assert _field(frontmatter, "name") == "elisym"


def test_description_hardline_standard(frontmatter: str) -> None:
    desc = _field(frontmatter, "description")
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit 60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"
    assert desc.count(".") == 1, "description must be a single sentence"
    assert "elisym" not in desc.lower(), "description must not repeat the skill name"


def test_author_credits_human_first(frontmatter: str) -> None:
    author = _field(frontmatter, "author")
    assert author.startswith("Igor Peregudov"), (
        f"author must credit the human contributor first, got: {author!r}"
    )


def test_platforms_cover_all_major(frontmatter: str) -> None:
    platforms = _field(frontmatter, "platforms")
    for os_name in ("linux", "macos", "windows"):
        assert os_name in platforms, f"platforms missing {os_name}: {platforms!r}"


def test_sections_present_in_modern_order(skill_text: str) -> None:
    positions = []
    for section in REQUIRED_SECTIONS:
        pos = skill_text.find(f"\n{section}\n")
        assert pos != -1, f"missing section: {section}"
        positions.append(pos)
    assert positions == sorted(positions), (
        "sections out of order; expected " + ", ".join(REQUIRED_SECTIONS)
    )


def test_only_real_mcp_tools_referenced(skill_text: str) -> None:
    body = skill_text.split("---", 2)[2]
    referenced = set(re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", body))
    unknown = referenced - MCP_TOOLS - TOOL_PARAMS
    assert not unknown, f"prose references unknown MCP tools: {sorted(unknown)}"


def test_no_legacy_tool_names(skill_text: str) -> None:
    present = {tool for tool in LEGACY_TOOLS if tool in skill_text}
    assert not present, f"legacy draft tool names present: {sorted(present)}"


def test_no_legacy_identity_path(skill_text: str) -> None:
    assert ".elisym/agents/" not in skill_text, (
        "legacy ~/.elisym/agents/ layout referenced; identities live at ~/.elisym/<name>/"
    )
