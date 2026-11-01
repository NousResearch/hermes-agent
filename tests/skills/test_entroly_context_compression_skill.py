"""Contract tests for the optional Entroly context-compression skill.

These tests intentionally inspect documentation only. They do not install Entroly,
launch a process, import the integration, or access the network.
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "software-development"
    / "entroly-context-compression"
    / "SKILL.md"
)
SKILL_BYTES = SKILL_PATH.read_bytes()
SKILL_TEXT = SKILL_BYTES.decode("utf-8")

REQUIRED_H2 = [
    "When to Use",
    "Prerequisites",
    "How to Run",
    "Quick Reference",
    "Procedure",
    "Pitfalls",
    "Verification",
]
REQUIRED_COMMANDS = {
    "python -m pip install -U entroly",
    "hermes mcp add entroly --command entroly",
    "hermes config edit",
    "hermes mcp test entroly",
    "hermes skills install official/software-development/entroly-context-compression",
}
REQUIRED_MCP_TOOLS = {
    "optimize_context",
    "entroly_retrieve",
    "create_context_receipt",
    "render_context_receipt",
    "explain_receipt_omission",
    "recover_receipt_omission",
    "verify_response",
}


def _parse_scalar(value: str):
    """Parse the small YAML scalar subset used by Hermes skill metadata."""
    value = value.strip()
    if not value:
        return ""
    if value[0] in {'"', "'"}:
        return ast.literal_eval(value)
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _frontmatter() -> dict[str, object]:
    lines = SKILL_TEXT.splitlines()
    assert lines[0] == "---", "SKILL.md must begin with YAML frontmatter"
    closing = lines.index("---", 1)
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]

    for raw_line in lines[1:closing]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        assert "\t" not in raw_line[:indent], "frontmatter indentation must use spaces"
        key, separator, raw_value = raw_line.strip().partition(":")
        assert separator and key, f"invalid frontmatter line: {raw_line!r}"

        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip():
            parent[key] = _parse_scalar(raw_value)
        else:
            child: dict[str, object] = {}
            parent[key] = child
            stack.append((indent, child))

    return root


def _fenced_blocks(language: str) -> list[str]:
    pattern = rf"^```{re.escape(language)}\s*\n(.*?)^```\s*$"
    return re.findall(pattern, SKILL_TEXT, flags=re.MULTILINE | re.DOTALL)


def _shell_commands() -> list[str]:
    blocks = []
    for language in ("bash", "sh", "shell"):
        blocks.extend(_fenced_blocks(language))
    return [
        line.strip()
        for block in blocks
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _yaml_list(document: str, key: str) -> list[str]:
    match = re.search(
        rf"^(?P<indent>[ \t]*){re.escape(key)}:[ \t]*(?P<value>[^\r\n]*)$",
        document,
        re.MULTILINE,
    )
    assert match, f"MCP YAML is missing {key!r}"
    inline = match.group("value").strip()
    if inline:
        parsed = _parse_scalar(inline)
        assert isinstance(parsed, list), f"{key!r} must be a YAML list"
        return [str(item) for item in parsed]

    key_indent = len(match.group("indent"))
    values: list[str] = []
    for line in document[match.end() :].splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= key_indent:
            break
        item = re.fullmatch(r"\s*-\s*(.+?)\s*", line)
        if item:
            values.append(str(_parse_scalar(item.group(1))))
    return values


def _section(name: str) -> str:
    match = re.search(
        rf"^## {re.escape(name)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        SKILL_TEXT,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"missing section: {name}"
    return match.group("body")


def test_frontmatter_starts_at_byte_zero_and_has_required_identity():
    assert SKILL_BYTES.startswith(b"---"), "no BOM or prose may precede frontmatter"
    metadata = _frontmatter()

    assert metadata["name"] == "entroly-context-compression"
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(metadata["version"]))
    assert str(metadata["author"]).startswith("juyterman1000")
    assert "Hermes Agent" in str(metadata["author"])
    assert metadata["license"] == "Apache-2.0"
    assert set(metadata["platforms"]) == {"linux", "macos", "windows"}


def test_frontmatter_has_normalized_discovery_metadata():
    metadata = _frontmatter()
    hermes = metadata["metadata"]["hermes"]

    assert hermes["category"] == "software-development"
    assert {"context-compression", "mcp"} <= set(hermes["tags"])
    assert {"code-wiki", "codebase-inspection"} <= set(hermes["related_skills"])
    for tag in hermes["tags"]:
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", tag), tag


def test_description_meets_the_skill_listing_contract():
    description = str(_frontmatter()["description"])

    assert len(description) <= 60, len(description)
    assert re.fullmatch(r"[^.!?]+[.!?]", description), description
    prohibited = {"powerful", "comprehensive", "seamless", "advanced"}
    assert not (prohibited & set(re.findall(r"[a-z]+", description.lower())))
    assert "entroly context compression" not in description.lower()


def test_body_uses_the_modern_section_order_exactly_once():
    assert re.search(r"^# Entroly Context Compression Skill\s*$", SKILL_TEXT, re.MULTILINE)
    headings = re.findall(r"^## (.+?)\s*$", SKILL_TEXT, re.MULTILINE)
    assert headings == REQUIRED_H2


def test_documented_setup_commands_are_copy_pasteable():
    commands = _shell_commands()
    how_to_run = _section("How to Run")

    assert REQUIRED_COMMANDS <= set(commands)
    assert "`terminal` tool" in how_to_run
    assert "/reload-mcp" in SKILL_TEXT
    for command in REQUIRED_COMMANDS:
        shlex.split(command)

    # `entroly serve` goes through a different launcher path and is not the
    # supported stdio registration command for Hermes.
    assert not any(shlex.split(line)[:2] == ["entroly", "serve"] for line in commands)
    assert "hermes mcp add entroly --command entroly serve" not in SKILL_TEXT


def test_mcp_example_uses_a_bounded_local_tool_surface():
    examples = [block for block in _fenced_blocks("yaml") if "mcp_servers:" in block]
    assert len(examples) == 1, "provide one canonical Hermes MCP configuration"
    config = examples[0]

    assert re.search(r"^\s*entroly:\s*$", config, re.MULTILINE)
    assert re.search(r"^\s*command:\s*['\"]?entroly['\"]?\s*$", config, re.MULTILINE)
    assert re.search(r"^\s*resources:\s*false\s*$", config, re.MULTILINE)
    assert re.search(r"^\s*prompts:\s*false\s*$", config, re.MULTILINE)

    included = set(_yaml_list(config, "include"))
    assert REQUIRED_MCP_TOOLS <= included
    assert "*" not in included
    assert 7 <= len(included) <= 15, "keep the agent-facing MCP surface reviewable"

    how_to_run = _section("How to Run")
    assert re.search(
        r"add flow.{0,180}leaves.{0,120}resource.{0,80}prompt",
        how_to_run,
        flags=re.IGNORECASE | re.DOTALL,
    ), "explain why CLI tool selection still needs config hardening"
    assert how_to_run.index("hermes mcp add") < how_to_run.index("hermes config edit")
    assert re.search(
        r"update only.{0,120}`entroly` child.{0,120}Preserve every other",
        how_to_run,
        flags=re.IGNORECASE | re.DOTALL,
    ), "config edits must preserve unrelated MCP servers"


def test_skill_separates_hermes_context_management_from_entroly_modes():
    introduction = SKILL_TEXT.split("## When to Use", 1)[0]
    assert "ContextCompressor" in introduction
    assert "ContextEngine" in introduction
    assert re.search(
        r"(?:does not|doesn't|not a replacement for).{0,120}(?:ContextCompressor|ContextEngine)",
        introduction,
        flags=re.IGNORECASE | re.DOTALL,
    )

    prerequisites = _section("Prerequisites")
    assert re.search(r"stdio MCP server named `entroly`", prerequisites)
    assert "entroly[proxy]" in SKILL_TEXT
    assert re.search(
        r"entroly\[proxy\].{0,120}(?:optional|not required|only)",
        SKILL_TEXT,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert re.search(r"external.{0,80}MCP|MCP.{0,80}external", introduction, re.IGNORECASE | re.DOTALL)
    assert "Python 3.10" in prerequisites


def test_verification_is_hermes_specific_and_reports_visible_success():
    verification = _section("Verification")

    assert "hermes mcp test entroly" in verification
    assert "hermes mcp list" in verification
    assert "/reload-mcp" in verification
    assert re.search(r"connected|connection", verification, re.IGNORECASE)
    assert re.search(r"mcp_entroly_", verification)


def test_skill_does_not_publish_universal_benchmark_claims():
    lower = SKILL_TEXT.lower()

    assert not re.search(r"\b\d+(?:[.-]\d+)?(?:\s*[\u2013-]\s*\d+)?\s*%", SKILL_TEXT)
    for unsupported_claim in (
        "100% accuracy",
        "accuracy retained",
        "needleinahaystack",
        "bfcl",
        "auroc",
        "all requests are automatically compressed",
    ):
        assert unsupported_claim not in lower


def test_pitfalls_protect_project_local_recovery_state():
    pitfalls = _section("Pitfalls")

    assert "`.entroly/`" in pitfalls
    assert re.search(r"commit|ignore", pitfalls, re.IGNORECASE)
    assert re.search(
        r"do not delete.{0,120}recoverable receipts",
        pitfalls,
        flags=re.IGNORECASE | re.DOTALL,
    )
