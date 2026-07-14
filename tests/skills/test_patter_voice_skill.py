"""Authoring-standard checks for the patter-voice optional skill.

Pure static assertions over SKILL.md and the sibling MCP manifest — no live
network, no subprocess. Uses stdlib + pytest; PyYAML is used to parse the
frontmatter and the manifest, matching the convention of the other skill
tests in this directory (see test_openclaw_migration.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT / "optional-skills" / "productivity" / "patter-voice" / "SKILL.md"
)
MANIFEST_PATH = REPO_ROOT / "optional-mcps" / "patter-voice" / "manifest.yaml"

# The fixed, closed set of tools the patter-voice MCP exposes. Both the skill
# prose and the manifest's default_enabled list must draw only from this set.
PATTER_TOOLS = {
    "make_call",
    "call_third_party",
    "get_calls",
    "get_transcript",
    "end_call",
    "get_metrics",
    "configure_inbound",
}

# Modern section order mandated by AGENTS.md "Skill authoring standards".
REQUIRED_HEADINGS = [
    "# Patter Voice Skill",
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]

MARKETING_WORDS = {"powerful", "comprehensive", "seamless", "advanced"}


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body) for a `---`-delimited SKILL.md."""
    assert text.startswith("---"), "SKILL.md must start with a YAML frontmatter block"
    parts = text.split("---", 2)
    assert len(parts) == 3, "SKILL.md frontmatter is not properly delimited by ---"
    meta = yaml.safe_load(parts[1])
    assert isinstance(meta, dict), "frontmatter did not parse to a mapping"
    return meta, parts[2]


def _strip_fenced_code(body: str) -> str:
    """Remove ```...``` fenced blocks so we only inspect prose + inline code."""
    return re.sub(r"```.*?```", "", body, flags=re.DOTALL)


def _load_skill() -> tuple[dict, str]:
    return _split_frontmatter(SKILL_PATH.read_text(encoding="utf-8"))


def test_skill_file_exists_and_frontmatter_parses():
    assert SKILL_PATH.exists(), f"missing SKILL.md at {SKILL_PATH}"
    meta, _ = _load_skill()
    assert meta.get("name") == "patter-voice"


def test_description_within_length_and_style_limits():
    meta, _ = _load_skill()
    description = meta["description"]

    # Rule 1: <= 60 chars.
    assert len(description) <= 60, (
        f"description is {len(description)} chars: {description!r}"
    )

    # One sentence ending with a period.
    assert description.endswith("."), "description must end with a period"
    assert description.count(".") == 1, "description must be a single sentence"

    # No marketing words.
    lowered = description.lower()
    assert not any(word in lowered for word in MARKETING_WORDS), (
        f"description contains a banned marketing word: {description!r}"
    )


def test_required_headings_present_and_in_order():
    _, body = _load_skill()
    last_index = -1
    for heading in REQUIRED_HEADINGS:
        pattern = re.compile(rf"^{re.escape(heading)}\s*$", re.MULTILINE)
        match = pattern.search(body)
        assert match is not None, f"missing required heading: {heading!r}"
        assert match.start() > last_index, f"heading out of order: {heading!r}"
        last_index = match.start()


def test_category_and_related_skills():
    meta, _ = _load_skill()
    hermes = meta["metadata"]["hermes"]
    assert hermes["category"] == "productivity"
    assert "telephony" in hermes["related_skills"]


def test_body_only_references_real_patter_tools():
    _, body = _load_skill()
    prose = _strip_fenced_code(body)

    inline_spans = re.findall(r"`([^`]+)`", prose)
    # A patter tool name is a bare snake_case identifier with >=1 underscore.
    tool_like = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
    referenced = {span for span in inline_spans if tool_like.match(span)}

    assert referenced, "SKILL.md prose references no patter tools in backticks"
    unknown = referenced - PATTER_TOOLS
    assert not unknown, f"SKILL.md references unknown tool name(s): {sorted(unknown)}"


def test_manifest_default_enabled_is_subset_of_known_tools():
    assert MANIFEST_PATH.exists(), f"missing manifest at {MANIFEST_PATH}"
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    default_enabled = manifest["tools"]["default_enabled"]
    assert isinstance(default_enabled, list) and default_enabled, (
        "manifest tools.default_enabled must be a non-empty list"
    )
    unknown = set(default_enabled) - PATTER_TOOLS
    assert not unknown, f"manifest enables unknown tool name(s): {sorted(unknown)}"


def test_configure_inbound_is_opt_in_not_default_enabled():
    """`configure_inbound` mutates the inbound-agent config, so it is pruned from
    the default-enabled set — a real tool, but off by default (opt-in at install)."""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    default_enabled = manifest["tools"]["default_enabled"]
    # It is a real patter tool...
    assert "configure_inbound" in PATTER_TOOLS
    # ...but must NOT be pre-checked by default.
    assert "configure_inbound" not in default_enabled


def test_manifest_install_ref_is_immutable_sha_pin():
    """Manifests never float HEAD: install.ref must be a full 40-hex commit SHA."""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    ref = manifest["install"]["ref"]
    assert re.fullmatch(r"[0-9a-f]{40}", ref), (
        f"install.ref is not an immutable 40-hex SHA: {ref!r}"
    )


def test_manifest_prompts_exactly_the_env_vars_the_server_reads():
    """The prompted env vars must mirror what patter-mcp@c87b1b17 and the getpatter
    SDK actually read from the environment: the hardcoded Twilio trio + the default
    Realtime engine's OPENAI_API_KEY, plus the optional Deepgram / ElevenLabs vars
    (ELEVENLABS_AGENT_ID drives the elevenlabs_convai engine).

    TELNYX_API_KEY is intentionally ABSENT — the server hardcodes Twilio and never
    reads a Telnyx key at this pin, so prompting for it would be a dead prompt.
    """
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    prompted = {entry["name"] for entry in manifest["auth"]["env"]}
    expected = {
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_PHONE_NUMBER",
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_AGENT_ID",
    }
    assert prompted == expected, (
        f"prompted env set drifted from the server: {sorted(prompted)}"
    )
    assert "TELNYX_API_KEY" not in prompted
