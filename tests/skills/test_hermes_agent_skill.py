"""The `hermes-agent` skill is what a running Hermes knows about itself.

`website/` is never packaged, so an installed Hermes has no local copy of the
user guide; skills ARE synced into `$HERMES_HOME/skills/`. The skill therefore
does not try to restate the product — it routes to the published `llms.txt`,
which is generated from the docs tree on every build and so can never be behind
the feature set. These tests keep that routing honest: the index has to be where
the skill says it is, and every reference has to be reachable, otherwise a
shipped feature is invisible and the agent answers "Hermes can't do that."
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "autonomous-ai-agents" / "hermes-agent"
SKILL_MD = SKILL_DIR / "SKILL.md"
GENERATOR = REPO / "website" / "scripts" / "generate-llms-txt.py"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_every_referenced_file_exists(skill_text):
    """Routing a question to a file that isn't there is a dead end."""
    targets = set(re.findall(r"`((?:references|templates)/[^`]+)`", skill_text))

    assert targets, "the skill's routing table no longer references any files"
    for target in sorted(targets):
        assert (SKILL_DIR / target).exists(), f"SKILL.md routes to missing {target}"


def test_every_reference_is_reachable_from_the_skill(skill_text):
    """An unrouted reference is one the agent will never think to open.

    This is the failure that produced the original complaint: content can exist
    and still be invisible because nothing points at it.
    """
    on_disk = {f"references/{path.name}" for path in (SKILL_DIR / "references").glob("*.md")}
    routed = set(re.findall(r"`(references/[^`]+)`", skill_text))

    assert not (on_disk - routed), (
        f"reference files no reader will ever reach: {sorted(on_disk - routed)} — "
        "add a routing-table row in SKILL.md"
    )


def test_unknown_features_route_to_the_published_index(skill_text):
    """The catch-all is what makes coverage of the whole product possible."""
    assert "/docs/llms.txt" in skill_text
    # web_extract can be disabled; terminal never is.
    assert "curl" in skill_text, "no way to reach the index without web tools"


def test_the_index_is_published_where_the_skill_says_it_is(skill_text):
    """A skill pointing at a URL nobody generates is worse than no routing."""
    spec = importlib.util.spec_from_file_location("generate_llms_txt", GENERATOR)
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    assert f"{gen.SITE_BASE}/llms.txt" in skill_text


NATIVE_MCP_TEXT = (
    SKILL_DIR / "references" / "native-mcp.md"
).read_text(encoding="utf-8")


def test_native_mcp_reference_matches_runtime_timeout():
    source = (REPO / "tools" / "mcp_tool.py").read_text(encoding="utf-8")
    match = re.search(r"_DEFAULT_TOOL_TIMEOUT\s*=\s*(\d+)", source)
    assert match, "the runtime no longer declares a default MCP tool timeout"
    assert f"| `timeout`         | int    | `{match.group(1)}`" in NATIVE_MCP_TEXT


def test_native_mcp_reference_documents_reload():
    assert "/reload-mcp" in NATIVE_MCP_TEXT
    assert "auto_reload_on_config_change" in NATIVE_MCP_TEXT
    assert "no hot-reload" not in NATIVE_MCP_TEXT
    assert "requires restarting the agent" not in NATIVE_MCP_TEXT


def test_native_mcp_reference_keeps_credentials_out_of_config():
    forbidden = ("ghp_x", "sk-x", 'GITHUB_PERSONAL_ACCESS_TOKEN: "ghp', "Bearer sk-")
    for token in forbidden:
        assert token not in NATIVE_MCP_TEXT
    assert "${GITHUB_PERSONAL_ACCESS_TOKEN}" in NATIVE_MCP_TEXT
    assert ".env" in NATIVE_MCP_TEXT


def test_windows_reference_does_not_dump_the_environment():
    text = (SKILL_DIR / "references" / "windows-quirks.md").read_text(encoding="utf-8")
    assert 'os.environ.get("SYSTEMROOT"' in text
    assert "echo `os.environ`" not in text


def test_portal_proxy_reference_requires_client_authorization():
    text = (
        SKILL_DIR / "references" / "portal-auth-for-third-party-apps.md"
    ).read_text(encoding="utf-8")
    assert "with any\nplaceholder key" not in text
    for required in ("SO_PEERCRED", "127.0.0.1", "trust boundary"):
        assert required in text
