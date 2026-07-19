from __future__ import annotations

from pathlib import Path

import pytest

from tools.skills_tool import (
    _get_required_environment_variables,
    _parse_frontmatter,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _environment_metadata(relative_path: str) -> dict[str, dict[str, object]]:
    frontmatter, _ = _parse_frontmatter(_read(relative_path))
    return {
        entry["name"]: entry
        for entry in _get_required_environment_variables(frontmatter)
    }


@pytest.mark.parametrize(
    ("relative_path", "environment_variable"),
    [
        (
            "optional-skills/research/duckduckgo-search/SKILL.md",
            "FIRECRAWL_API_KEY",
        ),
        (
            "optional-skills/research/searxng-search/SKILL.md",
            "FIRECRAWL_API_KEY",
        ),
    ],
)
def test_keyless_search_fallbacks_do_not_declare_firecrawl_requirement(
    relative_path: str,
    environment_variable: str,
) -> None:
    assert environment_variable not in _environment_metadata(relative_path)


@pytest.mark.parametrize(
    ("relative_path", "environment_variable"),
    [
        (
            "optional-skills/devops/pinggy-tunnel/SKILL.md",
            "PINGGY_TOKEN",
        ),
        (
            "optional-skills/migration/openclaw-migration/SKILL.md",
            "TELEGRAM_BOT_TOKEN",
        ),
        (
            "optional-skills/mlops/lambda-labs/SKILL.md",
            "LAMBDA_API_KEY",
        ),
        (
            "optional-skills/research/parallel-cli/SKILL.md",
            "PARALLEL_API_KEY",
        ),
        (
            "optional-skills/research/osint-investigation/SKILL.md",
            "COURTLISTENER_TOKEN",
        ),
        (
            "optional-skills/research/osint-investigation/SKILL.md",
            "OPENCORPORATES_API_TOKEN",
        ),
        (
            "optional-skills/research/osint-investigation/SKILL.md",
            "SENATE_LDA_TOKEN",
        ),
        (
            "optional-skills/productivity/siyuan/SKILL.md",
            "SIYUAN_URL",
        ),
        (
            "skills/productivity/teams-meeting-pipeline/SKILL.md",
            "MSGRAPH_WEBHOOK_CLIENT_STATE",
        ),
    ],
)
def test_optional_capabilities_do_not_block_skill_loading(
    relative_path: str,
    environment_variable: str,
) -> None:
    entry = _environment_metadata(relative_path)[environment_variable]

    assert entry["optional"] is True
    assert entry.get("required_for")


@pytest.mark.parametrize(
    ("relative_path", "environment_variable"),
    [
        ("optional-skills/productivity/canvas/SKILL.md", "CANVAS_BASE_URL"),
        (
            "skills/productivity/teams-meeting-pipeline/SKILL.md",
            "MSGRAPH_TENANT_ID",
        ),
        (
            "skills/productivity/teams-meeting-pipeline/SKILL.md",
            "MSGRAPH_CLIENT_ID",
        ),
    ],
)
def test_non_secret_settings_do_not_enter_secret_capture_metadata(
    relative_path: str,
    environment_variable: str,
) -> None:
    assert environment_variable not in _environment_metadata(relative_path)


def test_non_secret_guidance_keeps_values_in_local_configuration() -> None:
    expected_guidance = {
        "optional-skills/blockchain/hyperliquid/SKILL.md": (
            "`HYPERLIQUID_USER_ADDRESS` is set in `${HERMES_HOME:-~/.hermes}/.env`."
        ),
        "optional-skills/productivity/canvas/SKILL.md": (
            "Keep the non-secret Canvas base URL in `${HERMES_HOME:-~/.hermes}/.env`:"
        ),
        "optional-skills/productivity/telephony/SKILL.md": (
            "The helper only writes `${HERMES_HOME:-~/.hermes}/.env`; "
            "it does not persist values to Bitwarden Secrets Manager."
        ),
        "skills/productivity/notion/SKILL.md": (
            "Keep `NOTION_KEYRING=0` in your shell profile or "
            "`${HERMES_HOME:-~/.hermes}/.env`; it is a non-secret setting."
        ),
        "skills/productivity/teams-meeting-pipeline/SKILL.md": (
            "Keep the non-secret tenant and client IDs in "
            "`${HERMES_HOME:-~/.hermes}/.env`:"
        ),
    }

    for relative_path, expected in expected_guidance.items():
        assert expected in _read(relative_path)


def test_agentmail_uses_environment_interpolation_in_mcp_config() -> None:
    skill = _read("optional-skills/email/agentmail/SKILL.md")

    assert 'AGENTMAIL_API_KEY: "${AGENTMAIL_API_KEY}"' in skill
    assert "paste your actual key" not in skill
    assert "MCP env vars are not expanded" not in skill


def test_optional_1password_service_account_token_does_not_promise_a_prompt() -> None:
    skill = _read("optional-skills/security/1password/SKILL.md")

    assert "the skill will prompt for this on first load" not in skill
