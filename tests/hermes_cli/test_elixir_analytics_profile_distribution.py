from pathlib import Path

import yaml

from hermes_cli.profile_distribution import DistributionManifest
from hermes_cli.tools_config import _get_platform_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "profile-distributions" / "elixir-analytics"
SKILL_PATH = DIST_DIR / "skills" / "elixir-analytics" / "SKILL.md"


def _load_yaml(name: str):
    return yaml.safe_load((DIST_DIR / name).read_text(encoding="utf-8"))


def test_elixir_analytics_distribution_manifest_is_installable():
    manifest = DistributionManifest.from_dict(_load_yaml("distribution.yaml"))

    assert manifest.name == "elixir-analytics"
    assert manifest.description == "Slack-first Elixir analytics agent."
    assert "SOUL.md" in manifest.owned_paths()
    assert "ROADMAP.md" in manifest.owned_paths()
    assert "config.yaml" in manifest.owned_paths()

    required_env = {req.name for req in manifest.env_requires if req.required}
    assert {"SLACK_APP_TOKEN", "SLACK_BOT_TOKEN"}.issubset(required_env)


def test_elixir_analytics_profile_keeps_slack_self_improvement_tools():
    config = _load_yaml("config.yaml")
    slack_toolsets = _get_platform_tools(config, "slack")

    assert {
        "web",
        "terminal",
        "file",
        "skills",
        "todo",
        "memory",
        "session_search",
        "clarify",
        "code_execution",
        "cronjob",
        "messaging",
    }.issubset(slack_toolsets)


def test_elixir_analytics_profile_configures_inference_provider():
    config = _load_yaml("config.yaml")

    assert config["model"]["provider"] == "openai-codex"
    assert config["model"]["default"] == "gpt-5.5"


def test_elixir_analytics_profile_uses_smart_approvals_without_restricting_slack_code():
    config = _load_yaml("config.yaml")
    slack_toolsets = _get_platform_tools(config, "slack")

    assert config["approvals"]["mode"] == "smart"
    assert "code_execution" in slack_toolsets


def test_elixir_analytics_profile_excludes_unrelated_fluff():
    config = _load_yaml("config.yaml")
    slack_toolsets = _get_platform_tools(config, "slack")

    assert {
        "vision",
        "image_gen",
        "video",
        "video_gen",
        "tts",
        "browser",
        "homeassistant",
        "spotify",
        "discord",
        "discord_admin",
        "computer_use",
        "delegation",
    }.isdisjoint(slack_toolsets)


def test_elixir_analytics_distribution_ships_analytics_skill():
    body = SKILL_PATH.read_text(encoding="utf-8")
    frontmatter = body.split("---", 2)[1]

    assert "name: elixir-analytics" in frontmatter
    assert "description: Answer Elixir analytics questions." in frontmatter
    assert "/Users/ritik/Coding/claude-analytics/GLOSSARY.md" in body
    assert "/Users/ritik/Coding/claude-analytics/src/lib/analytics/metric-contracts.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/docs/ad-hoc-query-protocol.md" in body
    assert "/Users/ritik/Coding/claude-analytics/docs/source-change-workflow.md" in body
    assert "/Users/ritik/Coding/claude-analytics/docs/self-improvement-loop.md" in body
    assert "/Users/ritik/Coding/claude-analytics/docs/ops-readiness.md" in body
    assert "/Users/ritik/Coding/claude-analytics/docs/slack-smoke-suite.md" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/plan-analytics-question.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/plan-source-change.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/plan-self-improvement.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/check-ops-readiness.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/run-analytics-smoke-suite.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/run-saved-query-topic.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/run-ad-hoc-query.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/run-posthog-query.ts" in body
    assert "Plan every Slack analytics question" in body
    assert "Plan every definition/glossary/schema/dashboard source-change request" in body
    assert "Plan self-improvement reviews" in body
    assert "Check production readiness" in body
    assert "Run `scripts/run-analytics-smoke-suite.ts`" in body
    assert "`saved_topic`: run `recommendedCommand`" in body
    assert "`dashboard_candidate`" in body
    assert "`sourceChangeRequest`" in body
    assert "`blockers`" in body
    assert "`ops_readiness_contract`" in body
    assert "`posthog_ad_hoc`" in body
    assert "`requiresClarification`" in body
    assert "`requiredFiles`" in body
    assert "`prTitle`" in body
    assert "card-gtv-weekly" in body
    assert "which users spent on Swiggy this week" in body
    assert "how many app active users this week" in body
    assert "Do not combine Supabase and" in body
    assert "PostHog active-user definitions" in body
    assert "`active_app_user`" in body
    assert "loadDotenv({ path: \".env.local\"" in body
    assert "dynamic import" in body
    assert "`logEntry` in stdout" in body
    assert "`dashboardUrl`" in body
    assert "Ask clarification" in body
    assert "--env-file /Users/ritik/.hermes/profiles/elixir-analytics/.env" in body


def test_elixir_analytics_distribution_ships_product_roadmap():
    body = (DIST_DIR / "ROADMAP.md").read_text(encoding="utf-8")

    assert "Slack app: macros" in body
    assert "Production Gates" in body
    assert "Slack E2E Checklist" in body
    assert "show GTV last 30 days by week" in body
    assert "which users spent on Swiggy this week?" in body
    assert "active users this week" in body
    assert "how many app active users this week?" in body
    assert "delete from profiles" in body
    assert "ANALYTICS_DATABASE_URL" in body
    assert "codex/mock-single-dashboard" in body
