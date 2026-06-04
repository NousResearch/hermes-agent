import re
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
    assert "RELEASE_PACKAGING.md" in manifest.owned_paths()
    assert "config.yaml" in manifest.owned_paths()
    assert "profile_plugins" in manifest.owned_paths()

    required_env = {req.name for req in manifest.env_requires if req.required}
    assert {"SLACK_APP_TOKEN", "SLACK_BOT_TOKEN"}.issubset(required_env)


def test_elixir_analytics_profile_keeps_slack_self_improvement_tools():
    config = _load_yaml("config.yaml")
    slack_toolsets = _get_platform_tools(config, "slack")

    assert {
        "elixir-analytics-runner",
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


def test_elixir_analytics_profile_enables_runner_plugin():
    config = _load_yaml("config.yaml")

    assert "elixir-analytics-runner" in config["plugins"]["enabled"]


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
    assert "/Users/ritik/Coding/claude-analytics/scripts/check-source-change-scope.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/check-self-improvement-cadence.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/plan-self-improvement.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/check-ops-readiness.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/run-analytics-smoke-suite.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/check-slack-e2e-logs.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/run-analytics-question.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/run-saved-query-topic.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/run-ad-hoc-query.ts" in body
    assert "/Users/ritik/Coding/claude-analytics/scripts/run-posthog-query.ts" in body
    assert (
        "the first tool call must be\n"
        "   `elixir_analytics_runner` with mode `answer_question`"
    ) in body
    assert "Plan remaining Slack analytics questions" in body
    assert "Plan every definition/glossary/schema/dashboard source-change request" in body
    assert "`elixir_analytics_runner` mode `source_change_plan`" in body
    assert 'mode: "source_change_scope_check"' in body
    assert "`scripts/check-source-change-scope.ts`" in body
    assert "Check self-improvement cadence with `elixir_analytics_runner` mode" in body
    assert "`self_improvement_check`" in body
    assert "Plan self-improvement reviews with `elixir_analytics_runner` mode" in body
    assert "`self_improvement_plan`" in body
    assert "Check production readiness" in body
    assert "Run `scripts/run-analytics-smoke-suite.ts`" in body
    assert "Run `scripts/check-slack-e2e-logs.ts` after live Slack acceptance prompts" in body
    assert "`saved_topic`: run `recommendedCommand`" in body
    assert "`dashboard_candidate`" in body
    assert "`sourceChangeRequest`" in body
    assert "`blockers`" in body
    assert "`ops_readiness_contract`" in body
    assert "`posthog_ad_hoc`" in body
    assert "`requiresClarification`" in body
    assert "`requiredFiles`" in body
    assert "`prTitle`" in body
    assert "Mandatory First Call" in body
    assert "the first tool call must be\n`elixir_analytics_runner` with `mode: \"answer_question\"`" in body
    assert "Do not plan, inspect files, write SQL, run `execute_code`, edit\n" in body
    assert "Only if it returns `requires_model_request`" in body
    assert "Fast Slack Answer Path" in body
    assert "Prefer the `elixir_analytics_runner` tool" in body
    assert "use mode `answer_question` first" in body
    assert "exact raw Slack question" in body
    assert "Use a compact Slack row cap such as `max_rows: 25`" in body
    assert "runs `scripts/run-analytics-question.ts`" in body
    assert "If `answer_question` returns `requires_model_request`, use mode `plan`" in body
    assert "Call\n   `scripts/plan-analytics-question.ts` once" in body
    assert "Do not use `patch`, edit files, append `QUERY_LOG.md`" in body
    assert "Do not block a plain Slack data answer on source edits" in body
    assert "Do not use generic `execute_code` for the first pass" in body
    assert "Only after `answer_question` returns `requires_model_request`, call the\n" in body
    assert "deterministic planner" in body
    assert "card-gtv-weekly" in body
    assert "which users spent on Swiggy this week" in body
    assert "`answer_question` -> Supabase ad hoc payload -> direct Slack answer" in body
    assert "how many app active users this week" in body
    assert "`answer_question` -> PostHog query payload -> direct Slack answer" in body
    assert "Do not combine Supabase and" in body
    assert "PostHog active-user definitions" in body
    assert "`active_app_user`" in body
    assert "loadDotenv({ path: \".env.local\"" in body
    assert "dynamic import" in body
    assert "`logEntry` in stdout" in body
    assert "do not create a duplicate query number" in body
    assert "`dashboardUrl`" in body
    assert "Include a dashboard link in Slack answers" in body
    assert "default to\n    `https://analytics.joinelixir.club`" in body
    assert "Ask clarification" in body
    assert "--env-file /Users/ritik/.hermes/profiles/elixir-analytics/.env" in body
    assert (
        "scripts/check-ops-readiness.ts --current-branch '<branch>' --profile-home "
        "/Users/ritik/.hermes/profiles/elixir-analytics"
    ) in body
    assert "--gateway-hosted --slack-connected" not in body
    assert "--provider-authenticated openai-codex" in body
    assert "auth status openai-codex" in body
    assert "scripts/check-slack-e2e-logs.ts --gateway-log" in body
    assert "route, timing, call-count, and safe runner telemetry" in body
    assert "`source_change_workflow` for definition\nchanges that must become PR work" in body
    assert 'mode: "source_change_plan"' in body
    assert 'mode: "self_improvement_check"' in body
    assert 'mode: "self_improvement_plan"' in body
    assert "short redirect is acceptable" not in body


def test_elixir_analytics_distribution_ships_product_roadmap():
    body = (DIST_DIR / "ROADMAP.md").read_text(encoding="utf-8")

    assert "Slack app: macros" in body
    assert "Current Status Snapshot" in body
    assert "live ops-readiness check can load the installed Hermes profile" in body
    assert "Current live ops-readiness status is `blocked`" in body
    assert "ANALYTICS_DATABASE_URL" in body
    assert "saved-query data unavailable" in body
    assert "Production Gates" in body
    assert "Slack E2E Checklist" in body
    assert "Next Execution Milestone" in body
    assert "Milestone 11A: finish real Slack E2E acceptance." in body
    assert "show GTV last 30 days by week" in body
    assert "which users spent on Swiggy this week?" in body
    assert "metadata and dashboard link" in body
    assert "temporary dashboard link" in body
    assert "no third-party URL shorteners" in body
    assert "dashboard link needs fresh verification after Vercel env fix" in body
    assert "Passed live with direct link; latency failed at 418.3s / 33 calls" in body
    assert "Passed live at 13.6s / 3 calls" in body
    assert "Passed live; latency failed at 266.6s / 20 calls" in body
    assert "Passed live at 40.2s / 2 calls" in body
    assert "Partially complete on\n2026-06-04" in body
    assert "Milestone 14A: make Supabase ad hoc answers fast enough for Slack." in body
    assert "Milestone 14B: make PostHog ad hoc answers fast enough for Slack." in body
    assert "active users this week" in body
    assert "how many app active users this week?" in body
    assert "delete from profiles" in body
    assert "ANALYTICS_DATABASE_URL" in body
    assert "--provider-authenticated openai-codex" in body


def test_elixir_analytics_soul_requires_dashboard_links_for_data_answers():
    body = (DIST_DIR / "SOUL.md").read_text(encoding="utf-8")

    assert "dashboard or temporary visualization links for every runnable data answer" in body
    assert "include a dashboard link" in body
    assert "`ANALYTICS_BASE_URL`" in body
    assert "Do not use third-party URL shorteners" in body
    assert re.search(
        r"Do not finalize a Slack ad hoc data answer from manual `execute_code` output\s+alone",
        body,
    )


def test_elixir_analytics_distribution_ships_runner_plugin():
    plugin_dir = DIST_DIR / "profile_plugins" / "elixir-analytics-runner"
    manifest = yaml.safe_load((plugin_dir / "plugin.yaml").read_text(encoding="utf-8"))
    body = (plugin_dir / "__init__.py").read_text(encoding="utf-8")

    assert manifest["name"] == "elixir-analytics-runner"
    assert "elixir_analytics_runner" in manifest["provides_tools"]
    assert "scripts/plan-analytics-question.ts" in body
    assert "scripts/run-analytics-question.ts" in body
    assert "scripts/run-saved-query-topic.ts" in body
    assert "scripts/run-ad-hoc-query.ts" in body
    assert "scripts/run-posthog-query.ts" in body
    assert "scripts/plan-source-change.ts" in body
    assert "scripts/check-self-improvement-cadence.ts" in body
    assert "scripts/plan-self-improvement.ts" in body
