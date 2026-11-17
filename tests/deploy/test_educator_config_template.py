"""deploy/educator/config.yaml.template, rendered with values.example.yaml,
run through the real readers the gateway uses: boundary config, Notion write
guard, gated posting, autonomy, cost budget, github_pr policy, RBAC, pricing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

DEPLOY = Path(__file__).resolve().parents[2] / "deploy" / "educator"

DESIGN_TOOLSETS = {
    # Hermes tools
    "web", "browser", "file", "vision", "session_search", "memory", "skills", "todo", "clarify",
    # fork tools (zendesk + github each come as a read and a write toolset)
    "notion", "jira", "zendesk", "zendesk_write", "github_pr", "github_pr_write",
    "slack", "slack_post", "ownership", "video_frames", "pdf_pages", "cost_budget",
}
FORBIDDEN = {"terminal", "code_execution", "jira_write", "messaging", "cronjob", "delegation",
             "kanban", "webflow_assets"}


def _render_module():
    spec = importlib.util.spec_from_file_location("educator_render", DEPLOY / "render_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _values() -> dict:
    return yaml.safe_load((DEPLOY / "values.example.yaml").read_text())


def _template() -> str:
    return (DEPLOY / "config.yaml.template").read_text()


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> dict:
    text = _render_module().render(_template(), _values())
    home = tmp_path / "educator"
    home.mkdir()
    (home / "config.yaml").write_text(text)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import boundary_config
    boundary_config.clear_cache()
    return yaml.safe_load(text)


# ── template mechanics ───────────────────────────────────────────────────

def test_template_is_yaml_and_values_cover_every_placeholder():
    render = _render_module()
    yaml.safe_load(_template())  # every placeholder is quoted
    assert render.placeholders(_template()) == set(_values())


def test_render_refuses_missing_or_unknown_values():
    render = _render_module()
    values = _values()
    values.pop("HOME_CHANNEL_ID")
    with pytest.raises(ValueError, match="HOME_CHANNEL_ID"):
        render.render(_template(), values)
    with pytest.raises(ValueError, match="TYPO"):
        render.render(_template(), {**_values(), "TYPO": "x"})
    with pytest.raises(ValueError, match="double quote"):
        render.render(_template(), {**_values(), "ADMIN_NAME": 'a"b'})


# ── boundary ─────────────────────────────────────────────────────────────

def test_no_boundary_config_errors(cfg):
    from agent.boundary_config import boundary_config_errors
    assert boundary_config_errors(cfg) == []


def test_strict_reader_parses_the_rendered_file(cfg):
    from agent.boundary_config import is_unreadable, read_boundary_config
    loaded = read_boundary_config()
    assert not is_unreadable(loaded)
    assert loaded["agent_write_guard"] is True


def test_notion_write_guard_is_enforcing_and_valid(cfg):
    from tools import notion_write_guard as g
    assert g.is_enforcing(cfg)
    guard = g.load_guard(cfg)
    assert guard is not None and guard.error == ""
    assert guard.kits == {"6f1611b4c2044944a0c7026e67604c77", "c7436811d26d4a738c6069437f3fcb16"}
    assert guard.prds == {"2daff8496e1b4e4f960aa07acfbc2a1b"}
    assert guard.kit_prd_relation == "PRD"
    assert g.autonomy_precondition(cfg) is None


def test_gated_release_updates_settings(cfg):
    from tools.release_update_post import settings
    s = settings(cfg)
    assert s.config_error == ""
    assert s.type_value == "Slack post"  # the real Release Assets option
    assert s.prd_path == ("Release", "PRD")
    assert s.approved == {"Approved"}


def test_autonomy_levels_are_v1(cfg):
    from agent import autonomy
    expected = {"post_release_update": "signoff", "zendesk_create_draft": "signoff",
                "zendesk_publish": "off", "advance_asset_status": "off",
                "ideas_accept_own": "off", "merge_skill_pr": "off"}
    assert {a: autonomy.resolve_level(a, cfg) for a in expected} == expected
    channel, wait, _wh = autonomy.veto_settings(cfg)
    assert channel == _values()["HOME_CHANNEL_ID"] and wait == 4


def test_posting_policy(cfg):
    from agent.slack_post_policy import post_policy
    v = _values()
    p = post_policy(cfg)
    assert not p.error
    assert p.allowed == {v["HOME_CHANNEL_ID"], v["RELEASE_UPDATES_CHANNEL_ID"]}
    assert p.gated == {v["RELEASE_UPDATES_CHANNEL_ID"]}


def test_shadow_week_variant_reports_only_the_expected_line(cfg):
    from agent.boundary_config import boundary_config_errors
    v = _values()
    cfg["slack"]["post_allowed_channels"] = [v["HOME_CHANNEL_ID"]]
    errors = boundary_config_errors(cfg)
    assert len(errors) == 1 and "post_gated_channels lists channel(s) missing" in errors[0]


def test_zendesk_settings(cfg):
    from tools.zendesk_tool import _zendesk_settings
    z = _zendesk_settings(cfg)
    assert z["data_source"] == "c7436811-d26d-4a73-8c60-69437f3fcb16"
    assert int(z["default_permission_group_id"]) == 1234567


# ── budget, github, pricing ──────────────────────────────────────────────

def test_cost_budget_is_fail_closed_with_release_watch_essential(cfg):
    from agent.cost_budget import is_discretionary, load_budget_config
    b = load_budget_config(cfg)
    assert b is not None and b.warnings == ()
    assert (b.daily_usd, b.monthly_usd, b.per_run_usd) == (5.0, 100.0, 1.5)
    assert b.alert_channel == f"slack:{_values()['HOME_CHANNEL_ID']}"
    rw = _values()["RELEASE_WATCH_JOB_ID"]
    assert not is_discretionary({"id": rw, "name": "release-watch"}, b)
    assert is_discretionary({"id": "fedcba987654", "name": "explore"}, b)
    assert is_discretionary({"id": "aaaaaaaaaaaa", "name": "something-the-agent-made"}, b)


def test_github_pr_policy(cfg, monkeypatch):
    from tools import github_pr_tool
    monkeypatch.setattr(github_pr_tool, "_raw_config", lambda: cfg)
    repos, err = github_pr_tool._load_repo_policy()
    assert err is None
    assert repos["everafterai/product-skills"]["branch_prefix"] == "educator/"
    assert repos["everafterai/ea-monorepo"]["actions"] == {"read_pr", "read_commits"}


def test_pricing_override_applies_to_the_model(cfg):
    from agent.usage_pricing import get_pricing_entry
    entry = get_pricing_entry(cfg["model"]["default"], provider=cfg["model"]["provider"])
    assert entry is not None and entry.source == "user_override"


# ── tools + RBAC ─────────────────────────────────────────────────────────

def test_platform_toolsets_are_exactly_the_design_set_and_all_exist(cfg):
    offered = set(cfg["platform_toolsets"]["slack"])
    assert offered == DESIGN_TOOLSETS
    assert not offered & FORBIDDEN
    assert not any(t.startswith("mcp") for t in offered)
    import model_tools  # noqa: F401  (registers every tool)
    from tools.registry import registry
    registered = {e.toolset for e in registry._tools.values()}
    assert offered <= registered, offered - registered


def test_rbac_roles(cfg):
    from gateway.tool_access import policy_from_extra
    v = _values()
    policy = policy_from_extra(cfg["slack"])
    assert policy.enabled
    assert policy.role_for(v["ADMIN_SLACK_USER_ID"]) == "admin"
    assert policy.role_for(v["PRODUCT_TEAM_SLACK_USER_ID"]) == "educator_user"
    offered = frozenset(cfg["platform_toolsets"]["slack"])
    assert policy.allowed_toolsets(v["PRODUCT_TEAM_SLACK_USER_ID"], offered) == offered
    assert policy.allowed_toolsets("U0STRANGER", offered) == frozenset()
    assert not policy.is_authorized("U0STRANGER", None)
    assert cfg["slack"]["allow_admin_from"] == [v["ADMIN_SLACK_USER_ID"]]


def test_slack_behaviour_keys(cfg):
    s = cfg["slack"]
    assert s["free_response_channels"] == _values()["HOME_CHANNEL_ID"]
    assert s["require_mention"] is True and s["wake_in_bot_authored_threads"] is True
    from plugins.platforms.slack.adapter import _PROFILE_SETTING_KEYS
    for key in ("free_response_channels", "require_mention", "wake_in_bot_authored_threads",
                "home_channel_prompt"):
        assert key in _PROFILE_SETTING_KEYS  # honoured per profile under multiplex


def test_skills_and_cron(cfg):
    home = _values()["EDUCATOR_HOME"]
    assert cfg["skills"]["external_dirs"] == [
        f"{home}/workspace/checkouts/product-skills/plugins/release-comms/skills",
        f"{home}/workspace/checkouts/product-skills/plugins/product-education/skills",
    ]
    assert cfg["cron"]["allow_agent_shell"] is False
    assert cfg["agent_write_guard"] is True
