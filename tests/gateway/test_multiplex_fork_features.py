"""Acceptance: two fully separate Slack bots in one multiplexed gateway.

One real ``GatewayRunner`` (``gateway.multiplex_profiles: true``) serves the
``default`` profile (HERMES root) and a secondary ``educator`` profile, each
with its own Slack app (distinct tokens, mocked Slack clients), ``.env``,
``config.yaml``, RBAC roles, Slack flags, toolsets and MCP servers. One
inbound message per bot runs through the real adapter → message handler →
``_handle_message`` → ``_run_agent`` (profile scope) path; only the model run
itself (``_run_agent_inner``) is replaced by a probe that records, from
INSIDE the turn, everything the acceptance table names.

Every row is asserted in both directions (default → educator and educator →
default). The same user id has a different role in each profile, and every
credential has a different value per profile plus a third value in
``os.environ`` — so any leak shows up as the wrong value.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_slack_wake_bot_authored_threads_optin import _ensure_slack_mock

_ensure_slack_mock()

import plugins.platforms.slack.adapter as slack_mod  # noqa: E402

slack_mod.SLACK_AVAILABLE = True

from agent import secret_scope as ss  # noqa: E402
from gateway import tool_access  # noqa: E402
from gateway.config import GatewayConfig, Platform, PlatformConfig  # noqa: E402
from gateway.platforms.base import MessageEvent, MessageType  # noqa: E402

PROFILES = {
    "default": {
        "env": {
            "SLACK_BOT_TOKEN": "xoxb-default",
            "SLACK_APP_TOKEN": "xapp-default",
            "JIRA_BASE_URL": "https://default.atlassian.net",
            "JIRA_EMAIL": "default@x",
            "JIRA_API_TOKEN": "jira-default",
            "NOTION_API_KEY": "notion-default",
            "WEBFLOW_API_TOKEN": "wf-default",
            "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON": '{"who": "default"}',
        },
        "config": """\
slack:
  user_roles:
    USHARED: readonly
    UDEFADMIN: admin
  allow_admin_from: [UDEFADMIN]
  quiet_channels: [CDEFQUIET]
  ignored_channels: [CDEFIGNORED]
  channel_models:
    CSHARED: default-model
platform_toolsets:
  slack: [file, web, jira, notion, slack, mcp-webflow, session_search]
mcp_servers:
  webflow: {command: "true", enabled: false}
""",
    },
    "educator": {
        "env": {
            "SLACK_BOT_TOKEN": "xoxb-educator",
            "SLACK_APP_TOKEN": "xapp-educator",
            "JIRA_BASE_URL": "https://edu.atlassian.net",
            "JIRA_EMAIL": "edu@x",
            "JIRA_API_TOKEN": "jira-educator",
            "NOTION_API_KEY": "notion-educator",
            "WEBFLOW_API_TOKEN": "wf-educator",
            "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON": '{"who": "educator"}',
        },
        "config": """\
slack:
  user_roles:
    USHARED: admin
    UEDUONLY: operator
  allow_admin_from: [UEDUADMIN]
  quiet_channels: [CEDUQUIET]
  ignored_channels: [CEDUIGNORED]
  channel_models:
    CSHARED: educator-model
platform_toolsets:
  slack: [file, web, jira, notion, slack, mcp-webflow, session_search]
""",
    },
}

# What the gateway process inherited (the default profile's .env loaded at
# startup) PLUS a stray value per key, so any os.environ read is detectable.
PROCESS_ENV = {
    "JIRA_BASE_URL": "https://process.atlassian.net",
    "JIRA_EMAIL": "process@x",
    "JIRA_API_TOKEN": "jira-process",
    "NOTION_API_KEY": "notion-process",
    "WEBFLOW_API_TOKEN": "wf-process",
    "SLACK_BOT_TOKEN": "xoxb-process",
    "SLACK_QUIET_CHANNELS": "CPROCESSQUIET",
    "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON": '{"who": "process"}',
}


@pytest.fixture()
def gateway(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    homes = {"default": root, "educator": root / "profiles" / "educator"}
    for name, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        (home / ".env").write_text(
            "".join(f"{k}={v}\n" for k, v in PROFILES[name]["env"].items()), encoding="utf-8"
        )
        (home / "config.yaml").write_text(PROFILES[name]["config"], encoding="utf-8")
        (home / "SOUL.md").write_text(f"I am {name}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    for k, v in PROCESS_ENV.items():
        monkeypatch.setenv(k, v)
    tool_access._config_cache.clear()
    from agent import multiplex_file_guard

    multiplex_file_guard.set_served_profile_homes(list(homes.values()))
    yield homes
    ss.set_multiplex_active(False)
    multiplex_file_guard.set_served_profile_homes([])
    tool_access._config_cache.clear()


@contextmanager
def _scope(mapping):
    tok = ss.set_secret_scope(mapping)
    try:
        yield
    finally:
        ss.reset_secret_scope(tok)


def _slack_adapter(name, homes):
    """Build like the gateway does: the profile's YAML hook + its scope."""
    import yaml

    env = PROFILES[name]["env"]
    slack_yaml = yaml.safe_load(PROFILES[name]["config"])["slack"]
    with _scope(dict(env)):
        seeded = slack_mod._apply_yaml_config({}, dict(slack_yaml)) or {}
        extra = dict(seeded)
        extra.update(
            {k: slack_yaml[k] for k in ("user_roles", "allow_admin_from", "channel_models")}
        )
        adapter = slack_mod.SlackAdapter(
            PlatformConfig(enabled=True, token=env["SLACK_BOT_TOKEN"], extra=extra)
        )
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1790000000.000200"})
    client.reactions_add = AsyncMock(return_value={"ok": True})
    client.reactions_remove = AsyncMock(return_value={"ok": True})
    adapter._app = MagicMock()
    adapter._app.client = client
    adapter._bot_user_id = f"UBOT{name.upper()}"
    adapter._running = True
    return adapter


def _probe(other_home_of):
    """Stand-in for the model turn: record what the turn would see."""
    seen = {}

    async def _run_agent_inner(message, context_prompt, history, source, session_id, **kw):
        from agent.data_access_audit import _log_path as audit_log_path
        from agent import automation_ownership as ao
        from agent.file_safety import get_read_block_error, is_write_denied
        from gateway.multiplex_guard import filter_foreign_mcp_toolsets
        from gateway.run import _load_gateway_config
        from hermes_constants import get_hermes_home
        from plugins.google_drive_sa import client as gd
        from tools.jira_api_tool import _jira_creds
        from tools.notion_api_tool import _ntn_env
        from tools.slack_react_tool import _resolve_slack_token
        from tools.webflow_asset_tool import _webflow_token
        from hermes_state import _default_db_path

        profile = source.profile or "default"
        home = get_hermes_home()
        other = other_home_of[profile]
        runner = kw["_runner"]
        user_cfg = _load_gateway_config()
        policy = tool_access.policy_for_platform("slack")
        seen[profile] = {
            "home": str(home),
            "grant": sorted(policy.grant_for(source.user_id, source.chat_id) or []),
            "toolsets": runner._resolve_enabled_toolsets_for_source(user_cfg, source, "slack"),
            "mcp_filter": filter_foreign_mcp_toolsets(["mcp-webflow", "file"]),
            "jira": _jira_creds(),
            "notion": _ntn_env().get("NOTION_API_TOKEN"),
            "webflow": _webflow_token(),
            "slack_token": _resolve_slack_token(),
            "drive_sa": gd._env("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"),
            "soul": (home / "SOUL.md").read_text(encoding="utf-8"),
            "read_other_soul": get_read_block_error(str(other / "SOUL.md")),
            "write_other_config": is_write_denied(str(other / "config.yaml")),
            "read_own_soul": get_read_block_error(str(home / "SOUL.md")),
            "audit_log": str(audit_log_path()),
            "state_db": str(_default_db_path()),
            "ownership_registry": str(ao._registry_path()),
            "slash_admins": sorted(
                getattr(
                    runner._gateway_config_for_source(source).platforms.get(Platform.SLACK),
                    "extra",
                    {},
                ).get("allow_admin_from") or []
            ),
            "secret_scope_jira": ss.get_secret("JIRA_API_TOKEN"),
        }
        return {
            "final_response": f"reply from {profile}",
            "messages": [],
            "api_calls": 1,
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }

    return seen, _run_agent_inner


async def _drive(runner, adapter, *, user, chat, text, expect, timeout_s=30):
    source = adapter.build_source(
        chat_id=chat, chat_type="group", user_id=user, message_id="1790000000.000100"
    )
    event = MessageEvent(
        text=text, message_type=MessageType.TEXT, source=source, message_id="1790000000.000100"
    )
    event.directly_addressed = True
    await adapter.handle_message(event)
    for _ in range(int(timeout_s * 20)):  # first turn imports a lot
        if any(
            expect in str(c.kwargs.get("text", ""))
            for c in adapter._app.client.chat_postMessage.await_args_list
        ):
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.1)


@pytest.fixture()
def runner_and_adapters(gateway, monkeypatch):
    import gateway.run as gateway_run

    homes = gateway
    runner = gateway_run.GatewayRunner(GatewayConfig(multiplex_profiles=True))
    assert ss.is_multiplex_active()
    default = _slack_adapter("default", homes)
    edu = _slack_adapter("educator", homes)
    for a in (default, edu):
        a.gateway_runner = runner
    runner.adapters = {Platform.SLACK: default}
    runner._profile_adapters = {"educator": {Platform.SLACK: edu}}
    default.set_message_handler(runner._primary_message_handler())
    default.set_session_store(runner.session_store)
    default.set_busy_session_handler(runner._handle_active_session_busy_message)
    default.set_authorization_check(runner._make_adapter_auth_check(Platform.SLACK))
    runner._configure_profile_adapter(edu, "educator", Platform.SLACK)

    seen, probe = _probe({"default": homes["educator"], "educator": homes["default"]})

    async def _inner(*a, **kw):
        return await probe(*a, _runner=runner, **kw)

    runner._run_agent_inner = _inner
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    return runner, default, edu, seen, homes


@pytest.mark.asyncio
async def test_two_slack_bots_are_fully_separate(runner_and_adapters):
    runner, default, edu, seen, homes = runner_and_adapters

    await _drive(
        runner, default, user="USHARED", chat="CSHARED", text="<@UBOTDEFAULT> hi",
        expect="reply from default",
    )
    await _drive(
        runner, edu, user="USHARED", chat="CSHARED", text="<@UBOTEDUCATOR> hi",
        expect="reply from educator",
    )

    assert set(seen) == {"default", "educator"}, seen
    d, e = seen["default"], seen["educator"]

    # Identity — each bot replies through its own Slack client only.
    d_posts = [c.kwargs.get("text", "") for c in default._app.client.chat_postMessage.await_args_list]
    e_posts = [c.kwargs.get("text", "") for c in edu._app.client.chat_postMessage.await_args_list]
    assert any("reply from default" in t for t in d_posts), d_posts
    assert any("reply from educator" in t for t in e_posts), e_posts
    assert not any("educator" in t for t in d_posts)
    assert not any("reply from default" in t for t in e_posts)

    # State — home, SOUL, audit log and ownership registry are per profile.
    assert d["home"] == str(homes["default"]) and e["home"] == str(homes["educator"])
    assert d["soul"] == "I am default\n" and e["soul"] == "I am educator\n"
    assert d["audit_log"].startswith(str(homes["default"]))
    assert e["audit_log"].startswith(str(homes["educator"]))
    assert e["ownership_registry"].startswith(str(homes["educator"]))
    # Cost — each profile's sessions (and so `hermes costs`) use its own store.
    assert d["state_db"] == str(homes["default"] / "state.db")
    assert e["state_db"] == str(homes["educator"] / "state.db")

    # Permissions — USHARED is readonly on the default bot, admin on educator.
    assert "*" not in d["grant"] and "*" in e["grant"]
    assert d["slash_admins"] == ["UDEFADMIN"] and e["slash_admins"] == ["UEDUADMIN"]

    # Credentials — each profile's own .env, never os.environ.
    assert d["jira"] == ("https://default.atlassian.net", "default@x", "jira-default")
    assert e["jira"] == ("https://edu.atlassian.net", "edu@x", "jira-educator")
    assert (d["notion"], e["notion"]) == ("notion-default", "notion-educator")
    assert (d["webflow"], e["webflow"]) == ("wf-default", "wf-educator")
    assert (d["slack_token"], e["slack_token"]) == ("xoxb-default", "xoxb-educator")
    assert d["drive_sa"] == '{"who": "default"}' and e["drive_sa"] == '{"who": "educator"}'
    assert (d["secret_scope_jira"], e["secret_scope_jira"]) == ("jira-default", "jira-educator")

    # Tools — the default's MCP server is offered only to the default.
    assert e["mcp_filter"] == ["file"] and d["mcp_filter"] == ["mcp-webflow", "file"]
    assert not any(t.startswith("mcp-") for t in e["toolsets"])

    # Files — neither agent can read or write the other's home.
    assert d["read_other_soul"] and e["read_other_soul"]
    assert d["write_other_config"] and e["write_other_config"]
    assert d["read_own_soul"] is None and e["read_own_soul"] is None


def test_slack_behaviour_per_bot(runner_and_adapters):
    runner, default, edu, _seen, _homes = runner_and_adapters
    from gateway.session import SessionSource

    assert default._slack_quiet_channels() == {"CDEFQUIET"}
    assert edu._slack_quiet_channels() == {"CEDUQUIET"}

    def src(adapter, chat):
        source = adapter.build_source(chat_id=chat, chat_type="group", user_id="USHARED")
        if adapter is edu:
            source.profile = "educator"  # stamped by the profile message handler
        return source

    assert runner._slack_source_ignored(src(edu, "CEDUIGNORED"))
    assert not runner._slack_source_ignored(src(edu, "CDEFIGNORED"))
    assert runner._slack_source_ignored(src(default, "CDEFIGNORED"))
    assert not runner._slack_source_ignored(src(default, "CEDUIGNORED"))
    assert runner._multiplex_adapter_for(src(edu, "C1"), Platform.SLACK) is edu
    assert runner._multiplex_adapter_for(src(default, "C1"), Platform.SLACK) is default


@pytest.mark.asyncio
async def test_message_gate_uses_the_owning_profiles_roles(runner_and_adapters):
    """A user with a role on one bot only is denied by the other bot."""
    runner, default, edu, seen, _homes = runner_and_adapters

    # UDEFADMIN is admin on the default bot only; UEDUONLY on educator only.
    await _drive(
        runner, edu, user="UDEFADMIN", chat="CSHARED", text="<@UBOTEDUCATOR> hi",
        expect="reply from educator", timeout_s=2,
    )
    await _drive(
        runner, default, user="UEDUONLY", chat="CSHARED", text="<@UBOTDEFAULT> hi",
        expect="reply from default", timeout_s=2,
    )
    assert seen == {}
    await _drive(
        runner, edu, user="UEDUONLY", chat="CSHARED", text="<@UBOTEDUCATOR> hi",
        expect="reply from educator",
    )
    assert set(seen) == {"educator"}


@pytest.mark.asyncio
async def test_single_profile_gateway_is_unchanged(gateway, monkeypatch):
    """Multiplex off: the same turn sees exactly today's values — tool
    credentials from os.environ, no profile file boundary, no MCP filter."""
    import gateway.run as gateway_run

    homes = gateway
    runner = gateway_run.GatewayRunner(GatewayConfig())
    assert not ss.is_multiplex_active()
    default = _slack_adapter("default", homes)
    default.gateway_runner = runner
    runner.adapters = {Platform.SLACK: default}
    runner._profile_adapters = {}
    default.set_message_handler(runner._primary_message_handler())
    default.set_session_store(runner.session_store)
    default.set_busy_session_handler(runner._handle_active_session_busy_message)
    default.set_authorization_check(runner._make_adapter_auth_check(Platform.SLACK))
    seen, probe = _probe({"default": homes["educator"]})

    async def _inner(*a, **kw):
        return await probe(*a, _runner=runner, **kw)

    runner._run_agent_inner = _inner
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    await _drive(
        runner, default, user="USHARED", chat="CSHARED", text="<@UBOTDEFAULT> hi",
        expect="reply from default",
    )
    d = seen["default"]
    assert d["jira"] == ("https://process.atlassian.net", "process@x", "jira-process")
    assert d["notion"] == "notion-process"
    assert d["webflow"] == "wf-process"
    assert d["drive_sa"] == '{"who": "process"}'
    assert d["mcp_filter"] == ["mcp-webflow", "file"]
    assert d["read_other_soul"] is None and d["write_other_config"] is False
    assert "*" not in d["grant"]
