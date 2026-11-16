"""Default-profile MCP servers never reach a secondary profile (4c)."""
from __future__ import annotations

import json

import pytest

from agent import secret_scope as ss
from gateway import multiplex_guard as mg
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    edu = root / "profiles" / "educator"
    edu.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "mcp_servers:\n  webflow: {command: x}\n  stripe: {command: y}\n", encoding="utf-8"
    )
    (edu / "config.yaml").write_text("mcp_servers:\n  zendesk_own: {command: z}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    from tools.registry import registry

    registry.register_toolset_alias("webflow", "mcp-webflow")
    ss.set_multiplex_active(False)
    yield root, edu
    ss.set_multiplex_active(False)


class _In:
    def __init__(self, home):
        self.home = home

    def __enter__(self):
        self.tok = set_hermes_home_override(str(self.home))

    def __exit__(self, *a):
        reset_hermes_home_override(self.tok)


TOOLSETS = ["file", "mcp-webflow", "webflow", "mcp-stripe", "mcp-zendesk_own", "web"]


def test_secondary_drops_default_mcp_keeps_own(homes):
    _, edu = homes
    ss.set_multiplex_active(True)
    with _In(edu):
        assert mg.is_secondary_profile_context() is True
        assert mg.filter_foreign_mcp_toolsets(TOOLSETS) == ["file", "mcp-zendesk_own", "web"]


def test_default_profile_and_multiplex_off_unchanged(homes):
    root, edu = homes
    ss.set_multiplex_active(True)
    with _In(root):
        assert mg.is_secondary_profile_context() is False
        assert mg.filter_foreign_mcp_toolsets(TOOLSETS) == TOOLSETS
    ss.set_multiplex_active(False)
    with _In(edu):
        assert mg.filter_foreign_mcp_toolsets(TOOLSETS) == TOOLSETS


def test_dispatch_backstop(homes):
    root, edu = homes
    ss.set_multiplex_active(True)
    with _In(edu):
        assert mg.foreign_mcp_denial("mcp__webflow__list_sites") is not None
        assert mg.foreign_mcp_denial("mcp__zendesk_own__draft") is None
        assert mg.foreign_mcp_denial("read_file") is None
    with _In(root):
        assert mg.foreign_mcp_denial("mcp__webflow__list_sites") is None
    ss.set_multiplex_active(False)
    with _In(edu):
        assert mg.foreign_mcp_denial("mcp__webflow__list_sites") is None


def test_backstop_fails_closed_on_internal_error(homes, monkeypatch):
    _, edu = homes
    ss.set_multiplex_active(True)

    def boom(*a, **k):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(mg, "profile_declared_mcp_servers", boom)
    with _In(edu):
        assert mg.foreign_mcp_denial("mcp__zendesk_own__draft") is not None
        assert mg.foreign_mcp_denial("read_file") is None


def test_handle_function_call_blocks_foreign_mcp(homes):
    """The model_tools backstop returns an error before the tool runs."""
    _, edu = homes
    import model_tools

    ss.set_multiplex_active(True)
    with _In(edu):
        out = model_tools.handle_function_call("mcp__webflow__list_sites", {}, task_id="t")
    assert "does not configure" in json.loads(out)["error"]


def test_resolver_applies_filter_after_rbac(homes, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    _, edu = homes
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {}
    import hermes_cli.tools_config as tc

    monkeypatch.setattr(tc, "_get_platform_tools", lambda cfg, key: set(TOOLSETS))
    ss.set_multiplex_active(True)
    src = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1", profile="educator")
    with _In(edu):
        out = runner._resolve_enabled_toolsets_for_source({}, src, "slack")
    assert "mcp-webflow" not in out and "webflow" not in out and "mcp-stripe" not in out
    assert "mcp-zendesk_own" in out


def test_cron_job_of_secondary_profile_drops_default_mcp(homes, monkeypatch):
    import cron.scheduler as sched

    root, edu = homes
    monkeypatch.setattr(sched, "_cron_enabled_toolsets_rbac_capped", lambda job, cfg: list(TOOLSETS))
    ss.set_multiplex_active(True)
    with _In(edu):
        out = sched._cron_enabled_toolsets_with_ceiling({"id": "j"}, {})
    assert out == ["file", "mcp-zendesk_own", "web"]
    with _In(root):
        assert sched._cron_enabled_toolsets_with_ceiling({"id": "j"}, {}) == TOOLSETS
    monkeypatch.setattr(sched, "_cron_enabled_toolsets_rbac_capped", lambda job, cfg: None)
    with _In(edu):
        expanded = sched._cron_enabled_toolsets_with_ceiling({"id": "j"}, {})
    assert expanded is not None and not any(t.startswith("mcp-webflow") for t in expanded)
    ss.set_multiplex_active(False)
    with _In(edu):
        assert sched._cron_enabled_toolsets_with_ceiling({"id": "j"}, {}) is None
