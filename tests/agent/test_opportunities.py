"""Tests for proactive opportunities."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def opportunities_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    import agent.opportunities as opportunities

    importlib.reload(hermes_constants)
    importlib.reload(opportunities)
    return opportunities, home


def _add(store, key="k1", title="Learn thing", action_type="learn_skill"):
    return store.add_opportunity(
        title=title,
        description="desc",
        source="usage",
        action={"type": action_type, "request": "learn this workflow", "goal": "do the thing"},
        dedup_key=key,
    )


class TestOpportunityStore:
    def test_add_list_and_dedup(self, opportunities_env):
        store, _home = opportunities_env

        first = _add(store, key="dup")
        second = _add(store, key="dup")

        assert first is not None
        assert second is None
        assert len(store.list_pending()) == 1
        assert store.list_pending()[0]["status"] == "pending"

    def test_dismiss_latches(self, opportunities_env):
        store, _home = opportunities_env
        _add(store, key="dismiss-me")

        assert store.dismiss_opportunity("1") is True
        assert store.list_pending() == []
        assert _add(store, key="dismiss-me") is None

    def test_accept_learn_returns_normal_agent_seed(self, opportunities_env):
        store, _home = opportunities_env
        _add(store, key="learn")

        accepted = store.accept_opportunity("1")

        assert accepted is not None
        assert accepted["kind"] == "send"
        assert "skill_manage" in accepted["message"]
        assert "learn this workflow" in accepted["message"]
        assert store.list_pending() == []

    @pytest.mark.parametrize(
        "action_type, expected",
        [
            ("skill_bundle", "skill bundle"),
            ("profile", "specialist Hermes profile"),
            ("kanban_swarm", "multi-agent ecosystem"),
            ("cron_automation", "recurring automation"),
        ],
    )
    def test_accept_non_learn_opportunities_seed_agent_turn(self, opportunities_env, action_type, expected):
        store, _home = opportunities_env
        _add(store, key=action_type, action_type=action_type)

        accepted = store.accept_opportunity("1")

        assert accepted is not None
        assert expected in accepted["message"]

    def test_seed_starter_opportunities_is_idempotent(self, opportunities_env):
        store, _home = opportunities_env

        first = store.seed_starter_opportunities()
        second = store.seed_starter_opportunities()

        assert len(first) == 5
        assert second == []
        assert {r["action"]["type"] for r in first} == {
            "learn_skill",
            "skill_bundle",
            "profile",
            "kanban_swarm",
            "cron_automation",
        }


class TestUsageScanner:
    def test_scan_disabled_without_force(self, opportunities_env, tmp_path):
        store, _home = opportunities_env

        result = store.scan_recent_usage(config={"proactive": {"enabled": False}}, db_path=tmp_path / "missing.db")

        assert result["scanned"] is False
        assert result["reason"] == "disabled"

    def test_scan_repeated_user_messages_adds_learn_opportunity(self, opportunities_env, tmp_path):
        store, _home = opportunities_env
        from hermes_state import SessionDB

        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            db.create_session("s1", "cli")
            db.create_session("s2", "cli")
            db.append_message("s1", "user", "Review the authentication PR and find regression risks")
            db.append_message("s1", "assistant", "ok")
            db.append_message("s2", "user", "Review the authentication diff and find missing tests")
            db.append_message("s2", "assistant", "ok")
            db.append_message("s2", "user", "Review the authentication change and summarize blockers")
        finally:
            db.close()

        result = store.scan_recent_usage(
            force=False,
            config={
                "proactive": {
                    "enabled": True,
                    "min_repeats": 2,
                    "scan_recent_messages": 20,
                    "scan_interval_hours": 0,
                }
            },
            db_path=db_path,
        )

        assert result["scanned"] is True
        assert result["created"]
        pending = store.list_pending()
        assert any(r["action"]["type"] == "learn_skill" for r in pending)
        assert any("authentication" in " ".join(r.get("evidence", [])) for r in pending)

    def test_scan_domain_messages_adds_vertical_opportunity(self, opportunities_env, tmp_path):
        store, _home = opportunities_env
        from hermes_state import SessionDB

        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            db.create_session("s1", "cli")
            db.create_session("s2", "cli")
            db.append_message("s1", "user", "Draft a PRD for onboarding launch")
            db.append_message("s2", "user", "Turn this roadmap into backlog stories")
            db.append_message("s2", "user", "Review the product spec and launch plan")
        finally:
            db.close()

        result = store.scan_recent_usage(
            force=True,
            config={"proactive": {"min_repeats": 2, "scan_recent_messages": 20}},
            db_path=db_path,
        )

        assert result["scanned"] is True
        assert any(r["dedup_key"] == "usage:domain:product-manager" for r in store.list_pending())


def test_default_config_contains_opt_in_proactive_block():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["proactive"]["enabled"] is False
    assert DEFAULT_CONFIG["proactive"]["max_pending"] == 8
