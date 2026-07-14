"""Regression tests for the cron scheduler's multiplex profile handling.

The cron scheduler runs inside the multiplexed gateway process. When
``gateway.multiplex_profiles`` is on, every ``get_secret`` call
(provider keys, platform tokens) requires an active
``profile_runtime_scope`` — otherwise the agent/secret_scope module
fails closed with ``UnscopedSecretError``.

These tests cover the cron-side seam:

* ``_resolve_target_profile_home`` correctly maps a job's ``deliver``
  target to the right profile's HERMES_HOME (multiplex root vs.
  per-profile vs. stub).
* ``_process_job`` enters that scope before invoking ``run_one_job``,
  so both the LLM path (``resolve_runtime_provider``) and the
  delivery path (``_deliver_result`` → gateway-config load) read
  credentials from the right profile.
* Single-profile deployments are unaffected — the scope is only
  entered when ``is_multiplex_active()`` is True.

Characterization, not full E2E: the agent / provider / gateway code
paths are heavy and exercised elsewhere. We assert the cron-side
plumbing alone, against a temp HERMES_HOME tree that mirrors the
real shape (default profile at root, named profile under profiles/,
pairing files in both).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

import cron.scheduler as cs
import agent.secret_scope as ss


# ─── fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def multiplex_hermes_home(tmp_path: Path, monkeypatch) -> dict:
    """Build a temp HERMES_HOME shaped like the production multiplex setup.

    Layout:
      <tmp>/                                     (multiplex root, default home)
        .env                                     (default creds)
        pairing/weixin-approved.json             (legacy default whitelist)
        profiles/default/pairing/weixin-approved.json   (default profile stub)
        profiles/yangyang/.env                   (yangyang's full creds)
        profiles/yangyang/pairing/weixin-approved.json
    """
    hermes_home = tmp_path
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", False)

    (hermes_home / "pairing").mkdir()
    (hermes_home / "profiles" / "default" / "pairing").mkdir(parents=True)
    (hermes_home / "profiles" / "yangyang").mkdir()
    (hermes_home / "profiles" / "yangyang" / "pairing").mkdir()

    (hermes_home / ".env").write_text(
        "OPENROUTER_API_KEY=default-or-key\n"
        "WEIXIN_TOKEN=default-weixin-token\n",
        encoding="utf-8",
    )
    (hermes_home / "profiles" / "yangyang" / ".env").write_text(
        "OPENROUTER_API_KEY=yangyang-or-key\n"
        "WEIXIN_TOKEN=yangyang-weixin-token\n",
        encoding="utf-8",
    )

    (hermes_home / "pairing" / "weixin-approved.json").write_text(json.dumps({
        "ceo-chat": {"user_name": "CEO"},
        "yangyang-chat": {"user_name": "杨洋"},
    }), encoding="utf-8")
    (hermes_home / "profiles" / "default" / "pairing" / "weixin-approved.json").write_text(json.dumps({
        "ceo-chat": {"user_name": "CEO"},
    }), encoding="utf-8")
    (hermes_home / "profiles" / "yangyang" / "pairing" / "weixin-approved.json").write_text(json.dumps({
        "yangyang-chat": {"user_name": "杨洋"},
    }), encoding="utf-8")

    return {
        "home": hermes_home,
        "default_stub": hermes_home / "profiles" / "default",
        "yangyang": hermes_home / "profiles" / "yangyang",
    }


# ─── resolver tests ──────────────────────────────────────────────────

class TestParseDeliveryChatId:
    def test_weixin_with_chat_id(self):
        assert cs._parse_delivery_chat_id(
            "weixin:o9cq808JHZwoipViF6apGkdaygXg@im.wechat"
        ) == ("weixin", "o9cq808JHZwoipViF6apGkdaygXg@im.wechat")

    def test_weixin_with_thread_id_strips_suffix(self):
        assert cs._parse_delivery_chat_id(
            "weixin:o9cq808JHZwoipViF6apGkdaygXg@im.wechat:17585"
        ) == ("weixin", "o9cq808JHZwoipViF6apGkdaygXg@im.wechat")

    def test_local_deliver_returns_none(self):
        assert cs._parse_delivery_chat_id("local") == (None, None)
        assert cs._parse_delivery_chat_id("") == (None, None)

    def test_platform_only_returns_none(self):
        assert cs._parse_delivery_chat_id("feishu") == (None, None)
        assert cs._parse_delivery_chat_id("weixin:") == (None, None)


class TestResolveTargetProfileHome:

    def test_multiplex_off_returns_root_for_all(self, multiplex_hermes_home):
        h = cs._resolve_target_profile_home({
            "id": "x", "deliver": "weixin:yangyang-chat"
        })
        assert h == multiplex_hermes_home["home"]

    def test_multiplex_on_ceo_chat_id_routes_to_root_via_stub(
        self, multiplex_hermes_home
    ):
        ss.set_multiplex_active(True)
        try:
            h = cs._resolve_target_profile_home({
                "id": "ceo_news", "deliver": "weixin:ceo-chat"
            })
            assert h == multiplex_hermes_home["home"]
        finally:
            ss.set_multiplex_active(False)

    def test_multiplex_on_yangyang_chat_id_routes_to_yangyang(
        self, multiplex_hermes_home
    ):
        ss.set_multiplex_active(True)
        try:
            h = cs._resolve_target_profile_home({
                "id": "yang_brief", "deliver": "weixin:yangyang-chat"
            })
            assert h == multiplex_hermes_home["yangyang"]
        finally:
            ss.set_multiplex_active(False)

    def test_multiplex_on_local_deliver_returns_root(self, multiplex_hermes_home):
        ss.set_multiplex_active(True)
        try:
            h = cs._resolve_target_profile_home({"id": "x", "deliver": "local"})
            assert h == multiplex_hermes_home["home"]
        finally:
            ss.set_multiplex_active(False)

    def test_explicit_profile_field_overrides_auto(self, multiplex_hermes_home):
        ss.set_multiplex_active(True)
        try:
            h = cs._resolve_target_profile_home({
                "id": "x",
                "deliver": "weixin:ceo-chat",
                "profile": "yangyang",
            })
            assert h == multiplex_hermes_home["yangyang"]
        finally:
            ss.set_multiplex_active(False)

    def test_explicit_default_returns_root_not_stub(self, multiplex_hermes_home):
        ss.set_multiplex_active(True)
        try:
            h = cs._resolve_target_profile_home({
                "id": "x", "deliver": "weixin:yangyang-chat", "profile": "default"
            })
            assert h == multiplex_hermes_home["home"]
        finally:
            ss.set_multiplex_active(False)

    def test_explicit_profile_home_path(self, multiplex_hermes_home):
        custom = multiplex_hermes_home["home"] / "profiles" / "yangyang"
        h = cs._resolve_target_profile_home({
            "id": "x", "deliver": "local", "profile_home": str(custom)
        })
        assert h == custom

    def test_missing_profile_name_warns_and_falls_back(
        self, multiplex_hermes_home, caplog
    ):
        h = cs._resolve_target_profile_home({
            "id": "x", "deliver": "local", "profile": "nope"
        })
        assert h == multiplex_hermes_home["home"]


# ─── e2e: scope sets the right home and get_secret reads from it ────

class TestProfileRuntimeScope:

    def test_outside_scope_raises_in_multiplex(self, multiplex_hermes_home):
        ss.set_multiplex_active(True)
        try:
            with pytest.raises(RuntimeError, match="no profile secret scope"):
                ss.get_secret("WEIXIN_TOKEN")
        finally:
            ss.set_multiplex_active(False)

    def test_scope_yields_correct_home(self, multiplex_hermes_home):
        from agent.profile_scope import (
            active_profile_home,
            profile_runtime_scope,
        )
        with mock.patch.object(cs, "_is_multiplex_active", return_value=False):
            with profile_runtime_scope(multiplex_hermes_home["yangyang"]):
                assert active_profile_home() == multiplex_hermes_home["yangyang"]
            assert active_profile_home() is None

    def test_scope_yields_correct_secrets(self, multiplex_hermes_home):
        ss.set_multiplex_active(True)
        try:
            from agent.profile_scope import profile_runtime_scope
            with profile_runtime_scope(multiplex_hermes_home["yangyang"]):
                assert ss.get_secret("WEIXIN_TOKEN") == "yangyang-weixin-token"
                assert ss.get_secret("OPENROUTER_API_KEY") == "yangyang-or-key"
            with pytest.raises(RuntimeError):
                ss.get_secret("WEIXIN_TOKEN")
        finally:
            ss.set_multiplex_active(False)

    def test_scope_isolated_from_default(self, multiplex_hermes_home):
        ss.set_multiplex_active(True)
        try:
            from agent.profile_scope import profile_runtime_scope
            with profile_runtime_scope(multiplex_hermes_home["home"]):
                assert ss.get_secret("WEIXIN_TOKEN") == "default-weixin-token"
            with profile_runtime_scope(multiplex_hermes_home["yangyang"]):
                assert ss.get_secret("WEIXIN_TOKEN") == "yangyang-weixin-token"
        finally:
            ss.set_multiplex_active(False)


# ─── integration: _process_job wraps the body in the right scope ────

class TestProcessJobEntersScope:

    def test_multiplex_off_does_not_enter_scope(self, multiplex_hermes_home):
        captured = {"scopes": []}

        real_scope = __import__("agent.profile_scope", fromlist=["profile_runtime_scope"]).profile_runtime_scope

        def spy_scope(home):
            captured["scopes"].append(home)
            return real_scope(home)

        with mock.patch.object(cs, "_is_multiplex_active", return_value=False), \
             mock.patch.object(cs, "run_one_job", return_value=True) as run_one_job, \
             mock.patch("agent.profile_scope.profile_runtime_scope", side_effect=spy_scope):
            with mock.patch.object(cs, "get_due_jobs", return_value=[{
                "id": "j", "name": "t", "deliver": "weixin:yangyang-chat",
            }]), mock.patch.object(cs, "advance_next_run", return_value=True):
                cs.tick(verbose=False, sync=True)
            assert captured["scopes"] == [], "scope entered when multiplex off"
            assert run_one_job.called

    def test_multiplex_on_enters_yangyang_scope(self, multiplex_hermes_home):
        ss.set_multiplex_active(True)
        try:
            from agent.profile_scope import profile_runtime_scope, active_profile_home
            seen = {}
            captured_homes = []

            def spy_scope(home):
                captured_homes.append(home)
                return profile_runtime_scope(home)

            def spy_run_job(j):
                seen["weixin"] = ss.get_secret("WEIXIN_TOKEN")
                seen["home"] = active_profile_home()
                return (True, "stub output", "stub response", None)

            with mock.patch("agent.profile_scope.profile_runtime_scope", side_effect=spy_scope), \
                 mock.patch.object(cs, "run_job", side_effect=spy_run_job), \
                 mock.patch.object(cs, "save_job_output", return_value="/tmp/j"), \
                 mock.patch.object(cs, "_deliver_result", return_value=None), \
                 mock.patch.object(cs, "mark_job_run", return_value=None), \
                 mock.patch.object(cs, "get_due_jobs", return_value=[{
                     "id": "j", "name": "t", "deliver": "weixin:yangyang-chat",
                 }]), \
                 mock.patch.object(cs, "advance_next_run", return_value=True):
                cs.tick(verbose=False, sync=True)
            assert captured_homes == [multiplex_hermes_home["yangyang"]], captured_homes
            assert seen["weixin"] == "yangyang-weixin-token", seen
            assert seen["home"] == multiplex_hermes_home["yangyang"], seen
        finally:
            ss.set_multiplex_active(False)

    def test_multiplex_on_ceo_chat_uses_root_scope(self, multiplex_hermes_home):
        ss.set_multiplex_active(True)
        try:
            from agent.profile_scope import profile_runtime_scope, active_profile_home
            captured_homes = []
            seen = {}

            def spy_scope(home):
                captured_homes.append(home)
                return profile_runtime_scope(home)

            def spy_run_job(j):
                seen["weixin"] = ss.get_secret("WEIXIN_TOKEN")
                seen["home"] = active_profile_home()
                return (True, "stub output", "stub response", None)

            with mock.patch("agent.profile_scope.profile_runtime_scope", side_effect=spy_scope), \
                 mock.patch.object(cs, "run_job", side_effect=spy_run_job), \
                 mock.patch.object(cs, "save_job_output", return_value="/tmp/j"), \
                 mock.patch.object(cs, "_deliver_result", return_value=None), \
                 mock.patch.object(cs, "mark_job_run", return_value=None), \
                 mock.patch.object(cs, "get_due_jobs", return_value=[{
                     "id": "j", "name": "t", "deliver": "weixin:ceo-chat",
                 }]), \
                 mock.patch.object(cs, "advance_next_run", return_value=True):
                cs.tick(verbose=False, sync=True)
            assert captured_homes == [multiplex_hermes_home["home"]], captured_homes
            assert seen["weixin"] == "default-weixin-token", seen
            assert seen["home"] == multiplex_hermes_home["home"], seen
        finally:
            ss.set_multiplex_active(False)
