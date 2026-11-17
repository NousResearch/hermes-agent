"""The strict boundary-config reader: a config.yaml that fails to parse must
never read as "not configured". Every test writes REAL YAML to the test
profile's config.yaml (HERMES_HOME is a per-test temp dir)."""
import json
import os
import time

import pytest

import agent.boundary_config as bc
from hermes_constants import get_hermes_home

GOOD = """
agent_write_guard: true
notion:
  write_guard:
    enabled: true
    release_kits: {data_sources: ["11111111111111111111111111111111"]}
    prds: {data_sources: ["22222222222222222222222222222222"]}
    ideas: {data_sources: ["33333333333333333333333333333333"]}
slack:
  post_allowed_channels: [CHOME001]
  post_gated_channels: [CREL0001]
autonomy:
  announce_channel: CHOME001
  actions: {post_release_update: auto}
"""
BROKEN = "notion:\n  write_guard: {enabled: true\nslack: [oops\n"


@pytest.fixture(autouse=True)
def _fresh_cache():
    bc.clear_cache()
    yield
    bc.clear_cache()


def _write(text):
    path = get_hermes_home() / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    # distinct mtime + size per version, like a real edit
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + int(time.time_ns() % 1000) + 1_000_000))
    return path


# ── the reader ───────────────────────────────────────────────────────────

def test_missing_never_seen_is_not_configured():
    assert bc.read_boundary_config() == {}
    assert not bc.is_unreadable(bc.read_boundary_config())


def test_broken_without_good_version_is_deny_sentinel(caplog):
    _write(BROKEN)
    cfg = bc.read_boundary_config()
    assert bc.is_unreadable(cfg)
    assert "could not be parsed" in caplog.text
    assert bc.boundary_config_errors(cfg)


@pytest.mark.parametrize("text", ["- just\n- a list\n", "", "42\n"])
def test_non_mapping_is_deny_sentinel(text):
    _write(text)
    assert bc.is_unreadable(bc.read_boundary_config())


def test_broken_after_good_keeps_last_good():
    _write(GOOD)
    assert bc.read_boundary_config()["slack"]["post_gated_channels"] == ["CREL0001"]
    _write(BROKEN)
    cfg = bc.read_boundary_config()
    assert not bc.is_unreadable(cfg)
    assert cfg["autonomy"]["actions"]["post_release_update"] == "auto"


def test_missing_after_good_keeps_last_good():
    path = _write(GOOD)
    bc.read_boundary_config()
    path.unlink()
    assert bc.read_boundary_config()["notion"]["write_guard"]["enabled"] is True


def test_fixed_file_is_read_again():
    _write(BROKEN)
    assert bc.is_unreadable(bc.read_boundary_config())
    _write("agent_write_guard: false\n")
    assert bc.read_boundary_config() == {"agent_write_guard": False}


# ── every consumer denies on a broken file with no good version ──────────

def test_notion_api_writes_denied(monkeypatch):
    import tools.notion_api_tool as nt
    monkeypatch.setattr(nt.shutil, "which", lambda cmd: "/usr/local/bin/ntn")
    monkeypatch.setattr(nt, "_ntn_env", lambda: {"NOTION_API_TOKEN": "tok"})
    ran = []

    class P:
        returncode, stdout, stderr = 0, '{"ok": true}', ""
    monkeypatch.setattr(nt.subprocess, "run", lambda argv, **kw: ran.append(argv) or P())
    _write(BROKEN)
    out = json.loads(nt._notion_api_handler({"path": "v1/pages/" + "a" * 32, "method": "DELETE"}))
    assert "could not be parsed" in out["error"] and ran == []
    out = json.loads(nt._notion_api_handler({"path": "v1/pages/" + "a" * 32, "method": "PATCH",
                                              "body": {"properties": {}}}))
    assert "error" in out and ran == []
    assert json.loads(nt._notion_api_handler({"path": "v1/pages/" + "a" * 32}))["ok"] is True
    import tools.notion_write_guard as g
    assert g.denial_for_tool_call("notion_api", {"path": "v1/pages/x", "method": "DELETE"})


def test_slack_post_thread_refused(monkeypatch):
    import tools.slack_post_thread_tool as spt
    from model_tools import _run_async
    sent = []

    async def post(token, channel, ts, text):
        sent.append(text)
        return {"ok": True, "ts": "1.1"}
    monkeypatch.setattr(spt, "_post_message", post)
    monkeypatch.setattr(spt, "_resolve_slack_token", lambda: "xoxb")
    _write(BROKEN)
    out = json.loads(_run_async(spt._slack_post_thread_handler({"chat_id": "CREL0001", "message": "hi"})))
    assert "could not be parsed" in out["error"] and sent == []


def test_cron_delivery_refuses_slack():
    import cron.scheduler as sch
    _write(BROKEN)
    kept, refused = sch._apply_post_policy([{"platform": "slack", "chat_id": "CHOME001"},
                                            {"platform": "telegram", "chat_id": "9"}])
    assert kept == [{"platform": "telegram", "chat_id": "9"}] and refused


def test_profile_write_guard_turns_on():
    from agent.profile_write_guard import write_guard_denial
    path = _write(BROKEN)
    assert write_guard_denial(str(path))
    assert write_guard_denial(str(get_hermes_home() / "SOUL.md"))


def test_profile_write_guard_off_only_when_parsed_off():
    from agent.profile_write_guard import write_guard_denial
    path = _write("agent_write_guard: false\n")
    assert write_guard_denial(str(path)) is None
    _write(BROKEN)  # a typo keeps the last good value (off) rather than flipping
    assert write_guard_denial(str(path)) is None


def test_profile_write_guard_stays_on_after_typo():
    from agent.profile_write_guard import write_guard_denial
    path = _write("agent_write_guard: true\n")
    assert write_guard_denial(str(path))
    _write("agent_write_guard: true\nslack: [oops\n")
    assert write_guard_denial(str(path))


def test_autonomy_off_and_preconditions_refuse():
    import agent.autonomy as au
    import tools.notion_write_guard as g
    import tools.zendesk_tool as zt
    _write(BROKEN)
    assert au.resolve_level("post_release_update") == "off"
    assert au.resolve_level("zendesk_create_draft") == "off"
    cfg = bc.read_boundary_config()
    assert "could not be parsed" in g.autonomy_precondition(cfg)
    assert zt._write_guard_enabled(cfg) is False
    assert g.is_enforcing(cfg) and "could not be parsed" in g.load_guard(cfg).error


def test_gated_ingress_ignores_everything():
    from types import SimpleNamespace
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False, platforms={})
    _write(BROKEN)
    assert runner._slack_source_gated(SimpleNamespace(chat_id="CANY0001"))


def test_guard_keeps_last_good_rules_through_a_typo():
    import tools.notion_write_guard as g
    _write(GOOD)
    assert g.load_guard(bc.read_boundary_config()).error == ""
    _write(GOOD + "\nautonomy: [broken\n")
    guard = g.load_guard(bc.read_boundary_config())
    assert guard.error == "" and guard.prds
