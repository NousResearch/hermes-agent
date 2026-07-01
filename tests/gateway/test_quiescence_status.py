"""Tests for the busy-gateway-quiescence feature — Phase 1 (status persistence).

Spec: ~/.hermes/plans/2026-06-30_safe-restart-busy-gateway-quiescence-SPEC.md (v0.4)
Pinned fork HEAD: 0e21a7b93479592a3b666c4076662f732070c0ea

Phase 1 = the additive, zero-risk foundation: persist `active_agent_keys` (the live
running-session keys, not just the count) + a `schema_version`, with the secret-bearing
file written `0600` (D-1/RC-5). Everything else (reaper, per-session quiescence) builds
on the watcher being able to read these keys.
"""
import json
import os
import stat

import gateway.status as status


def test_write_runtime_status_persists_active_agent_keys(tmp_path, monkeypatch):
    """D-1: active_agent_keys (the live session-key list) is persisted additively."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_runtime_status(
        gateway_state="running",
        active_agents=2,
        active_agent_keys=["agent:main:discord:group:111:222", "agent:main:discord:group:333:444"],
    )
    rec = status.read_runtime_status()
    assert rec["active_agents"] == 2
    assert rec["active_agent_keys"] == [
        "agent:main:discord:group:111:222",
        "agent:main:discord:group:333:444",
    ]


def test_write_runtime_status_sets_schema_version(tmp_path, monkeypatch):
    """D-7: schema_version stamped so future readers can branch."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_runtime_status(gateway_state="running", active_agents=0)
    rec = status.read_runtime_status()
    assert rec.get("schema_version") == 2


def test_active_agent_keys_omitted_when_not_passed_is_back_compat(tmp_path, monkeypatch):
    """INV-6: an absent active_agent_keys arg does NOT inject the field (old readers
    unaffected); a count-only write still works exactly as before."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_runtime_status(gateway_state="running", active_agents=1)
    rec = status.read_runtime_status()
    assert rec["active_agents"] == 1
    assert "active_agent_keys" not in rec  # not injected when caller didn't pass it


def test_active_agent_keys_empty_list_persists_as_empty(tmp_path, monkeypatch):
    """An explicit empty list (idle gateway) persists as [] — distinct from 'omitted'."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_runtime_status(gateway_state="running", active_agents=0, active_agent_keys=[])
    rec = status.read_runtime_status()
    assert rec["active_agent_keys"] == []


def test_gateway_state_json_is_mode_0600(tmp_path, monkeypatch):
    """D-1/RC-5: active_agent_keys carry chat-id + user-id (multi-user metadata), so the
    file MUST be 0600 (owner-only) — the host-local threat-model mitigation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_runtime_status(
        gateway_state="running", active_agents=1,
        active_agent_keys=["agent:main:discord:group:111:222"],
    )
    path = status._get_runtime_status_path()
    file_mode = stat.S_IMODE(os.stat(path).st_mode)
    assert file_mode == 0o600, f"gateway_state.json mode {oct(file_mode)} != 0600"


def test_active_agents_count_unchanged_alongside_keys(tmp_path, monkeypatch):
    """INV-6: adding keys does not change the count semantics or its coercion."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status.write_runtime_status(active_agents=3, active_agent_keys=["a", "b", "c"])
    rec = status.read_runtime_status()
    assert rec["active_agents"] == 3
    # negative coercion still clamps (existing parse_active_agents contract)
    status.write_runtime_status(active_agents=-5)
    assert status.read_runtime_status()["active_agents"] == 0
