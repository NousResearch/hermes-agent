"""config.yaml agent.* bridges for the FTS v2 knobs (config-authoritative)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

import gateway.run as gateway_run


def _write_home(tmp_path: Path, agent_cfg: dict, env_text: str = "") -> Path:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"agent": agent_cfg}), encoding="utf-8"
    )
    (hermes_home / ".env").write_text(env_text, encoding="utf-8")
    return hermes_home


def test_fts_v2_read_bridged_from_config(tmp_path, monkeypatch):
    home = _write_home(tmp_path, {"fts_v2_read": False})
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setenv("HERMES_FTS_V2_READ", "1")
    gateway_run._reload_runtime_env_preserving_config_authority()
    assert os.environ["HERMES_FTS_V2_READ"] == "False"


def test_search_slow_ms_bridged_from_config(tmp_path, monkeypatch):
    home = _write_home(tmp_path, {"search_slow_ms": 250})
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.delenv("HERMES_SEARCH_SLOW_MS", raising=False)
    gateway_run._reload_runtime_env_preserving_config_authority()
    assert os.environ["HERMES_SEARCH_SLOW_MS"] == "250"


def test_env_survives_when_config_omits_fts_knobs(tmp_path, monkeypatch):
    home = _write_home(tmp_path, {"max_turns": 90})
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setenv("HERMES_FTS_V2_READ", "0")
    monkeypatch.setenv("HERMES_SEARCH_SLOW_MS", "700")
    gateway_run._reload_runtime_env_preserving_config_authority()
    assert os.environ["HERMES_FTS_V2_READ"] == "0"
    assert os.environ["HERMES_SEARCH_SLOW_MS"] == "700"


def test_fts_knobs_have_documented_defaults():
    """The advertised config surface must exist in DEFAULT_CONFIG (no
    user-facing env switch): v2 reads default ON, slow-search log at 1s."""
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"]["fts_v2_read"] is True
    assert DEFAULT_CONFIG["agent"]["search_slow_ms"] == 1000


def test_config_false_disables_v2_read_semantics(tmp_path, monkeypatch):
    """The bridged 'False' string must parse as OFF in hermes_state."""
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "missing.so"))
    d = SessionDB(db_path=tmp_path / "state.db")
    try:
        d._fts_v2_ready = True  # pretend the index is ready
        d._fts_v1_present = True
        monkeypatch.setenv("HERMES_FTS_V2_READ", "False")
        assert not d._fts_v2_query_allowed("graphiti")
        monkeypatch.setenv("HERMES_FTS_V2_READ", "True")
        assert d._fts_v2_query_allowed("graphiti")
    finally:
        d.close()
