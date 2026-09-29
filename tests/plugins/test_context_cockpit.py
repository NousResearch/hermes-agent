"""Unit tests for Context Cockpit status, SQLite helper, and launcher."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "plugins" / "hermes-context-visor"
sys.path.insert(0, str(SCRIPTS))

from context_cockpit.launcher import (  # noqa: E402
    build_visor_argv,
    build_visor_url,
    launch_context_visor,
    platform_fallback_instructions,
)
from context_cockpit.controls import build_action_controls  # noqa: E402
from context_cockpit.sqlite_ro import busy_timeout_of, open_readonly  # noqa: E402
from context_cockpit.status import classify_lcm_state, classify_status, build_status_payload  # noqa: E402
from context_cockpit.web import render_cockpit_html, render_operator_guide_html  # noqa: E402


def _base_metrics(**overrides):
    m = {
        "freshness": "fresh",
        "prompt_pct": 10.0,
        "prompt_tokens": 40_000,
        "message_count": 20,
        "model": "deepseek/deepseek-v4-flash",
        "model_alert": None,
        "window": 1_000_000,
        "lcm": {
            "loaded": True,
            "threshold_ratio": 0.25,
            "threshold_tokens": 250_000,
            "compressions": 0,
            "fill_of_lcm": 0.16,
            "cache_state": "hot",
            "turns_since_leaf": 2,
            "last_leaf_compaction_at": None,
            "last_compaction_duration_ms": None,
            "last_api_call_at": None,
        },
        "cost": {
            "estimated_usd": 0.5,
            "actual_usd": None,
            "billing_mode": "payg",
            "cost_status": "estimated",
            "burn": {},
        },
        "liveness": {"running": True, "heartbeat_age_sec": 5.0},
    }
    m.update(overrides)
    if "lcm" in overrides:
        base_lcm = {
            "loaded": True,
            "threshold_ratio": 0.25,
            "threshold_tokens": 250_000,
            "compressions": 0,
            "fill_of_lcm": 0.16,
            "cache_state": "hot",
            "turns_since_leaf": 2,
            "last_leaf_compaction_at": None,
            "last_compaction_duration_ms": None,
            "last_api_call_at": None,
        }
        base_lcm.update(overrides["lcm"])
        m["lcm"] = base_lcm
    return m


def test_classify_healthy():
    s = classify_status(_base_metrics())
    assert s["ribbon"] == "ALL GOOD"
    assert "Do nothing" in s["next_action"]


def test_classify_compress_soon_near_lcm():
    s = classify_status(
        _base_metrics(
            prompt_tokens=230_000,
            prompt_pct=23.0,
            lcm={"fill_of_lcm": 0.92, "threshold_tokens": 250_000},
        )
    )
    assert s["ribbon"] == "GETTING FULL"


def test_classify_lcm_threshold_reached_without_active_proof():
    s = classify_status(
        _base_metrics(
            prompt_tokens=260_000,
            prompt_pct=26.0,
            lcm={"fill_of_lcm": 1.04, "threshold_tokens": 250_000},
        )
    )
    assert s["ribbon"] == "MEMORY LINE HIT"
    payload = build_status_payload(
        _base_metrics(
            prompt_tokens=260_000,
            prompt_pct=26.0,
            lcm={"fill_of_lcm": 1.04, "threshold_tokens": 250_000},
        )
    )
    assert "does not prove Hermes is shrinking" in payload["lcm_state"]["detail"]


def test_classify_lcm_waiting_when_pending_status_is_present():
    s = classify_status(
        _base_metrics(
            prompt_tokens=260_000,
            prompt_pct=26.0,
            lcm={
                "fill_of_lcm": 1.04,
                "threshold_tokens": 250_000,
                "last_compression_status": "pending",
            },
        )
    )
    assert s["ribbon"] == "SHRINK QUEUED"


def test_classify_lcm_blocked_when_noop_reason_is_present():
    s = classify_status(
        _base_metrics(
            prompt_tokens=260_000,
            prompt_pct=26.0,
            lcm={
                "fill_of_lcm": 1.04,
                "threshold_tokens": 250_000,
                "last_compression_status": "noop",
                "last_compression_noop_reason": "no eligible raw backlog outside fresh tail",
            },
        )
    )
    assert s["ribbon"] == "CAN'T SHRINK YET"
    assert "older chat" in s["summary"].lower()


def test_classify_lcm_noop_surfaces_live_counts():
    state = classify_lcm_state(
        _base_metrics(
            prompt_tokens=260_000,
            prompt_pct=26.0,
            lcm={
                "fill_of_lcm": 1.04,
                "threshold_tokens": 250_000,
                "last_compression_status": "noop",
                "last_compression_noop_reason": "no eligible raw backlog outside fresh tail",
                "fresh_tail_count": 32,
                "pre_tail_message_count": 0,
                "total_message_count": 32,
            },
        )
    )
    assert state["ribbon"] == "CAN'T SHRINK YET"
    assert "32 messages in view" in state["detail"]
    assert "0 older" in state["detail"]


def test_classify_lcm_recent_when_compaction_just_happened():
    lcm_state = classify_lcm_state(
        _base_metrics(
            prompt_tokens=180_000,
            prompt_pct=18.0,
            lcm={
                "fill_of_lcm": 0.72,
                "threshold_tokens": 250_000,
                "compressions": 1,
                "turns_since_leaf": 1,
                "last_leaf_compaction_at": 1000.0,
                "last_api_call_at": 1060.0,
            },
        )
    )
    assert lcm_state["ribbon"] == "JUST SHRANK"
    assert lcm_state["active_proven"] is False


def test_classify_lcm_unknown_when_not_loaded():
    lcm_state = classify_lcm_state(_base_metrics(lcm={"loaded": False}))
    assert lcm_state["ribbon"] == "MEMORY UNKNOWN"


def test_classify_model_warning():
    s = classify_status(_base_metrics(model_alert="Model changed: gpt-5.4 → flash"))
    assert s["ribbon"] == "MODEL CHANGED"


def test_classify_stale_overrides_healthy_numbers():
    s = classify_status(
        _base_metrics(freshness="stale", prompt_pct=5.0, lcm={"fill_of_lcm": 0.1})
    )
    assert s["ribbon"] == "OLD NUMBERS"
    assert s["dim_gauges"] is True


def test_classify_offline():
    s = classify_status(_base_metrics(freshness="offline"))
    assert s["ribbon"] == "HERMES OFFLINE"
    assert s["dim_gauges"] is True
    assert "wait a few seconds" in s["next_action"]


def test_classify_watch_band():
    s = classify_status(
        _base_metrics(
            prompt_tokens=180_000,
            prompt_pct=18.0,
            lcm={"fill_of_lcm": 0.72, "threshold_tokens": 250_000},
        )
    )
    assert s["ribbon"] == "QUIET"


def test_classify_idle_copy_does_not_claim_chat_is_dead():
    s = classify_status(_base_metrics(freshness="idle"))
    assert s["ribbon"] == "QUIET"
    assert "no new chat activity" in s["summary"].lower()
    assert "turn is still running" in s["next_action"].lower()


def test_freshness_prefers_live_lcm_over_stale_state_db(tmp_path: Path):
    """Long Desktop turns can leave state.db mtime old while LCM snapshot is live."""
    from context_cockpit.metrics import collect_metrics

    profile_dir = tmp_path / "personal-ops"
    profile_dir.mkdir()
    (profile_dir / "state.db").write_bytes(b"")
    old = time.time() - 300
    os.utime(profile_dir / "state.db", (old, old))

    with patch("context_cockpit.metrics.hermes_liveness") as live_fn, patch(
        "context_cockpit.metrics.current_session",
        return_value={
            "id": "sess-1",
            "model": "deepseek/deepseek-v4-flash",
            "system_prompt": "",
        },
    ), patch(
        "context_cockpit.metrics.conversation_mass",
        return_value={"tokens": 1000, "messages": 4, "chars": 100},
    ), patch(
        "context_cockpit.metrics.resolve_window",
        return_value=1_000_000,
    ), patch(
        "context_cockpit.metrics.lcm_telemetry",
        return_value={
            "loaded": True,
            "live_snapshot_loaded": True,
            "live_snapshot_age_sec": 1.5,
            "last_api_call_at": time.time() - 1.0,
            "last_observed_prompt_tokens": 12000,
            "threshold_tokens": 250000,
            "total_compactions": 0,
        },
    ):
        live_fn.return_value = {
            "running": True,
            "pid": 1,
            "source": "process_scan",
            "command": "Hermes",
            "gateway_state": None,
            "gateway_age_sec": 999,
            "processes_age_sec": None,
            "state_db_age_sec": 300,
            "heartbeat_age_sec": 300,
            "heartbeat_source": "process_scan+state.db",
        }
        metrics = collect_metrics("personal-ops", profile_dir, {})
    assert metrics["freshness"] == "fresh"
    assert metrics["liveness"]["activity_source"] in {
        "live-lcm-snapshot",
        "last-api-call",
    }
    assert metrics["liveness"]["heartbeat_age_sec"] < 30


def test_json_payload_shape():
    payload = build_status_payload(_base_metrics())
    assert payload["ok"] is True
    assert payload["read_only"] is True
    assert payload["ribbon"] == "ALL GOOD"
    assert "lcm_state" in payload
    assert "controls" in payload
    assert "metrics" in payload
    assert "status" in payload


def test_controls_allowlist_and_modes():
    controls = build_action_controls(classify_status(_base_metrics()))
    copy_commands = {c["command"] for c in controls if c["action_type"] == "copy_command" and c["allowed"]}
    assert copy_commands == {"/lcm status", "/compress --preview", "/usage", "/model", "/status"}
    assert any(c["action_type"] == "refresh_status" and c["allowed"] for c in controls)
    assert any(c["action_type"] == "open_readonly_url" and c["url"] == "/api/status" for c in controls)
    assert any(c["action_type"] == "open_readonly_url" and c["url"] == "/operator-guide" for c in controls)


def test_controls_block_mutating_recommendation_in_browser():
    controls = build_action_controls(
        {
            "command": "/compress here 4",
            "next_action": "Run /compress after this answer.",
        }
    )
    blocked = next(c for c in controls if c["id"] == "blocked-recommended")
    assert blocked["allowed"] is False
    assert blocked["mode"] == "blocked"
    assert blocked["command"] == "/compress here 4"


def test_controls_reject_injection_like_recommendation_from_copy_path():
    controls = build_action_controls({"command": "/compress; rm -rf /"})
    blocked = next(c for c in controls if c["id"] == "blocked-recommended")
    assert blocked["allowed"] is False
    assert blocked["command"] == "/compress; rm -rf /"
    assert all(c.get("command") != "/compress; rm -rf /" or not c["allowed"] for c in controls)


def test_controls_do_not_reference_broker_hindsight_or_lcm_writes():
    blob = str(build_action_controls(classify_status(_base_metrics()))).lower()
    assert "broker" not in blob
    assert "hindsight" not in blob
    assert "rotate apply" not in blob
    assert "real_execution" not in blob


def test_sqlite_ro_busy_timeout(tmp_path: Path):
    db = tmp_path / "t.db"
    conn_w = sqlite3.connect(db)
    conn_w.execute("CREATE TABLE t (id INTEGER)")
    conn_w.commit()
    conn_w.close()

    conn = open_readonly(db, busy_timeout_ms=5000)
    assert busy_timeout_of(conn) == 5000
    # Prove read works
    assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 0
    conn.close()


def test_build_visor_argv_fixed_no_shell_chars(tmp_path: Path, monkeypatch):
    script = tmp_path / "context_visor.py"
    script.write_text("# stub\n")
    py = tmp_path / "python3"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    monkeypatch.setenv("HERMES_VISOR_SCRIPT", str(script))
    monkeypatch.setenv("HERMES_PYTHON", str(py))
    monkeypatch.delenv("HERMES_CONTEXT_VISOR_BIN", raising=False)
    # Force missing bin so we use python+script path
    with patch("context_cockpit.launcher.resolve_visor_bin", return_value=None):
        argv = build_visor_argv("personal-ops")
    assert argv[0] == str(py)
    assert argv[1] == str(script)
    assert argv[2:4] == ["--profile", "personal-ops"]
    assert all(";" not in a and "|" not in a for a in argv)


def test_build_visor_url_localhost():
    url = build_visor_url("personal-ops", port=8421)
    assert url == "http://127.0.0.1:8421/"


def test_build_visor_argv_rejects_bad_profile():
    with pytest.raises(ValueError):
        build_visor_argv("personal-ops; rm -rf /")


def test_build_visor_argv_rejects_unsafe_extra():
    with pytest.raises(ValueError):
        build_visor_argv("personal-ops", extra=["--once;evil"])


def test_launch_reuses_existing_server():
    with patch("context_cockpit.launcher._server_ready", return_value=True):
        with patch("context_cockpit.launcher.open_browser", return_value=True):
            result = launch_context_visor()
    assert result.ok is True
    assert result.already_running is True
    assert result.method == "browser_existing"
    assert "browser" in result.message.lower()


def test_launch_starts_server_and_opens_browser(tmp_path: Path):
    fake_bin = tmp_path / "hermes-context-visor"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    argv = [str(fake_bin), "--serve", "--no-browser", "--port", "8421"]
    with patch("context_cockpit.launcher.build_visor_argv", return_value=argv):
        with patch("context_cockpit.launcher._server_ready", side_effect=[False, True]):
            with patch("context_cockpit.launcher.open_browser", return_value=True):
                with patch("context_cockpit.launcher.subprocess.Popen") as popen:
                    result = launch_context_visor()
    assert result.ok is True
    assert result.method == "browser_started"
    assert popen.called
    assert "browser" in result.message.lower()


def test_launch_waits_for_runtime_warmup_message_when_status_stays_offline(tmp_path: Path):
    fake_bin = tmp_path / "hermes-context-visor"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    argv = [str(fake_bin), "--serve", "--no-browser", "--port", "8421"]
    offline_payload = {
        "metrics": {
            "profile": "personal-ops",
            "freshness": "offline",
            "liveness": {"running": False},
        }
    }
    with patch("context_cockpit.launcher.build_visor_argv", return_value=argv):
        with patch("context_cockpit.launcher._server_ready", side_effect=[False, True]):
            with patch("context_cockpit.launcher._warm_runtime_status", return_value=offline_payload):
                with patch("context_cockpit.launcher.open_browser", return_value=True):
                    with patch("context_cockpit.launcher.subprocess.Popen"):
                        result = launch_context_visor()
    assert result.ok is True
    assert "warming up" in result.message.lower()


def test_launch_spawns_fixed_argv_linux(tmp_path: Path):
    fake_bin = tmp_path / "hermes-context-visor"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    argv = [str(fake_bin), "--serve", "--no-browser", "--port", "8421"]
    with patch("context_cockpit.launcher.build_visor_argv", return_value=argv):
        with patch("context_cockpit.launcher._server_ready", side_effect=[False, True]):
            with patch("context_cockpit.launcher.open_browser", return_value=False):
                with patch("context_cockpit.launcher.subprocess.Popen") as popen:
                    result = launch_context_visor()
    assert result.ok is True
    assert result.method == "browser_started"
    called = popen.call_args[0][0]
    assert called == argv
    assert popen.call_args.kwargs.get("shell") in (None, False)


def test_platform_fallback_mentions_windows_and_linux():
    text = platform_fallback_instructions("personal-ops")
    assert "hermes-context-visor" in text
    assert "personal-ops" in text
    assert "http://127.0.0.1:" in text


def test_render_cockpit_html_contains_required_cards():
    html = render_cockpit_html(profile="personal-ops")
    assert "Context Cockpit" in html
    assert "Flight Deck" in html
    assert "instrument-deck" in html
    assert "context-arc" in html
    assert "context-needle" in html
    assert "context-ticks" in html
    assert "lcm-arc" in html
    assert "lcm-needle" in html
    assert "LCM / Auto-shrink" in html
    assert "Distance to auto-shrink" in html
    assert "% left" in html
    assert "live-sweep" in html
    assert "heart-spin" in html
    assert "details-summary-hint" in html
    assert "cost-burn" in html
    assert "model-orb" in html
    assert "heart-ring" in html
    assert "next-strip" in html
    assert "live-pill" in html
    assert "/api/stream" in html
    assert "/api/demo" in html
    assert "EventSource" in html
    assert "Details" in html
    assert "one-row · expand for numbers" in html
    assert "action-bar" in html
    assert "Read-only. Buttons only copy, refresh, or open local pages" in html
    assert "gauge-readout" in html
    assert "paintMeter" in html
    assert "gauge-total" in html
    assert "context-max" in html
    assert "Orange = shrink zone" in html
    assert "Auto cleanup" not in html


def test_stream_interval_adapts_to_ribbon():
    from context_cockpit.web import stream_interval_ms

    assert stream_interval_ms("ALL GOOD") == 8000
    assert stream_interval_ms("CAN'T SHRINK YET") == 2500
    assert stream_interval_ms("HERMES OFFLINE") == 1000
    assert stream_interval_ms("OLD NUMBERS") == 1000
    assert stream_interval_ms("COST WARNING") == 1000


def test_demo_scenarios_cover_alert_states():
    from context_cockpit.web import DEMO_SCENARIOS, build_demo_payload

    expected = {
        "healthy": "ALL GOOD",
        "near_threshold": "GETTING FULL",
        "shrink_queued": "SHRINK QUEUED",
        "shrinking": "SHRINKING NOW",
        "just_shrank": "JUST SHRANK",
        "stale": "OLD NUMBERS",
        "offline": "HERMES OFFLINE",
        "model_warning": "MODEL CHANGED",
        "cost_warning": "COST WARNING",
    }
    assert set(DEMO_SCENARIOS) == set(expected)
    for scenario, ribbon in expected.items():
        payload = build_demo_payload(scenario)
        assert payload["demo"] is True
        assert payload["ribbon"] == ribbon, scenario
        assert payload["read_only"] is True


def test_render_operator_guide_html_escapes_markup():
    html = render_operator_guide_html("# Hello\n<script>alert(1)</script>")
    assert "Context Cockpit Operator Guide" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_unbound_snapshot_does_not_clobber_bound_snapshot(tmp_path: Path):
    from context_cockpit.live_lcm import (
        read_live_lcm_snapshot,
        write_live_lcm_snapshot_for_engine,
    )

    class _Engine:
        def __init__(self, session_id="", conversation_id="", status=None, preview=None):
            self.current_session_id = session_id
            self.current_conversation_id = conversation_id
            self._status = status or {"last_compression_status": "idle"}
            self._preview = preview or {"ok": False, "reason": "no_active_session"}
            self._config = type("C", (), {"fresh_tail_count": 32, "leaf_chunk_tokens": 20000})()

        def get_status(self):
            return self._status

        def rotate_active_session(self, apply=False):
            return self._preview

    bound = _Engine(
        session_id="20260708_193111_e438e8",
        conversation_id="20260708_193111_e438e8",
        status={
            "last_compression_status": "noop",
            "last_compression_noop_reason": "fresh_tail_only",
            "compression_count": 1,
            "threshold_tokens": 250000,
            "last_prompt_tokens": 250541,
            "context_length": 1000000,
        },
        preview={
            "ok": True,
            "noop": True,
            "reason": "fresh_tail_only",
            "total_message_count": 40,
            "fresh_tail_count": 32,
            "pre_tail_message_count": 8,
            "current_frontier_store_id": 1,
            "new_frontier_store_id": 1,
        },
    )
    first = write_live_lcm_snapshot_for_engine(bound, tmp_path)
    assert first["conversation_id"] == "20260708_193111_e438e8"
    assert first["last_compression_status"] == "noop"

    unbound = _Engine()
    second = write_live_lcm_snapshot_for_engine(unbound, tmp_path)
    assert second["conversation_id"] == "20260708_193111_e438e8"
    assert second["last_compression_status"] == "noop"
    assert read_live_lcm_snapshot(tmp_path)["last_compression_status"] == "noop"


def test_lcm_telemetry_merges_bound_snapshot_by_session_id(tmp_path: Path):
    from context_cockpit.live_lcm import write_live_lcm_snapshot_for_engine
    from context_cockpit.metrics import lcm_telemetry

    class _Engine:
        current_session_id = "sess-1"
        current_conversation_id = "sess-1"
        _config = type("C", (), {"fresh_tail_count": 12, "leaf_chunk_tokens": 20000})()

        def get_status(self):
            return {
                "last_compression_status": "pending",
                "last_compression_noop_reason": "",
                "compression_count": 0,
                "threshold_tokens": 250000,
                "last_prompt_tokens": 260000,
                "context_length": 1000000,
            }

        def rotate_active_session(self, apply=False):
            return {
                "ok": True,
                "noop": False,
                "reason": "eligible",
                "total_message_count": 50,
                "fresh_tail_count": 12,
                "pre_tail_message_count": 38,
                "current_frontier_store_id": 2,
                "new_frontier_store_id": 3,
            }

    lcm_db = tmp_path / "lcm.db"
    conn = sqlite3.connect(lcm_db)
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE lcm_lifecycle_state (conversation_id TEXT, debt_kind TEXT, debt_size_estimate INTEGER, updated_at REAL)"
    )
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        ("compaction_telemetry:sess-1", '{"total_compactions": 0, "cache_state": "hot"}'),
    )
    conn.commit()
    conn.close()

    write_live_lcm_snapshot_for_engine(_Engine(), tmp_path)
    tele = lcm_telemetry(tmp_path, lcm_db, "sess-1")
    assert tele["live_snapshot_loaded"] is True
    assert tele["last_compression_status"] == "pending"
    assert tele["fresh_tail_count"] == 12
    assert tele["pre_tail_message_count"] == 38


def test_plugin_handler_rejects_arbitrary_args():
    plugin_dir = (
        Path(__file__).resolve().parents[2] / "plugins" / "hermes-context-visor"
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hermes_context_visor_plugin", plugin_dir / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    msg = mod.handle_visor_command("rm -rf /")
    assert "Refusing" in msg


def test_plugin_handler_launches_with_mock():
    plugin_dir = (
        Path(__file__).resolve().parents[2] / "plugins" / "hermes-context-visor"
    )
    import importlib.util
    from context_cockpit.launcher import LaunchResult

    spec = importlib.util.spec_from_file_location(
        "hermes_context_visor_plugin2", plugin_dir / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    fake = LaunchResult(
        True, "Opened Context Cockpit in browser.", method="browser_started", url="http://127.0.0.1:8421/"
    )

    with patch.object(mod, "_ensure_cockpit_importable"):
        with patch(
            "context_cockpit.launcher.launch_context_visor", return_value=fake
        ):
            msg = mod.handle_visor_command("")
    assert "Context Cockpit:" in msg
    assert "127.0.0.1:8421" in msg
