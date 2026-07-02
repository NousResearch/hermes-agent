import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_gateway_stability.py"
spec = importlib.util.spec_from_file_location("diagnose_gateway_stability", SCRIPT)
diag = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(diag)


def test_resolved_pre_restart_stalls_do_not_poison_current_recommendation(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "gui.log").write_text(
        "\n".join(
            [
                "2026-07-02 17:50:31,296 WARNING hermes_cli.web_server: event loop stalled 43.8s (GIL pressure suspected)",
                "2026-07-02 17:50:42,740 WARNING tui_gateway.ws: ws response send failed peer=127.0.0.1:61920 id=1367 method=setup.status",
                "2026-07-02 17:53:13,354 INFO tui_gateway.ws: ws closed peer=127.0.0.1:55168 reason=ready_send_failed messages=0",
                "2026-07-03 01:08:26,534 INFO hermes_cli.web_server: Desktop cron scheduler started (provider=builtin, interval=60s)",
                "2026-07-03 01:08:27,528 INFO tui_gateway.ws: ws accepted peer=127.0.0.1:61181",
            ]
        ),
        encoding="utf-8",
    )

    since = diag._latest_recovery_marker(log_dir)
    assert since is not None
    historical = diag._count_patterns(log_dir)
    current = diag._count_patterns(log_dir, since=since)
    timeline = diag._event_timeline(log_dir, since=since)

    assert historical["event_loop_stalled"] == 1
    assert historical["response_send_failed"] == 1
    assert historical["ready_send_failed"] == 1
    assert current["event_loop_stalled"] == 0
    assert current["response_send_failed"] == 0
    assert current["ready_send_failed"] == 0
    assert timeline["states"]["event_loop_stalled"]["status"] == "resolved"
    assert timeline["states"]["response_send_failed"]["status"] == "resolved"
    assert timeline["active"] == []
    assert "PASS" in diag._recommend(current, {"status": "ok"}, {"counts": {"slash_worker": 0}})


def test_current_window_active_stall_still_warns(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "gui.log").write_text(
        "\n".join(
            [
                "2026-07-03 01:08:26,534 INFO hermes_cli.web_server: Desktop cron scheduler started (provider=builtin, interval=60s)",
                "2026-07-03 01:08:27,528 INFO tui_gateway.ws: ws accepted peer=127.0.0.1:61181",
                "2026-07-03 01:09:00,000 WARNING tui_gateway.ws: ws write slow (loop stalled >10.0s) peer=127.0.0.1:61181 — frame left in flight",
            ]
        ),
        encoding="utf-8",
    )

    since = diag._latest_recovery_marker(log_dir)
    current = diag._count_patterns(log_dir, since=since)
    timeline = diag._event_timeline(log_dir, since=since)

    assert current["ws_write_slow"] == 1
    assert timeline["states"]["ws_write_slow"]["status"] == "active"
    assert "WARN" in diag._recommend(current, {"status": "ok"}, {"counts": {"slash_worker": 0}})
