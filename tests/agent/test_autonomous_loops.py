import json
import sqlite3

from agent.autonomous_loops import audit_autonomous_loops


def test_autonomous_loop_audit_inventory_is_metadata_only(tmp_path):
    hermes_home = tmp_path / "hermes"
    cron_dir = hermes_home / "cron"
    cron_dir.mkdir(parents=True)
    (cron_dir / "jobs.json").write_text(json.dumps({
        "jobs": [
            {
                "id": "job-private-1",
                "name": "Private production monitor",
                "prompt": "Check prod token sk-" + "x" * 32 + " and report every time",
                "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
                "enabled": True,
                "deliver": ["origin", "all"],
                "enabled_toolsets": ["terminal", "file"],
                "no_agent": False,
            },
            {
                "id": "job-watchdog-2",
                "name": "GPU watchdog",
                "prompt": None,
                "schedule": {"kind": "interval", "minutes": 10, "display": "every 10m"},
                "enabled": True,
                "deliver": "local",
                "script": "gpu-watchdog.sh",
                "no_agent": True,
            },
            {
                "id": "job-unbounded-3",
                "name": "Daily digest",
                "prompt": "Summarize news when there is something new. Use [SILENT] otherwise.",
                "schedule": {"kind": "cron", "expr": "0 9 * * *", "display": "0 9 * * *"},
                "enabled": True,
                "deliver": "telegram:-100123:55",
                "no_agent": False,
            },
            {
                "id": "job-disabled-4",
                "name": "Disabled local fallback",
                "prompt": "Disabled one-shot",
                "schedule": {"kind": "once", "run_at": "2099-01-01T00:00:00Z"},
                "enabled": False,
                "deliver": None,
                "no_agent": False,
            },
        ],
    }))

    db_path = hermes_home / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?)",
        (
            "goal:private-session",
            json.dumps({
                "goal": "Fix private launch blocker with password hunter2",
                "status": "active",
                "turns_used": 2,
                "max_turns": 50,
                "subgoals": ["private subgoal"],
            }),
        ),
    )
    conn.commit()
    conn.close()

    summary = audit_autonomous_loops(hermes_home=hermes_home)

    assert summary["content_policy"] == "metadata_only"
    assert summary["mode"] == "audit_only_no_create"
    assert summary["cron"]["job_count"] == 4
    assert summary["cron"]["active_count"] == 3
    assert summary["cron"]["recurring_count"] == 3
    assert summary["cron"]["script_only_watchdog_count"] == 1
    assert summary["cron"]["agent_job_count"] == 3
    assert summary["cron"]["delivery_counts"]["all"] == 1
    assert summary["cron"]["delivery_counts"]["local"] == 2
    assert summary["goals"]["total_goal_rows"] == 1
    assert summary["goals"]["active_goal_count"] == 1

    codes = {issue["code"] for issue in summary["issues"]}
    assert {
        "loop_missing_silence_condition",
        "loop_broad_delivery",
        "loop_side_effect_policy_missing",
        "loop_tool_scope_unbounded",
    } <= codes

    guidance_ids = {item["id"] for item in summary["guidance"]}
    assert {"cron_agent_prompt", "script_only_watchdog", "side_effect_approval"} <= guidance_ids

    raw = json.dumps(summary, sort_keys=True)
    assert "sk-" not in raw
    assert "hunter2" not in raw
    assert "Private production monitor" not in raw
    assert "gpu-watchdog.sh" not in raw
    assert "telegram:-100123" not in raw
    assert "private-session" not in raw


def test_autonomous_loop_audit_empty_state_is_structural(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()

    summary = audit_autonomous_loops(hermes_home=hermes_home)

    assert summary["cron"]["job_count"] == 0
    assert summary["goals"]["total_goal_rows"] == 0
    assert summary["issues"] == []
