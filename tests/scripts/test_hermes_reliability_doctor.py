import json
import sqlite3

from scripts import hermes_reliability_doctor as doctor


def test_parse_rclone_config_reports_remote_names_only(tmp_path):
    config_path = tmp_path / "rclone.conf"
    config_path.write_text(
        """
[onedrive]
type = onedrive
drive_type = personal
extra = SENSITIVE_VALUE

[familyhub-onedrive]
type = onedrive
drive_type = personal

[littleton-google-drive]
type = drive
scope = drive
""".strip(),
        encoding="utf-8",
    )

    result = doctor.parse_rclone_config(config_path)

    assert result == {
        "exists": True,
        "remote_count": 3,
        "remotes": [
            {"name": "onedrive", "type": "onedrive", "drive_type": "personal"},
            {"name": "familyhub-onedrive", "type": "onedrive", "drive_type": "personal"},
            {"name": "littleton-google-drive", "type": "drive", "drive_type": None},
        ],
    }
    assert "SENSITIVE_VALUE" not in json.dumps(result)


def test_inspect_active_task_store_counts_without_session_keys(tmp_path):
    store_path = tmp_path / "session_active_tasks.json"
    store_path.write_text(
        json.dumps(
            {
                "discord:channel:thread:private-session-key": {
                    "session_key": "discord:channel:thread:private-session-key",
                    "mode": "foreground_session",
                    "status": "active",
                    "final_report_status": "pending",
                    "repo_path": "/repo/a",
                },
                "discord:channel:other:private-session-key": {
                    "session_key": "discord:channel:other:private-session-key",
                    "mode": "approved_execute",
                    "status": "interrupted",
                    "final_report_status": "failed",
                    "repo_path": "/repo/b",
                },
            }
        ),
        encoding="utf-8",
    )

    result = doctor.inspect_active_task_store(store_path)

    assert result == {
        "exists": True,
        "parsed": True,
        "record_count": 2,
        "foreground_count": 1,
        "status_counts": {"active": 1, "interrupted": 1},
        "mode_counts": {"approved_execute": 1, "foreground_session": 1},
        "final_report_counts": {"failed": 1, "pending": 1},
        "foreground_missing_task_count": 1,
        "stale_active_foreground_count": 1,
        "updated_at_age_buckets": {"bad_or_missing": 2},
    }
    rendered = json.dumps(result)
    assert "private-session-key" not in rendered
    assert "discord:channel" not in rendered


def test_inspect_active_task_store_flags_stale_foreground_records_without_text(tmp_path):
    store_path = tmp_path / "session_active_tasks.json"
    store_path.write_text(
        json.dumps(
            {
                "discord:channel:thread:private-session-key": {
                    "session_key": "discord:channel:thread:private-session-key",
                    "mode": "foreground_session",
                    "status": "active",
                    "repo_path": "/repo/a",
                    "task_summary": "private task body",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                },
                "discord:channel:other:private-session-key": {
                    "session_key": "discord:channel:other:private-session-key",
                    "mode": "foreground_session",
                    "status": "active",
                    "repo_path": "/repo/b",
                    "updated_at": "not-a-date",
                },
            }
        ),
        encoding="utf-8",
    )

    result = doctor.inspect_active_task_store(store_path)

    assert result["foreground_count"] == 2
    assert result["foreground_missing_task_count"] == 1
    assert result["stale_active_foreground_count"] == 2
    assert result["updated_at_age_buckets"] == {
        "bad_or_missing": 1,
        "stale": 1,
    }
    rendered = json.dumps(result)
    assert "private-session-key" not in rendered
    assert "private task body" not in rendered


def test_inspect_goal_store_uses_read_only_counts(tmp_path):
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?)",
        ("goal:session-a", json.dumps({"active": True, "done": False})),
    )
    conn.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?)",
        ("goal:session-b", json.dumps({"active": False, "done": True})),
    )
    conn.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?)",
        ("scheduler:last", "ignored"),
    )
    conn.commit()
    conn.close()

    result = doctor.inspect_goal_store(db_path)

    assert result == {
        "exists": True,
        "readable": True,
        "goal_count": 2,
        "active_count": 1,
        "done_count": 1,
        "paused_count": 0,
        "status_counts": {"active": 1, "done": 1},
        "field_presence": {},
    }
    assert "session-a" not in json.dumps(result)


def test_inspect_goal_store_reports_safe_metadata_without_goal_text(tmp_path):
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?)",
        (
            "goal:private-session-id",
            json.dumps(
                {
                    "goal": "private mission body",
                    "status": "active",
                    "created_at": 1.0,
                    "last_turn_at": 2.0,
                    "turns_used": 3,
                    "max_turns": 9,
                    "subgoals": ["private subgoal"],
                }
            ),
        ),
    )
    conn.commit()
    conn.close()

    result = doctor.inspect_goal_store(db_path)

    assert result["status_counts"] == {"active": 1}
    assert result["field_presence"]["goal"] == 1
    assert result["field_presence"]["subgoals"] == 1
    rendered = json.dumps(result)
    assert "private mission body" not in rendered
    assert "private subgoal" not in rendered
    assert "private-session-id" not in rendered


def test_check_storage_policy_presence_reads_only_canonical_markers(tmp_path):
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    (memories / "USER.md").write_text(
        "local machines stay lean\nonedrive: working\nfamilyhub-onedrive: background\n"
        "outside OneDrive AI requires explicit approval\n",
        encoding="utf-8",
    )
    (memories / "MEMORY.md").write_text(
        "onedrive:AI/\nfamilyhub-onedrive:Family Hub/App Storage/\n"
        "littleton-google-drive:\ngemini-drive:\ncommand-based rclone\n",
        encoding="utf-8",
    )
    brain = tmp_path / "ai-ops-brain"
    runbook = brain / "ai-ops/storage-hygiene/global-storage-drive-rules-2026-06-02.md"
    runbook.parent.mkdir(parents=True)
    runbook.write_text(
        "14 days\n30 days\n48 hours\nkeeper\nreview-package\ndry-run\nproof-gated\n",
        encoding="utf-8",
    )

    result = doctor.check_storage_policy_presence(hermes_home, brain)

    assert result["all_required_markers_present"] is True
    assert result["files"]["USER.md"]["exists"] is True
    assert result["files"]["MEMORY.md"]["exists"] is True
    assert result["files"]["storage_runbook"]["exists"] is True


def test_check_quality_policy_presence_reports_markers_without_prompt_text(tmp_path):
    repo = tmp_path / "repo"
    gateway = repo / "gateway"
    gateway.mkdir(parents=True)
    (gateway / "session.py").write_text(
        "QUALITY_LANE_POLICY_MARKERS = ('implementation lane', 'verification lane')\n"
        "def render_quality_lane_policy_for_prompt(): return ''\n"
        "def build_session_context_prompt():\n"
        "    return render_quality_lane_policy_for_prompt()\n",
        encoding="utf-8",
    )
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text("WAHA\nSignal Room\n", encoding="utf-8")
    (memories / "USER.md").write_text("video production\n", encoding="utf-8")

    result = doctor.check_quality_policy_presence(repo, hermes_home)

    assert result["injection_path_enabled"] is True
    assert result["memory_markers"]["WAHA"] is True
    assert result["memory_markers"]["Signal Room"] is True
    rendered = json.dumps(result)
    assert "implementation lane" not in rendered
    assert "verification lane" not in rendered


def test_evaluate_runtime_topology_flags_cli_split_brain():
    result = doctor.evaluate_runtime_topology(
        expected_runtime_checkout="/clean",
        service_working_directory="/clean",
        proc_cwd="/clean",
        module_paths={
            "hermes_cli.main": "/clean/hermes_cli/main.py",
            "gateway.run": "/clean/gateway/run.py",
        },
        cli_project="/dirty",
    )

    assert result["service_matches_expected"] is True
    assert result["process_matches_expected"] is True
    assert result["modules_match_expected"] is True
    assert result["cli_project_matches_expected"] is False
    assert result["split_brain_risk"] is True
