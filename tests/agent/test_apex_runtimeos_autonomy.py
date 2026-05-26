import json

from agent.apex_runtimeos_autonomy import (
    apply_runtime_rewrite,
    build_runtime_rewrite_plan,
    build_runtimeos_health_report,
    build_runtimeos_health_watchdog_notice,
    _watchdog_config,
    execute_promotion_rollbacks,
    persist_autowrite_candidate,
    promote_autowrite_candidates,
    record_cron_dryrun_result,
    run_autopromotion_scheduler,
    score_autowrite_candidates,
    summarize_cron_dryrun_ledger,
)


class DummyCompressorAgent:
    compression_enabled = True
    context_compressor = object()

    def _compress_context(self, messages, system_message, approx_tokens=None, task_id=None):
        return messages[:1] + messages[-1:], "compressed-system"


def _heavy_checkpoint():
    return {
        "recommendations": {
            "status": "WATCH",
            "items": [
                {
                    "organ": "planner",
                    "code": "planner_context_heavy",
                    "severity": "warn",
                    "actions": ["compress_context", "review_budget"],
                    "applied": False,
                    "mutates_runtime": False,
                }
            ],
        }
    }


def test_runtime_rewrite_plan_disabled_by_default(monkeypatch):
    monkeypatch.delenv("APEX_RUNTIMEOS_AUTO_REWRITE_ENABLED", raising=False)
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    plan = build_runtime_rewrite_plan(_heavy_checkpoint())
    assert plan["enabled"] is False
    assert plan["actions"] == []
    assert plan["mutates_runtime"] is False


def test_runtime_rewrite_applies_compression_only_when_enabled(monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTO_REWRITE_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    result = apply_runtime_rewrite(
        agent=DummyCompressorAgent(),
        messages=messages,
        system_message="sys",
        active_system_prompt="sys",
        approx_tokens=999999,
        task_id="t1",
        checkpoint=_heavy_checkpoint(),
    )
    assert result["applied"] is True
    assert result["mutates_runtime"] is True
    assert result["before_message_count"] == 3
    assert result["after_message_count"] == 2
    assert result["messages"] == [messages[0], messages[-1]]


def test_autowrite_candidate_is_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTOWRITE_DIR", str(tmp_path))
    monkeypatch.delenv("APEX_RUNTIMEOS_AUTO_WRITE_ENABLED", raising=False)
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    result = persist_autowrite_candidate(stage="pre_completion", session_id="secret-session", recommendations=_heavy_checkpoint()["recommendations"])
    assert result["written"] is False
    assert not (tmp_path / "candidates.jsonl").exists()


def test_autowrite_candidate_writes_sanitized_promotion_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTOWRITE_DIR", str(tmp_path))
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTO_WRITE_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    result = persist_autowrite_candidate(
        stage="pre_completion",
        session_id="secret-session",
        recommendations={
            "status": "WATCH",
            "items": [
                {
                    "organ": "gene_selector",
                    "code": "gene_completion_incomplete",
                    "severity": "warn",
                    "actions": ["hold_gene_entry", "/Users/appleoppa/private"],
                    "reason": "raw prompt should not be stored",
                }
            ],
        },
    )
    assert result["written"] is True
    assert result["promotion_required"] is True
    assert result["applied_to_core_memory_or_skill"] is False
    raw = (tmp_path / "candidates.jsonl").read_text(encoding="utf-8")
    assert "/Users/appleoppa/private" not in raw
    assert "secret-session" not in raw
    record = json.loads(raw)
    assert record["schema"] == "ApexRuntimeOSAutoWriteCandidate/v1"
    assert record["promotion_required"] is True
    assert record["applied_to_core_memory_or_skill"] is False
    assert record["items"][0]["actions"][1] == "[REDACTED_PATH]"
    assert "reason" not in record["items"][0]


def test_promotion_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTOWRITE_DIR", str(tmp_path / "auto"))
    monkeypatch.delenv("APEX_RUNTIMEOS_PROMOTION_ENABLED", raising=False)
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    result = promote_autowrite_candidates(candidate_path=tmp_path / "missing.jsonl", target="memory")
    assert result["promoted"] == 0
    assert result["reason"] == "disabled_or_not_enforce"


def test_promote_candidate_to_memory_with_audit_and_dedup(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTOWRITE_DIR", str(tmp_path / "auto"))
    monkeypatch.setenv("APEX_RUNTIMEOS_PROMOTION_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    candidate_path = tmp_path / "candidates.jsonl"
    candidate = {
        "schema": "ApexRuntimeOSAutoWriteCandidate/v1",
        "promotion_required": True,
        "items": [{"code": "planner_context_heavy", "severity": "warn", "actions": ["compress_context"]}],
    }
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False) + "\n" + json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8")
    first = promote_autowrite_candidates(candidate_path=candidate_path, target="memory")
    second = promote_autowrite_candidates(candidate_path=candidate_path, target="memory")
    assert first["promoted"] == 1
    assert first["skipped"] == 1
    assert second["promoted"] == 0
    assert (home / "memories" / "MEMORY.md").exists()
    memory_text = (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    assert "planner_context_heavy" in memory_text
    assert "factual task completion" in memory_text
    audit_text = (tmp_path / "auto" / "promotions.jsonl").read_text(encoding="utf-8")
    assert "ApexRuntimeOSPromotion/v1" in audit_text
    assert "rollback" in audit_text
    assert str(home) not in audit_text


def test_promote_candidate_to_skill_with_rollback_metadata(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTOWRITE_DIR", str(tmp_path / "auto"))
    monkeypatch.setenv("APEX_RUNTIMEOS_PROMOTION_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    candidate_path = tmp_path / "candidates.jsonl"
    candidate = {
        "schema": "ApexRuntimeOSAutoWriteCandidate/v1",
        "promotion_required": True,
        "items": [{"code": "gene_completion_incomplete", "severity": "warn", "actions": ["hold_gene_entry"]}],
    }
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8")
    result = promote_autowrite_candidates(candidate_path=candidate_path, target="skill")
    assert result["promoted"] == 1
    skill_path = home / "skills" / "apex-runtimeos-autogen" / "SKILL.md"
    assert skill_path.exists()
    text = skill_path.read_text(encoding="utf-8")
    assert "gene_completion_incomplete" in text
    assert "不包含原始对话、路径或凭据" in text
    audit_text = (tmp_path / "auto" / "promotions.jsonl").read_text(encoding="utf-8")
    assert "path_hash" in audit_text
    assert str(skill_path) not in audit_text


def test_stability_score_requires_min_occurrences(tmp_path):
    candidate_path = tmp_path / "candidates.jsonl"
    c1 = {"schema": "ApexRuntimeOSAutoWriteCandidate/v1", "promotion_required": True, "items": [{"code": "planner_context_heavy", "severity": "warn"}]}
    c2 = {"schema": "ApexRuntimeOSAutoWriteCandidate/v1", "promotion_required": True, "items": [{"code": "planner_context_heavy", "severity": "warn"}]}
    c3 = {"schema": "ApexRuntimeOSAutoWriteCandidate/v1", "promotion_required": True, "items": [{"code": "gene_completion_incomplete", "severity": "warn"}]}
    candidate_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [c1, c2, c3]) + "\n", encoding="utf-8")
    score = score_autowrite_candidates(candidate_path=candidate_path, min_occurrences=2)
    assert score["ready_count"] == 1
    assert score["ready"][0]["key"] == "planner_context_heavy"
    assert score["ready"][0]["count"] == 2


def test_autopromotion_scheduler_disabled_after_scoring(tmp_path, monkeypatch):
    candidate_path = tmp_path / "candidates.jsonl"
    candidate = {"schema": "ApexRuntimeOSAutoWriteCandidate/v1", "promotion_required": True, "items": [{"code": "planner_context_heavy", "severity": "warn"}]}
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False) + "\n" + json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.delenv("APEX_RUNTIMEOS_AUTOPROMOTE_ENABLED", raising=False)
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    result = run_autopromotion_scheduler(candidate_path=candidate_path, min_occurrences=2)
    assert result["promoted"] == 0
    assert result["score"]["ready_count"] == 1
    assert result["reason"] == "disabled_or_not_enforce"


def test_autopromotion_scheduler_promotes_stable_memory(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTOWRITE_DIR", str(tmp_path / "auto"))
    monkeypatch.setenv("APEX_RUNTIMEOS_AUTOPROMOTE_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    candidate_path = tmp_path / "candidates.jsonl"
    stable = {"schema": "ApexRuntimeOSAutoWriteCandidate/v1", "promotion_required": True, "items": [{"code": "planner_context_heavy", "severity": "warn", "actions": ["compress_context"]}]}
    unstable = {"schema": "ApexRuntimeOSAutoWriteCandidate/v1", "promotion_required": True, "items": [{"code": "gene_completion_incomplete", "severity": "warn"}]}
    candidate_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [stable, stable, unstable]) + "\n", encoding="utf-8")
    result = run_autopromotion_scheduler(candidate_path=candidate_path, target="memory", min_occurrences=2)
    assert result["promoted"] == 1
    assert result["score"]["ready_count"] == 1
    memory_text = (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    assert "planner_context_heavy" in memory_text
    assert "gene_completion_incomplete" not in memory_text


def test_rollback_disabled_defaults_to_dry_run(tmp_path, monkeypatch):
    audit_path = tmp_path / "promotions.jsonl"
    record = {"schema": "ApexRuntimeOSPromotion/v1", "target": "memory", "success": True, "content_hash": "h1", "rollback": {"action": "remove", "old_text": "x"}, "rollback_status": "pending"}
    audit_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.delenv("APEX_RUNTIMEOS_ROLLBACK_ENABLED", raising=False)
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    result = execute_promotion_rollbacks(audit_path=audit_path, target="memory", dry_run=False)
    assert result["dry_run"] is True
    assert result["rolled_back"] == 1
    assert result["events"][0]["status"] == "dry_run"


def test_rollback_memory_remove_executes_when_enabled(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("APEX_RUNTIMEOS_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    mem = home / "memories" / "MEMORY.md"
    mem.parent.mkdir(parents=True)
    mem.write_text("entry to remove", encoding="utf-8")
    audit_path = tmp_path / "promotions.jsonl"
    record = {"schema": "ApexRuntimeOSPromotion/v1", "target": "memory", "success": True, "content_hash": "h1", "rollback": {"action": "remove", "old_text": "entry to remove"}, "rollback_status": "pending"}
    audit_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    result = execute_promotion_rollbacks(audit_path=audit_path, target="memory", dry_run=False)
    assert result["rolled_back"] == 1
    assert "entry to remove" not in mem.read_text(encoding="utf-8")
    raw = audit_path.read_text(encoding="utf-8")
    assert "ApexRuntimeOSRollbackEvent/v1" in raw
    assert "\"rollback_status\":\"done\"" in raw
    second = execute_promotion_rollbacks(audit_path=audit_path, target="memory", dry_run=False)
    assert second["rolled_back"] == 0
    assert second["events"][0]["reason"] == "already_done"


def test_rollback_skill_delete_checks_current_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    skill = tmp_path / "SKILL.md"
    skill.write_text("new content", encoding="utf-8")
    from agent.apex_runtimeos_autonomy import _hash_text
    audit_path = tmp_path / "promotions.jsonl"
    record = {"schema": "ApexRuntimeOSPromotion/v1", "target": "skill", "success": True, "content_hash": "h2", "rollback": {"action": "delete", "path": str(skill), "current_content_hash": _hash_text("different")}, "rollback_status": "pending"}
    audit_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    result = execute_promotion_rollbacks(audit_path=audit_path, target="skill", dry_run=False)
    assert result["rolled_back"] == 0
    assert result["events"][0]["reason"] == "content_hash_mismatch"
    assert skill.exists()


def test_rollback_skill_restore_executes_when_hash_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("APEX_RUNTIMEOS_ROLLBACK_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    skill = tmp_path / "SKILL.md"
    skill.write_text("new content", encoding="utf-8")
    from agent.apex_runtimeos_autonomy import _hash_text
    audit_path = tmp_path / "promotions.jsonl"
    record = {"schema": "ApexRuntimeOSPromotion/v1", "target": "skill", "success": True, "content_hash": "h3", "rollback": {"action": "restore", "path": str(skill), "current_content_hash": _hash_text("new content"), "old_content": "old content"}, "rollback_status": "pending"}
    audit_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    result = execute_promotion_rollbacks(audit_path=audit_path, target="skill", dry_run=False)
    assert result["rolled_back"] == 1
    assert skill.read_text(encoding="utf-8") == "old content"


def test_apex_runtimeos_cli_rollback_dry_run_json(monkeypatch, tmp_path):
    from hermes_cli.apex_runtimeos import run_apex_runtimeos_cli
    monkeypatch.setenv("APEX_RUNTIMEOS_PROMOTION_AUDIT_PATH", str(tmp_path / "promotions.jsonl"))
    output = run_apex_runtimeos_cli(["rollback", "--json"])
    data = json.loads(output)
    assert data["object"] == "hermes.apex_runtimeos.rollback"
    assert data["result"]["dry_run"] is True


def test_cron_dryrun_ledger_records_redacted_idempotent_summary(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    payload = {
        "task": "apex-runtimeos-autopromote-dryrun",
        "enabled": False,
        "mode": "warn",
        "ready_count": 0,
        "secret": "sk-should-not-appear",
        "path": str(tmp_path / "private"),
    }
    first = record_cron_dryrun_result(task="apex-runtimeos-autopromote-dryrun", result=payload, ledger_path=ledger)
    second = record_cron_dryrun_result(task="apex-runtimeos-autopromote-dryrun", result=payload, ledger_path=ledger)
    assert first["ledger_key"] == second["ledger_key"]
    assert second["count"] == 2
    raw = ledger.read_text(encoding="utf-8")
    assert "sk-should-not-appear" not in raw
    assert str(tmp_path) not in raw
    summary = summarize_cron_dryrun_ledger(ledger_path=ledger)
    assert summary["unique_keys"] == 1
    assert summary["tasks"]["apex-runtimeos-autopromote-dryrun"]["total_count"] == 2


def test_cron_dryrun_ledger_bad_lines_quarantined_only_when_enabled(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    record_cron_dryrun_result(task="apex-runtimeos-rollback-dryrun", result={"task": "apex-runtimeos-rollback-dryrun", "mode": "warn"}, ledger_path=ledger)
    ledger.write_text(ledger.read_text(encoding="utf-8") + "not-json-secret-token\n", encoding="utf-8")
    summary = summarize_cron_dryrun_ledger(ledger_path=ledger)
    assert summary["bad_lines"] == 1
    assert summary["quarantine_exists"] is False
    assert "not-json-secret-token" in ledger.read_text(encoding="utf-8")

    monkeypatch.setenv("APEX_RUNTIMEOS_CRON_LEDGER_REPAIR_ENABLED", "1")
    monkeypatch.setenv("APEX_RUNTIMEOS_GATE_MODE", "enforce")
    repaired = summarize_cron_dryrun_ledger(ledger_path=ledger)
    assert repaired["bad_lines"] == 1
    assert repaired["repair_enabled"] is True
    assert (tmp_path / "ledger.jsonl.bad").exists()
    assert "not-json-secret-token" not in ledger.read_text(encoding="utf-8")
    assert "not-json-secret-token" not in (tmp_path / "ledger.jsonl.bad").read_text(encoding="utf-8")


def test_runtimeos_health_report_threshold_alerts():
    report = build_runtimeos_health_report({
        "pending_rollbacks": 2,
        "stable_ready_count": 1,
        "autopromote_enabled": False,
        "cron_dryrun": {"bad_lines": 1, "unique_keys": 3},
    })
    assert report["status"] == "WATCH"
    assert report["alert_count"] == 3
    codes = {item["code"] for item in report["alerts"]}
    assert "cron_ledger_bad_lines" in codes
    assert "pending_rollbacks" in codes
    assert "stable_candidates_waiting" in codes
    assert report["side_effects"] == "read_only_report"


def test_runtimeos_health_watchdog_notice_suppresses_repeat_watch(tmp_path):
    payload = {
        "status": {
            "health_report": {
                "status": "WATCH",
                "alert_count": 1,
                "alerts": [{"code": "cron_ledger_bad_lines", "severity": "warn", "count": 1}],
            }
        }
    }
    state_path = tmp_path / "watchdog_state.json"
    first = build_runtimeos_health_watchdog_notice(payload, state_path=state_path, cooldown_seconds=2700, now_ts=1000.0)
    second = build_runtimeos_health_watchdog_notice(payload, state_path=state_path, cooldown_seconds=2700, now_ts=1100.0)
    third = build_runtimeos_health_watchdog_notice({"status": {"health_report": {"status": "OK", "alerts": []}}}, state_path=state_path, cooldown_seconds=2700, now_ts=1200.0)
    assert first["should_emit"] is True
    assert first["reason"] == "alert"
    assert second["should_emit"] is False
    assert second["reason"] == "cooldown"
    assert third["should_emit"] is False
    assert third["reason"] == "ok"
    assert not state_path.exists()


def test_runtimeos_health_watchdog_notice_uses_configured_cooldown(tmp_path):
    payload = {
        "status": {
            "health_report": {
                "status": "WATCH",
                "alert_count": 1,
                "alerts": [{"code": "cron_ledger_bad_lines", "severity": "warn", "count": 1}],
            }
        }
    }
    state_path = tmp_path / "watchdog_state.json"
    first = build_runtimeos_health_watchdog_notice(payload, state_path=state_path, cooldown_seconds=60, now_ts=1000.0)
    second = build_runtimeos_health_watchdog_notice(payload, state_path=state_path, cooldown_seconds=60, now_ts=1030.0)
    third = build_runtimeos_health_watchdog_notice(payload, state_path=state_path, cooldown_seconds=60, now_ts=1070.0)
    assert first["should_emit"] is True
    assert second["should_emit"] is False
    assert second["reason"] == "cooldown"
    assert third["should_emit"] is True
    assert third["reason"] == "alert"


def test_watchdog_config_exposes_default_values(monkeypatch):
    monkeypatch.delenv("APEX_RUNTIMEOS_WATCHDOG_STATE_PATH", raising=False)
    monkeypatch.delenv("APEX_RUNTIMEOS_WATCHDOG_COOLDOWN_SECONDS", raising=False)
    cfg = _watchdog_config()
    assert cfg["schema"] == "ApexRuntimeOSWatchdogConfig/v1"
    assert cfg["state_configured"] is False
    assert cfg["cooldown_seconds"] == 2700
