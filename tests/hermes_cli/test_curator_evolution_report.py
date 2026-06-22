"""Tests for `hermes curator evolution-report` and skill evaluation hooks."""

from types import SimpleNamespace


def test_evolution_report_command_prints_path(monkeypatch, capsys, tmp_path):
    import hermes_cli.curator as curator_cli

    monkeypatch.setattr(
        "agent.evolution_report.generate_report",
        lambda: {
            "markdown_path": str(tmp_path / "REPORT.md"),
            "json_path": str(tmp_path / "run.json"),
            "report_id": "20260622-000000",
            "summary": {"total_skills": 1, "evaluated_skills": 1, "unevaluated_skills": 0},
        },
    )

    rc = curator_cli._cmd_evolution_report(SimpleNamespace())

    out = capsys.readouterr().out
    assert rc == 0
    assert "evolution report" in out.lower()
    assert str(tmp_path / "REPORT.md") in out
    assert "total=1" in out


def test_evaluate_command_records_passed_skill(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli

    calls = []
    monkeypatch.setattr(
        "tools.skill_usage.record_evaluation",
        lambda skill, *, passed, summary="": calls.append((skill, passed, summary)),
    )
    monkeypatch.setattr("tools.skill_manager_tool._find_skill", lambda name: object())

    rc = curator_cli._cmd_evaluate(
        SimpleNamespace(skill="test-driven-development", outcome="pass", summary="pytest passed")
    )

    assert rc == 0
    assert calls == [("test-driven-development", True, "pytest passed")]
    assert "successful evaluation" in capsys.readouterr().out


def test_evaluate_command_records_failed_skill(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli

    calls = []
    monkeypatch.setattr(
        "tools.skill_usage.record_evaluation",
        lambda skill, *, passed, summary="": calls.append((skill, passed, summary)),
    )
    monkeypatch.setattr("tools.skill_manager_tool._find_skill", lambda name: object())

    rc = curator_cli._cmd_evaluate(
        SimpleNamespace(skill="test-driven-development", outcome="fail", summary="user rejected output")
    )

    assert rc == 0
    assert calls == [("test-driven-development", False, "user rejected output")]
    assert "failed evaluation" in capsys.readouterr().out


def test_evaluate_command_rejects_missing_skill(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli

    monkeypatch.setattr("tools.skill_manager_tool._find_skill", lambda name: None)

    rc = curator_cli._cmd_evaluate(
        SimpleNamespace(skill="ghost", outcome="pass", summary="not real")
    )

    assert rc == 1
    assert "not found" in capsys.readouterr().out.lower()

def test_trace_start_command_prints_run_id(monkeypatch, capsys, tmp_path):
    import hermes_cli.curator as curator_cli

    monkeypatch.setattr(
        "agent.evolution_trace.start_trace",
        lambda **kwargs: {
            "run_id": "run-123",
            "trace_path": str(tmp_path / "trace.json"),
            "trace_dir": str(tmp_path),
        },
    )

    rc = curator_cli._cmd_trace_start(
        SimpleNamespace(mission="hermes-evolution", session_id="s1", design="measured loop")
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "run-123" in out
    assert str(tmp_path / "trace.json") in out


def test_trace_start_command_can_use_mission_manifest(monkeypatch, capsys, tmp_path):
    import hermes_cli.curator as curator_cli

    calls = []
    monkeypatch.setattr(
        "agent.evolution_trace.start_trace_from_manifest",
        lambda mission, *, session_id=None: calls.append((mission, session_id))
        or {
            "run_id": "manifest-run",
            "trace_path": str(tmp_path / "trace.json"),
            "trace_dir": str(tmp_path),
            "manifest_path": str(tmp_path / "mission.yaml"),
        },
    )

    rc = curator_cli._cmd_trace_start(
        SimpleNamespace(mission="genesis", session_id="s1", design=None, from_manifest=True)
    )

    assert rc == 0
    assert calls == [("genesis", "s1")]
    assert "manifest-run" in capsys.readouterr().out


def test_trace_event_command_records_event(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli

    calls = []
    monkeypatch.setattr(
        "agent.evolution_trace.record_event",
        lambda run_id, event_type, **fields: calls.append((run_id, event_type, fields))
        or {"run_id": run_id, "event_count": 1, "trace_path": "trace.json"},
    )

    rc = curator_cli._cmd_trace_event(
        SimpleNamespace(
            run_id="run-123",
            event_type="tool_call",
            tool="terminal",
            path=None,
            command=None,
            signal=None,
            status="success",
            summary="pytest passed",
        )
    )

    assert rc == 0
    assert calls == [("run-123", "tool_call", {"tool": "terminal", "status": "success", "summary": "pytest passed"})]
    assert "event_count=1" in capsys.readouterr().out


def test_trace_finish_command_sets_status(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli

    calls = []
    monkeypatch.setattr(
        "agent.evolution_trace.finish_trace",
        lambda run_id, *, final_status, summary="": calls.append((run_id, final_status, summary))
        or {"run_id": run_id, "final_status": final_status, "trace_path": "trace.json"},
    )

    rc = curator_cli._cmd_trace_finish(
        SimpleNamespace(run_id="run-123", final_status="success", summary="done")
    )

    assert rc == 0
    assert calls == [("run-123", "success", "done")]
    assert "success" in capsys.readouterr().out

def test_trace_event_command_records_evidence_source_fields(monkeypatch, capsys):
    import hermes_cli.curator as curator_cli

    calls = []
    monkeypatch.setattr(
        "agent.evolution_trace.record_event",
        lambda run_id, event_type, **fields: calls.append((run_id, event_type, fields))
        or {"run_id": run_id, "event_count": 1, "trace_path": "trace.json"},
    )

    rc = curator_cli._cmd_trace_event(
        SimpleNamespace(
            run_id="run-obsidian",
            event_type="evidence_source",
            tool=None,
            path="C:/vault/_master-index.md",
            command=None,
            signal=None,
            status=None,
            summary="MUST READ checked",
            source_type="obsidian_note",
            title="Master Index",
            gate=None,
            decision=None,
        )
    )

    assert rc == 0
    assert calls == [(
        "run-obsidian",
        "evidence_source",
        {
            "path": "C:/vault/_master-index.md",
            "summary": "MUST READ checked",
            "source_type": "obsidian_note",
            "title": "Master Index",
        },
    )]
    assert "event_count=1" in capsys.readouterr().out


def test_trace_event_command_records_gate_and_decision_fields(monkeypatch):
    import hermes_cli.curator as curator_cli

    calls = []
    monkeypatch.setattr(
        "agent.evolution_trace.record_event",
        lambda run_id, event_type, **fields: calls.append((run_id, event_type, fields))
        or {"run_id": run_id, "event_count": len(calls), "trace_path": "trace.json"},
    )

    curator_cli._cmd_trace_event(
        SimpleNamespace(
            run_id="run-1", event_type="gate", tool=None, path=None, command=None, signal=None,
            status="passed", summary="master index read", source_type=None, title=None,
            gate="obsidian_master_index_checked", decision=None,
        )
    )
    curator_cli._cmd_trace_event(
        SimpleNamespace(
            run_id="run-1", event_type="decision", tool=None, path=None, command=None, signal=None,
            status="hold", summary="raw evidence required", source_type=None, title=None,
            gate=None, decision="HOLD",
        )
    )

    assert calls == [
        ("run-1", "gate", {"status": "passed", "summary": "master index read", "gate": "obsidian_master_index_checked"}),
        ("run-1", "decision", {"status": "hold", "summary": "raw evidence required", "decision": "HOLD"}),
    ]



def test_trace_event_command_records_context_pack_fields(monkeypatch):
    import hermes_cli.curator as curator_cli

    calls = []
    monkeypatch.setattr(
        "agent.evolution_trace.record_event",
        lambda run_id, event_type, **fields: calls.append((run_id, event_type, fields))
        or {"run_id": run_id, "event_count": 1, "trace_path": "trace.json"},
    )

    rc = curator_cli._cmd_trace_event(
        SimpleNamespace(
            run_id="run-1",
            event_type="context_pack",
            tool=None,
            path="C:/tmp/pack.md",
            command=None,
            signal=None,
            status="hold",
            summary="pack compiled",
            source_type=None,
            title=None,
            gate=None,
            decision=None,
            topic="GENESIS",
            source_count="3",
            missing_evidence="runtime proof",
        )
    )

    assert rc == 0
    assert calls == [(
        "run-1",
        "context_pack",
        {
            "path": "C:/tmp/pack.md",
            "status": "hold",
            "summary": "pack compiled",
            "topic": "GENESIS",
            "source_count": "3",
            "missing_evidence": "runtime proof",
        },
    )]


def test_trace_event_command_records_change_packet_fields(monkeypatch):
    import hermes_cli.curator as curator_cli

    calls = []
    monkeypatch.setattr(
        "agent.evolution_trace.record_event",
        lambda run_id, event_type, **fields: calls.append((run_id, event_type, fields))
        or {"run_id": run_id, "event_count": 1, "trace_path": "trace.json"},
    )

    result = curator_cli._cmd_trace_event(
        SimpleNamespace(
            run_id="run-1",
            event_type="change_packet",
            tool=None,
            path="packet.md",
            command=None,
            signal=None,
            source_type=None,
            title=None,
            gate=None,
            decision=None,
            topic="Hermes Change Packet",
            objective="Bound implementation",
            source_count=None,
            missing_evidence=None,
            status="hold",
            summary="packet generated",
        )
    )

    assert result == 0
    assert calls == [
        (
            "run-1",
            "change_packet",
            {
                "path": "packet.md",
                "topic": "Hermes Change Packet",
                "objective": "Bound implementation",
                "status": "hold",
                "summary": "packet generated",
            },
        )
    ]
