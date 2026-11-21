"""Behaviour contracts for the prompt-guidance ablation harness (#134640)."""
import importlib.util
import json

import pytest
import yaml
from pathlib import Path

def _harness_module():
    path = Path(__file__).parents[2] / "evals" / "toolperf_abeval" / "ab_eval.py"
    spec = importlib.util.spec_from_file_location("toolperf_abeval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_guidance_arm_uses_isolated_home_and_only_changes_guidance(tmp_path, monkeypatch):
    harness = _harness_module()
    source_home = tmp_path / "source-home"
    source_home.mkdir()
    (source_home / ".env").write_text("OPENROUTER_API_KEY=test\n", encoding="utf-8")
    (source_home / "config.yaml").write_text(
        "model:\n  provider: openrouter\nagent:\n  task_completion_guidance: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(harness, "HOME", source_home)

    arm_home = harness._home_for_guidance_arm(tmp_path / "run", "execution_off")

    config = (arm_home / "config.yaml").read_text(encoding="utf-8")
    assert "model:\n  provider: openrouter\n" in config
    assert "agent:\n  task_completion_guidance: true\n" in config
    assert "  tool_use_enforcement: true\n" in config
    assert "  execution_guidance: false\n" in config
    assert (arm_home / ".env").read_text(encoding="utf-8") == "OPENROUTER_API_KEY=test\n"
    assert (source_home / "config.yaml").read_text(encoding="utf-8") == (
        "model:\n  provider: openrouter\nagent:\n  task_completion_guidance: true\n"
    )


def test_guidance_tasks_score_both_required_tool_work_and_trivial_edit(tmp_path):
    harness = _harness_module()
    harness.make_sandbox(tmp_path)
    project = tmp_path / "proj"

    assert harness.SUCCESS["guidance_tool_action"]("TOKEN=tool_action_2c47", tmp_path)
    (project / "release.txt").write_text("release=ready\n", encoding="utf-8")
    assert harness.SUCCESS["guidance_write_readback"]("release=ready", tmp_path)
    (project / "flag.txt").write_text("enabled\n", encoding="utf-8")
    assert harness.SUCCESS["guidance_trivial_edit"]("DONE", tmp_path)



@pytest.mark.parametrize("agent", [
    "agent:\n    task_completion_guidance: false\n",
    "agent: {task_completion_guidance: false}\n",
])
def test_guidance_arm_preserves_valid_yaml_settings(tmp_path, monkeypatch, agent):
    harness = _harness_module()
    home = tmp_path / "source"
    home.mkdir()
    original = "# preserve this comment\nmodel: custom/model\n" + agent
    (home / "config.yaml").write_text(original)
    monkeypatch.setattr(harness, "HOME", home)
    output = harness._home_for_guidance_arm(tmp_path / "run", "execution_off")
    config = yaml.safe_load((output / "config.yaml").read_text())
    assert config["model"] == "custom/model"
    assert config["agent"] == {
        "task_completion_guidance": False,
        "tool_use_enforcement": True,
        "execution_guidance": False,
    }
    assert "# preserve this comment" in (output / "config.yaml").read_text()
    assert (home / "config.yaml").read_text() == original


def test_report_rejects_readback_without_tool_trace(tmp_path, monkeypatch, capsys):
    harness = _harness_module()
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    results = tmp_path / "results/model/all_on"
    results.mkdir(parents=True)
    work = tmp_path / "runs/model/all_on/guidance_write_readback-r0/proj"
    work.mkdir(parents=True)
    (work / "release.txt").write_text("release=ready")
    meta = {"run_id": "guidance_write_readback-r0", "task": "guidance_write_readback",
            "tail": "release=ready", "wall_s": 1, "exit": 0}
    (results / "meta.jsonl").write_text(json.dumps(meta) + "\n")
    harness.report(["model"], ["all_on"])
    row = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("guidance_write_readback"))
    assert "  0%" in row


@pytest.mark.parametrize("case,expected", [
    ("valid", True), ("read_first", False), ("write_fails", False),
    ("read_fails", False), ("wrong_path", False), ("wrong_id", False),
    ("read_overlaps", False), ("wrong_content", False), ("nonzero_exit", False),
    ("missing_end", False),
])
def test_report_requires_completed_ordered_readback(tmp_path, monkeypatch, capsys, case, expected):
    harness = _harness_module()
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    results = tmp_path / "results/model/all_on"
    results.mkdir(parents=True)
    work = tmp_path / "runs/model/all_on/guidance_write_readback-r0"
    (work / "proj").mkdir(parents=True)
    target = work / "proj/release.txt"
    target.write_text("release=ready")

    def event(identity, name, phase, data):
        return {"kind": "scope", "category": "tool", "scope_category": phase,
                "uuid": identity, "name": name, "data": data}

    write_start = event("write", "write_file", "start", {"path": str(target), "content": "release=ready"})
    write_end = event("write", "write_file", "end", {"success": True})
    read_start = event("read", "read_file", "start", {"path": str(target)})
    read_end = event("read", "read_file", "end", {"content": "release=ready"})
    events = [write_start, write_end, read_start, read_end]
    variants = {
        "read_first": [read_start, read_end, write_start, write_end],
        "read_overlaps": [write_start, read_start, write_end, read_end],
        "missing_end": [write_start, write_end, read_start],
    }
    events = variants.get(case, events)
    mutations = {
        "write_fails": (write_end, "data", {"error": "write denied"}),
        "read_fails": (read_end, "metadata", {"otel.status_code": "ERROR"}),
        "wrong_path": (read_start, "data", {"path": str(work / "other/release.txt")}),
        "wrong_id": (read_end, "uuid", "unrelated"),
        "wrong_content": (read_end, "data", {"content": "release=pending"}),
    }
    if case in mutations:
        event_to_change, field, value = mutations[case]
        event_to_change[field] = value
    meta = {"run_id": "guidance_write_readback-r0", "task": "guidance_write_readback",
            "tail": "release=ready", "wall_s": 1, "exit": 1 if case == "nonzero_exit" else 0}
    (results / "meta.jsonl").write_text(json.dumps(meta) + "\n")
    (results / "guidance_write_readback-r0.atof.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events))
    harness.report(["model"], ["all_on"])
    row = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("guidance_write_readback"))
    assert ("100%" in row) is expected


def test_guidance_readback_accepts_real_relay_tool_spans(tmp_path):
    import asyncio
    relay = pytest.importorskip("nemo_relay")
    harness = _harness_module()
    project = tmp_path / "proj"
    project.mkdir()
    target = project / "release.txt"
    events = []

    def observe(event):
        if event.kind == "scope" and event.category == "tool":
            events.append({key: getattr(event, key) for key in (
                "uuid", "kind", "category", "scope_category", "name", "data", "metadata")})

    async def exercise():
        def write(args):
            Path(args["path"]).write_text(args["content"])
            return relay.ToolExecutionResult({"success": True})

        await relay.tools.execute("write_file", {"path": str(target), "content": "release=ready"}, write)
        await relay.tools.execute("read_file", {"path": str(target)},
                                  lambda args: relay.ToolExecutionResult({"content": Path(args["path"]).read_text()}))

    relay.subscribers.register("guidance-regression", observe)
    try:
        asyncio.run(exercise())
        relay.subscribers.flush()
    finally:
        relay.subscribers.deregister("guidance-regression")
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    assert harness.guidance_trace_success("guidance_write_readback", tmp_path, trace)
