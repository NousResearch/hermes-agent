"""tests/plugins/test_needle_swarm.py - Test suite for Needle Swarm plugin."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import pytest

from plugins.needle_swarm.registry import SkillRegistry, SkillMeta
from plugins.needle_swarm.needle_ops import LoadedNeedleInstance, embed_text, cosine_similarity
from plugins.needle_swarm.working_set import WorkingSet
from plugins.needle_swarm.router import Router
from plugins.needle_swarm.swarm import SwarmDispatcher, SwarmResult
from plugins.needle_swarm.escalation_log import EscalationLog
from plugins.needle_swarm.hindsight_store import HindsightStore
from plugins.needle_swarm.growth_dreaming import GrowthDreamer
from plugins.needle_swarm.dream_rsi import MetaExplorationPolicy
from plugins.needle_swarm.trainer import Trainer
from plugins.needle_swarm.orchestrator import Orchestrator
import plugins.needle_swarm as needle_plugin


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="test_needle_swarm_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_registry(tmp_dir):
    reg = SkillRegistry(tmp_dir / "catalog")
    meta = SkillMeta(
        skill_id="s1",
        name="Skill 1",
        description="Format code",
        kind="ACTION",
        size_mb=10.0,
        tools=["tool1"],
        clusters=["coding"],
    )
    reg.register(meta)

    fetched = reg.get("s1")
    assert fetched is not None
    assert fetched.name == "Skill 1"
    assert len(reg.list_skills(cluster="coding")) == 1

    reg.record_usage("s1", confidence=0.8, escalated=False)
    assert reg.get("s1").use_count == 1

    reg.add_pending_example("s1", {"utterance": "fix code"})
    assert len(reg.get("s1").pending_examples) == 1
    examples = reg.clear_pending_examples("s1")
    assert len(examples) == 1
    assert len(reg.get("s1").pending_examples) == 0


def test_needle_ops_embeddings():
    v1 = embed_text("python formatting")
    v2 = embed_text("python formatting")
    v3 = embed_text("unrelated security threat")

    sim_same = cosine_similarity(v1, v2)
    sim_diff = cosine_similarity(v1, v3)

    assert sim_same == pytest.approx(1.0, abs=1e-3)
    assert sim_diff < 0.99


def test_working_set(tmp_dir):
    reg = SkillRegistry(tmp_dir / "catalog")
    reg.register(SkillMeta(skill_id="s1", name="S1", description="d1", kind="ACTION", size_mb=10.0, clusters=["c1"]))
    reg.register(SkillMeta(skill_id="s2", name="S2", description="d2", kind="ACTION", size_mb=15.0, clusters=["c1"]))

    ws = WorkingSet(reg, max_mb=20.0)
    inst1 = ws.load_skill("s1")
    assert inst1 is not None
    assert ws.current_mb() == 10.0

    # Loading s2 (15 MB) forces eviction of s1 (10 MB) to fit in 20 MB budget
    inst2 = ws.load_skill("s2")
    assert inst2 is not None
    assert not ws.is_loaded("s1")
    assert ws.is_loaded("s2")


def test_router(tmp_dir):
    reg = SkillRegistry(tmp_dir / "catalog")
    ws = WorkingSet(reg, max_mb=64.0)
    reg.register(SkillMeta(skill_id="fmt", name="Fmt", description="Format Python code blocks", kind="ACTION"))
    reg.register(SkillMeta(skill_id="sec", name="Sec", description="Check security logs for threat IPs", kind="ACTION"))

    ws.load_skill("fmt")
    ws.load_skill("sec")

    router = Router(reg, ws, threshold=0.5)
    best_id, score, ranked = router.route("format python script")
    assert best_id == "fmt"
    assert score >= 0.5


def test_swarm_dispatcher(tmp_dir):
    reg = SkillRegistry(tmp_dir / "catalog")
    ws = WorkingSet(reg, max_mb=64.0)
    reg.register(SkillMeta(skill_id="a1", name="A1", description="action 1", kind="ACTION", tools=["t1"]))
    reg.register(SkillMeta(skill_id="p1", name="P1", description="playbook 1", kind="PLAYBOOK"))
    reg.register(SkillMeta(skill_id="m1", name="M1", description="memory 1", kind="MEMORY"))

    ws.load_skill("a1")
    ws.load_skill("p1")
    ws.load_skill("m1")

    dispatcher = SwarmDispatcher(ws)
    res = dispatcher.dispatch_swarm("threat event on server")

    assert len(res.action_signals) == 1
    assert len(res.playbook_signals) == 1
    assert len(res.memory_signals) == 1
    ctx_str = res.as_bonsai_context()
    assert "Candidate Tool Dispatch Signals" in ctx_str


def test_escalation_log(tmp_dir):
    log = EscalationLog(tmp_dir / "escalations.json")
    ev = log.log(
        utterance="deploy app",
        routed_skill_id=None,
        confidence=0.2,
        resolved_by="CLOUD_LLM",
        outcome="ESCALATED",
        resolved_action={"status": "ok"},
    )
    assert ev.event_id.startswith("evt_")
    assert len(log.get_events(unresolved_only=True)) == 1


def test_hindsight_store(tmp_dir):
    hs = HindsightStore(tmp_dir / "hindsight")
    f1 = hs.retain("User asked to format python code", source_type="DISPATCH")
    f2 = hs.retain("User asked to format python code again", source_type="DISPATCH")

    new_obs = hs.consolidate(similarity_threshold=0.5)
    assert new_obs > 0

    recalled = hs.recall("format python code")
    assert len(recalled) > 0


test_growth_dreamer = None  # placeholder


def test_growth_dreamer_replay(tmp_dir):
    log = EscalationLog(tmp_dir / "escalations.json")
    log.log("query 1", "s1", 0.8, "NEEDLE", "SUCCESS")
    log.log("query 2", None, 0.2, "CLOUD_LLM", "ESCALATED")

    dreamer = GrowthDreamer(log)
    thresh_res = dreamer.replay_thresholds([0.5, 0.7])
    assert 0.5 in thresh_res
    assert thresh_res[0.5]["auto_resolved"] == 1

    hypo_res = dreamer.replay_hypothetical_skill("query 2 handler")
    assert hypo_res["captured_count"] >= 0


def test_orchestrator_end_to_end(tmp_dir):
    def dummy_llm(prompt: str) -> str:
        return "dummy_skill: Handles dummy requests"

    orch = Orchestrator(data_dir=tmp_dir, brain_llm_callback=dummy_llm, confidence_threshold=0.5)

    orch.registry.register(SkillMeta(
        skill_id="formatter",
        name="Formatter",
        description="Format python code",
        kind="ACTION",
        tools=["format_code"],
        clusters=["coding"],
    ))

    orch.enter_mode("coding")
    res1 = orch.handle_single("Format python code")
    assert res1["status"] in ("AUTO_RESOLVED", "ESCALATED")

    for i in range(5):
        orch.handle_single(f"Unmatched request {i}")

    growth_res = orch.run_growth_cycle()
    assert "consolidated_observations" in growth_res


def test_plugin_registration(tmp_dir, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_dir))

    registered_tools = {}
    registered_commands = {}

    class MockContext:
        def register_tool(self, name, toolset, schema, handler, **kw):
            registered_tools[name] = handler

        def register_command(self, name, handler, description="", args_hint=""):
            registered_commands[name] = handler

    ctx = MockContext()
    needle_plugin.register(ctx)

    assert "needle_single_dispatch" in registered_tools
    assert "needle_swarm_fanout" in registered_tools
    assert "needle_growth_cycle" in registered_tools
    assert "needle_hindsight_recall" in registered_tools
    assert "needle" in registered_commands

    cmd_out = registered_commands["needle"]("status")
    assert "Needle Swarm Status" in cmd_out
