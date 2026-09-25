"""`agent.pre_verify_on_no_edit_turns` — the general no-edit `pre_verify` gate.

Drives the REAL `AIAgent.run_conversation` loop with a REAL plugin loaded
through the real discovery path, so the assertions are about runtime behaviour,
not about the shape of the source.

Contract under test:

* default (flag absent/false) — a turn that edits no files never calls
  `pre_verify`; shipped behaviour is byte-for-byte unchanged;
* enabled — the same call site fires once with `changed_paths=[]`, the
  directive continues the turn, and the existing `agent.max_verify_nudges`
  bound still caps it;
* edited-code turns keep firing either way.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

PLUGIN_SRC = '''
import json, os
CALLS = os.environ.get("PROBE_LOG", "/tmp/pre_verify_probe.jsonl")

def on_pre_verify(session_id="", platform="", model="", coding=False, attempt=0,
                  final_response="", changed_paths=None, **kw):
    with open(CALLS, "a") as fh:
        fh.write(json.dumps({"attempt": attempt, "changed_paths": list(changed_paths or []),
                             "final_response": final_response}) + "\\n")
    if attempt == 0:
        return {"action": "continue", "message": "PROBE: keep going and finish the job."}
    return None

def register(ctx):
    ctx.register_hook("pre_verify", on_pre_verify)
'''


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with one real pre_verify plugin installed."""
    home = tmp_path / "hermes-home"
    (home / "plugins" / "probe-verify").mkdir(parents=True)
    (home / "plugins" / "probe-verify" / "plugin.yaml").write_text(
        "name: probe-verify\nversion: 0.0.1\ndescription: pre_verify probe\n"
    )
    log = tmp_path / "calls.jsonl"
    (home / "plugins" / "probe-verify" / "__init__.py").write_text(
        PLUGIN_SRC.replace("/tmp/pre_verify_probe.jsonl", str(log))
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    def configure(**agent_cfg):
        cfg = {"plugins": {"enabled": ["probe-verify"]}}
        if agent_cfg:
            cfg["agent"] = agent_cfg
        (home / "config.yaml").write_text(yaml.safe_dump(cfg))
        from hermes_cli import plugins as P

        P.discover_plugins(force=True)

    def calls():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    return SimpleNamespace(home=home, configure=configure, calls=calls)


def _make_agent(session_id="pre-verify-no-edit"):
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            session_id=session_id, api_key="k", base_url="https://example.invalid/v1",
            provider="openai-compat", model="test/model", max_iterations=6,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent._cached_system_prompt = "stable test prompt"
    agent._session_db = None
    agent._session_json_enabled = False
    agent.save_trajectories = False
    agent.compression_enabled = False
    agent._cleanup_task_resources = lambda *_a, **_kw: None
    agent._save_trajectory = lambda *_a, **_kw: None
    return agent


def _scripted(agent, *, edits=None, answers=("first answer", "second answer")):
    """A model that answers text-only; optionally reports file mutations."""
    seen = {"n": 0}

    def model_call(_api_kwargs):
        idx = min(seen["n"], len(answers) - 1)
        seen["n"] += 1
        if edits:
            agent._turn_file_mutation_paths = set(edits)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=answers[idx], tool_calls=None, reasoning=None),
                finish_reason="stop")],
            model="test/model", usage=None,
        )

    agent._interruptible_api_call = model_call
    return seen


def test_no_edit_turn_does_not_fire_pre_verify_by_default(probe):
    probe.configure()  # no agent section at all -> shipped default
    agent = _make_agent()
    seen = _scripted(agent)

    result = agent.run_conversation("just answer me")

    assert probe.calls() == [], "default must not call pre_verify on a no-edit turn"
    assert result["final_response"] == "first answer"
    assert seen["n"] == 1


def test_explicit_false_is_also_off(probe):
    probe.configure(pre_verify_on_no_edit_turns=False)
    agent = _make_agent()
    _scripted(agent)

    agent.run_conversation("just answer me")

    assert probe.calls() == []


def test_enabled_fires_once_with_empty_changed_paths_and_continues(probe):
    probe.configure(pre_verify_on_no_edit_turns=True, max_verify_nudges=3)
    agent = _make_agent()
    seen = _scripted(agent)

    result = agent.run_conversation("just answer me")

    calls = probe.calls()
    assert calls, "enabled flag must reach the pre_verify call site"
    assert calls[0]["changed_paths"] == []
    assert calls[0]["final_response"] == "first answer"
    # The directive really continued the turn: a second model call happened and
    # the delivered answer is the post-continuation one.
    assert seen["n"] >= 2
    assert result["final_response"] == "second answer"
    # Role alternation is intact (no two user rows in a row).
    roles = [m["role"] for m in result["messages"]]
    assert all(not (a == b == "user") for a, b in zip(roles, roles[1:])), roles


def test_bound_still_applies_when_enabled(probe):
    """max_verify_nudges=0 keeps the gate closed even with the flag on."""
    probe.configure(pre_verify_on_no_edit_turns=True, max_verify_nudges=0)
    agent = _make_agent()
    seen = _scripted(agent)

    agent.run_conversation("just answer me")

    assert probe.calls() == []
    assert seen["n"] == 1


def test_edited_turns_still_fire_with_the_flag_off(probe):
    probe.configure(max_verify_nudges=3)
    agent = _make_agent()
    _scripted(agent, edits=["changed.py"])

    agent.run_conversation("edit changed.py")

    calls = probe.calls()
    assert calls and calls[0]["changed_paths"] == ["changed.py"]
