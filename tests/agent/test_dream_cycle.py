import json
import pytest
from pathlib import Path
from agent.dream_cycle import (
    DreamCycleEngine,
    DreamEpisode,
    ReflexRule,
)


@pytest.fixture
def dream_engine(tmp_path):
    return DreamCycleEngine(hermes_home=tmp_path)


def test_harvest_episodes_from_tool_failure(dream_engine):
    messages = [
        {"role": "user", "content": "Update the git branch"},
        {"role": "assistant", "content": "Calling git checkout..."},
        {"role": "tool", "name": "terminal", "content": "fatal: Needed a single revision, exit code 128"},
    ]
    episodes = dream_engine.harvest_episodes_from_messages(messages, session_id="test_sess_1")
    assert len(episodes) == 1
    assert episodes[0].session_id == "test_sess_1"
    assert episodes[0].turn_index == 2
    assert "fatal" in episodes[0].error_signal.lower()


def test_harvest_episodes_from_user_corrections(dream_engine):
    messages = [
        {"role": "user", "content": "Start docker container"},
        {"role": "assistant", "content": "Running docker run..."},
        {"role": "user", "content": "No, that's wrong! You used the wrong port mapping."},
    ]
    episodes = dream_engine.harvest_episodes_from_messages(messages, session_id="test_sess_2")
    assert len(episodes) == 1
    assert episodes[0].session_id == "test_sess_2"
    assert episodes[0].error_signal == "user_correction"
    assert "wrong port mapping" in episodes[0].user_correction


def test_distill_rule_from_episode(dream_engine):
    ep = DreamEpisode(
        session_id="test_sess",
        turn_index=1,
        error_signal="fatal: git branch origin/feature does not exist",
    )
    rule = dream_engine.distill_rule_from_episode(ep)
    assert rule is not None
    assert rule.category == "git"
    assert "remote-tracking" in rule.heuristic.lower()
    assert rule.severity == "mandatory"
    assert rule.confidence > 0.9


def test_deduplicate_and_merge_rules(dream_engine):
    r1 = ReflexRule(
        rule_id="r1",
        category="git",
        trigger="When checking branches",
        heuristic="Always verify remote tracking ref before deleting",
        confidence=0.9,
    )
    r2 = ReflexRule(
        rule_id="r2",
        category="git",
        trigger="When managing branches",
        heuristic="Always verify remote tracking ref before deleting",
        confidence=0.9,
    )
    merged = dream_engine.deduplicate_and_merge([r1], [r2])
    assert len(merged) == 1
    assert merged[0].confidence == 0.95
    assert merged[0].times_applied == 1


def test_prune_stale_rules(dream_engine):
    rules = [
        ReflexRule(rule_id=f"r_{i}", category="test", trigger="t", heuristic="h", confidence=0.4 if i == 0 else 0.9)
        for i in range(5)
    ]
    retained, pruned = dream_engine.prune_stale_rules(rules, max_rules=3, min_confidence=0.5)
    assert len(retained) == 3
    assert pruned == 2
    assert all(r.confidence >= 0.5 for r in retained)


def test_save_and_load_reflex_rules(dream_engine):
    rule = ReflexRule(
        rule_id="test_rule_1",
        category="docker",
        trigger="When launching container",
        heuristic="Ensure ports are open",
        confidence=0.95,
    )
    dream_engine.save_reflex_rules([rule])

    loaded = dream_engine.load_reflex_rules()
    assert len(loaded) == 1
    assert loaded[0].rule_id == "test_rule_1"
    assert loaded[0].heuristic == "Ensure ports are open"

    # Verify Markdown journal was generated
    md_content = dream_engine.reflex_md_path.read_text(encoding="utf-8")
    assert "# Cognitive Reflexes" in md_content
    assert "test_rule_1" in md_content


def test_execute_dream_cycle_end_to_end(dream_engine):
    trajectories = [
        [
            {"role": "user", "content": "Deploy the docker app"},
            {"role": "tool", "content": "Error: Docker container failed to mount path"},
        ],
        [
            {"role": "user", "content": "Update branch"},
            {"role": "user", "content": "You made a mistake on that branch name"},
        ]
    ]
    summary = dream_engine.execute_dream_cycle(recent_trajectories=trajectories)
    assert summary.episodes_harvested == 2
    assert summary.rules_synthesized >= 1
    assert summary.total_active_rules >= 1
    assert "Dream Cycle complete" in summary.journal_entry

    rules = dream_engine.load_reflex_rules()
    assert len(rules) == summary.total_active_rules


def test_inject_reflexes_into_prompt(dream_engine):
    rule = ReflexRule(
        rule_id="r_git_1",
        category="git",
        trigger="When switching branches",
        heuristic="Check unmerged commits first",
        confidence=0.98,
    )
    dream_engine.save_reflex_rules([rule])

    base_prompt = "You are Hermes, an autonomous agent."
    augmented = dream_engine.inject_reflexes_into_prompt(base_prompt)
    assert "<cognitive_reflexes>" in augmented
    assert "[GIT]" in augmented
    assert "Check unmerged commits first" in augmented
