"""Extended Brain Networks tests — persistence, focus, tone, runtime wiring."""

from __future__ import annotations

import threading

import pytest


@pytest.fixture
def brain_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.brain_networks.runtime import reset_orchestrator_for_tests

    reset_orchestrator_for_tests()
    yield hermes_home
    reset_orchestrator_for_tests()


class TestPersistence:
    def test_ecn_roundtrip(self, brain_env):
        from agent.brain_networks.persistence import load_ecn_state, save_ecn_state

        save_ecn_state(
            "sess-1",
            current_task="ship focus feature",
            task_stack=["prior"],
            focus_level=0.9,
            distraction_count=1,
            state="focused",
            pinned=True,
        )
        data = load_ecn_state("sess-1")
        assert data is not None
        assert data["current_task"] == "ship focus feature"
        assert data["pinned"] is True
        assert data["task_stack"] == ["prior"]

    def test_ecn_clear(self, brain_env):
        from agent.brain_networks.persistence import (
            clear_ecn_state,
            load_ecn_state,
            save_ecn_state,
        )

        save_ecn_state(
            "sess-2",
            current_task="x",
            task_stack=[],
            focus_level=1.0,
            distraction_count=0,
            state="focused",
            pinned=True,
        )
        assert clear_ecn_state("sess-2") is True
        assert load_ecn_state("sess-2") is None

    def test_dream_persist_and_load(self, brain_env):
        from agent.brain_networks.persistence import dream_count, load_recent_dreams, save_dream_episode

        rid = save_dream_episode(
            {
                "narrative": "Consolidated recent work on focus",
                "insights": ["persist across restarts"],
                "emotional_tone": "reflective",
                "source_count": 3,
                "source_episodes": [1, 2, 3],
            }
        )
        assert rid > 0
        assert dream_count() >= 1
        recent = load_recent_dreams(limit=5)
        assert recent
        assert "focus" in recent[0]["narrative"].lower() or recent[0]["emotional_tone"]

    def test_concurrent_ecn_writes(self, brain_env):
        from agent.brain_networks.persistence import load_ecn_state, save_ecn_state

        errors = []

        def writer(i: int) -> None:
            try:
                save_ecn_state(
                    "race-sess",
                    current_task=f"task-{i}",
                    task_stack=[f"old-{i}"],
                    focus_level=0.5 + (i % 5) * 0.1,
                    distraction_count=i % 3,
                    state="focused",
                    pinned=i % 2 == 0,
                )
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        data = load_ecn_state("race-sess")
        assert data is not None
        assert data["current_task"].startswith("task-")


class TestECNFocusPersistence:
    def test_set_focus_persists_across_instances(self, brain_env):
        from agent.brain_networks.ecn import ExecutiveControlNetwork

        a = ExecutiveControlNetwork()
        a.initialize({})
        a.bind_session("focus-sess")
        a.set_focus("implement /focus command", pinned=True)

        b = ExecutiveControlNetwork()
        b.initialize({})
        b.bind_session("focus-sess")
        assert b.current_task == "implement /focus command"
        assert b.pinned is True

    def test_pinned_resists_auto_switch(self, brain_env):
        from agent.brain_networks.ecn import ExecutiveControlNetwork

        ecn = ExecutiveControlNetwork()
        ecn.initialize({})
        ecn.bind_session("pin-sess")
        ecn.set_focus("keep this", pinned=True)
        result = ecn.evaluate_focus(
            {"user_message": "hello world there", "session_id": "pin-sess"}
        )
        assert result["current_task"] == "keep this"
        assert result["reminder"]
        assert "keep this" in result["reminder"]

    def test_clear_focus(self, brain_env):
        from agent.brain_networks.ecn import ExecutiveControlNetwork

        ecn = ExecutiveControlNetwork()
        ecn.initialize({})
        ecn.bind_session("clr")
        ecn.set_focus("temp")
        ecn.clear_focus()
        assert ecn.current_task is None
        assert ecn.state == "idle"


class TestDreamToneDeterministic:
    def test_emotional_tone_not_random(self, brain_env):
        from agent.brain_networks.dreaming import DreamEngine

        engine = DreamEngine()
        engine.initialize({"persist": False})
        episodes = [
            {"id": 1, "content": "critical urgent security bug", "importance": 0.9},
            {"id": 2, "content": "emergency deadline asap", "importance": 0.95},
        ]
        tones = {engine._determine_emotional_tone(episodes) for _ in range(8)}
        # Deterministic — single tone across repeats
        assert len(tones) == 1
        tone = tones.pop()
        assert tone in ("urgent", "concerned", "frustrated", "angry", "reflective")

    def test_generate_dream_persists(self, brain_env, monkeypatch):
        from agent.brain_networks.dreaming import DreamEngine
        from agent.brain_networks.persistence import dream_count

        engine = DreamEngine()
        engine.initialize({"enabled": True, "persist": True})

        def fake_collect(_self):
            return [
                {"id": 1, "source": "experience", "content": "great progress thanks", "importance": 0.7},
                {"id": 2, "source": "episodic", "content": "excellent work awesome", "importance": 0.8},
                {"id": 3, "source": "experience", "content": "love this feature", "importance": 0.6},
            ]

        monkeypatch.setattr(DreamEngine, "_collect_episodes", fake_collect)
        monkeypatch.setattr(
            "agent.brain_networks.llm_helper.generate_with_llm",
            lambda *a, **k: None,
        )

        dream = engine.generate_dream()
        assert dream is not None
        assert dream["type"] == "dream"
        assert dream_count() >= 1


class TestOrchestratorRuntime:
    def test_normalize_flat_config(self, brain_env):
        from agent.brain_networks.runtime import normalize_brain_config

        cfg = normalize_brain_config(
            {
                "enabled": True,
                "dmn_reflection_chance": 0.1,
                "ecn_max_task_stack": 7,
                "dream_idle_threshold_seconds": 42,
            }
        )
        assert cfg["dmn"]["reflection_chance"] == 0.1
        assert cfg["ecn"]["max_task_stack"] == 7
        assert cfg["dreaming"]["idle_threshold_seconds"] == 42

    def test_get_orchestrator_respects_disabled(self, brain_env, monkeypatch):
        from agent.brain_networks.runtime import get_orchestrator, reset_orchestrator_for_tests

        reset_orchestrator_for_tests()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"brain_networks": {"enabled": False}},
        )
        assert get_orchestrator() is None

    def test_get_orchestrator_enabled(self, brain_env, monkeypatch):
        from agent.brain_networks.runtime import get_orchestrator, reset_orchestrator_for_tests

        reset_orchestrator_for_tests()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "brain_networks": {
                    "enabled": True,
                    "ecn_max_task_stack": 5,
                    "dream_idle_threshold_seconds": 10,
                }
            },
        )
        orch = get_orchestrator()
        assert orch is not None
        orch.bind_session("rt-1")
        orch.set_focus("runtime focus")
        block = orch.format_turn_block(
            orch.on_turn_start({"user_message": "continue the work", "session_id": "rt-1"})
        )
        assert "brain-networks" in block or "ECN focus" in block

    def test_build_brain_turn_context(self, brain_env, monkeypatch):
        from agent.brain_networks.runtime import (
            build_brain_turn_context,
            reset_orchestrator_for_tests,
        )

        reset_orchestrator_for_tests()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"brain_networks": {"enabled": True}},
        )
        block = build_brain_turn_context(
            "I'm frustrated with this critical bug",
            session_id="ctx-1",
        )
        assert block
        assert "brain-networks" in block


class TestDoctorPersistence:
    def test_doctor_includes_persistence(self, brain_env):
        from agent.brain_networks import doctor_check

        result = doctor_check()
        assert result["ok"] is True
        assert "persistence" in result
        assert result["persistence"]["ok"] is True
