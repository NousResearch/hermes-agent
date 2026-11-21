"""Per-turn skill candidate ranking — cache-stable overlay on the user turn."""

from types import SimpleNamespace

from agent.skill_selection import (
    apply_turn_skill_selection,
    format_candidate_overlay,
    log_unused_skill_shortlist,
    rank_skill_candidates,
    resolve_selection_mode,
)


def _entry(name, description, triggers=None):
    return {"name": name, "description": description, "triggers": list(triggers or [])}


DEBUG = _entry(
    "systematic-debugging",
    "4-phase root cause debugging: understand bugs before fixing.",
    triggers=["root cause", "reproduce the bug"],
)
TDD = _entry(
    "test-driven-development",
    "TDD: enforce RED-GREEN-REFACTOR, tests before code.",
    triggers=["failing test first"],
)
COOK = _entry(
    "weeknight-cooking",
    "Plan grocery lists and weeknight dinners.",
)


class TestResolveSelectionMode:
    def test_missing_and_unknown_keep_catalog(self):
        assert resolve_selection_mode({}) == "catalog"
        assert resolve_selection_mode({"selection": "routed"}) == "catalog"
        assert resolve_selection_mode({"selection": ""}) == "catalog"

    def test_explicit_modes(self):
        assert resolve_selection_mode({"selection": "shortlist"}) == "shortlist"
        assert resolve_selection_mode({"selection": "catalog"}) == "catalog"
        assert resolve_selection_mode({"selection": "off"}) == "off"
        assert resolve_selection_mode({"selection": "SHORTLIST"}) == "shortlist"


class TestRankSkillCandidates:
    def test_task_pressure_ranks_relevant_skill_above_unrelated(self):
        ranked = rank_skill_candidates(
            "fix this bug, run the gate",
            [COOK, TDD, DEBUG],
            limit=8,
        )
        names = [row["name"] for row in ranked]
        assert names[0] == "systematic-debugging"
        assert "weeknight-cooking" not in names

    def test_trigger_phrase_is_a_deterministic_floor(self):
        ranked = rank_skill_candidates(
            "Please write a failing test first then implement.",
            [COOK, DEBUG, TDD],
            limit=8,
        )
        assert ranked[0]["name"] == "test-driven-development"
        assert ranked[0]["via_trigger"] is True

    def test_unrelated_query_returns_no_candidates(self):
        assert rank_skill_candidates("hello there", [COOK, DEBUG, TDD], limit=8) == []

    def test_limit_caps_the_shortlist(self):
        extras = [_entry(f"debug-tool-{i}", "debugging bugs and root cause analysis") for i in range(6)]
        ranked = rank_skill_candidates("debug this bug root cause", extras + [DEBUG], limit=3)
        assert len(ranked) == 3

    def test_disabled_or_blank_names_are_ignored(self):
        ranked = rank_skill_candidates(
            "debug this bug",
            [_entry("", "debugging bugs"), DEBUG],
            limit=8,
        )
        assert [row["name"] for row in ranked] == ["systematic-debugging"]


class TestOverlayAndPersist:
    def test_shortlist_annotates_api_copy_and_keeps_persist_clean(self, monkeypatch):
        catalog = [DEBUG, TDD, COOK]
        monkeypatch.setattr(
            "agent.skill_selection.collect_skill_index_entries",
            lambda: catalog,
        )
        monkeypatch.setattr(
            "agent.skill_selection._skills_section",
            lambda: {"selection": "shortlist", "selection_limit": 8},
        )
        api, persist, candidates = apply_turn_skill_selection(
            "fix this bug, run the gate",
            None,
        )
        assert persist == "fix this bug, run the gate"
        assert "fix this bug, run the gate" in api
        assert "<skill_candidates>" in api
        assert "systematic-debugging" in api
        assert "weeknight-cooking" not in api
        assert api != persist
        assert candidates[0]["name"] == "systematic-debugging"

    def test_catalog_and_off_leave_the_user_message_untouched(self, monkeypatch):
        monkeypatch.setattr(
            "agent.skill_selection.collect_skill_index_entries",
            lambda: [DEBUG, TDD],
        )
        for mode in ("catalog", "off"):
            monkeypatch.setattr(
                "agent.skill_selection._skills_section",
                lambda mode=mode: {"selection": mode},
            )
            api, persist, candidates = apply_turn_skill_selection(
                "fix this bug, run the gate",
                None,
            )
            assert api == "fix this bug, run the gate"
            assert persist is None
            assert candidates == []

    def test_existing_persist_override_is_preserved(self, monkeypatch):
        monkeypatch.setattr(
            "agent.skill_selection.collect_skill_index_entries",
            lambda: [DEBUG],
        )
        monkeypatch.setattr(
            "agent.skill_selection._skills_section",
            lambda: {"selection": "shortlist"},
        )
        api, persist, _ = apply_turn_skill_selection(
            "Voice: fix this bug",
            "fix this bug",
        )
        assert persist == "fix this bug"
        assert "<skill_candidates>" in api
        assert persist != api

    def test_multimodal_appends_a_text_block(self, monkeypatch):
        monkeypatch.setattr(
            "agent.skill_selection.collect_skill_index_entries",
            lambda: [DEBUG],
        )
        monkeypatch.setattr(
            "agent.skill_selection._skills_section",
            lambda: {"selection": "shortlist"},
        )
        original = [{"type": "text", "text": "fix this bug"}]
        api, persist, _ = apply_turn_skill_selection(original, None)
        assert persist is original
        assert api is not original
        assert api[0] == original[0]
        assert api[-1]["type"] == "text"
        assert "<skill_candidates>" in api[-1]["text"]

    def test_overlay_is_idempotent(self):
        overlay = format_candidate_overlay(
            [{"name": "systematic-debugging", "description": "debug", "via_trigger": False}]
        )
        once = "fix this bug\n\n" + overlay
        api, persist, candidates = apply_turn_skill_selection(once, None)
        assert api == once
        assert persist is None
        assert candidates == []


class TestUnusedShortlistAudit:
    def test_logs_when_no_skill_view_ran(self, caplog):
        import logging

        agent = SimpleNamespace(
            _skill_selection_candidates=[{"name": "systematic-debugging", "description": "d"}],
        )
        messages = [
            {"role": "user", "content": "fix this"},
            {"role": "assistant", "content": "working"},
        ]
        with caplog.at_level(logging.INFO, logger="agent.skill_selection"):
            log_unused_skill_shortlist(agent, messages)
        assert "systematic-debugging" in caplog.text
        assert "skill_view" in caplog.text.lower() or "unused" in caplog.text.lower()

    def test_silent_when_skill_view_ran(self, caplog):
        import logging

        agent = SimpleNamespace(
            _skill_selection_candidates=[{"name": "systematic-debugging", "description": "d"}],
        )
        messages = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "skill_view", "arguments": "{}"}}]},
            {"role": "tool", "content": "ok"},
        ]
        with caplog.at_level(logging.INFO, logger="agent.skill_selection"):
            log_unused_skill_shortlist(agent, messages)
        assert caplog.text == ""


class TestConfigAndRealCatalog:
    def test_default_config_registers_selection_keys(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        skills = DEFAULT_CONFIG["skills"]
        assert skills["selection"] == "catalog"
        assert skills["selection_limit"] == 12

    def test_real_skills_dir_shortlist_on_temp_home(self, tmp_path, monkeypatch):
        from agent.skill_selection import apply_turn_skill_selection, collect_skill_index_entries
        from agent import skill_utils

        home = tmp_path / ".hermes"
        skills = home / "skills" / "software-development"
        skills.mkdir(parents=True)
        (skills / "systematic-debugging").mkdir()
        (skills / "systematic-debugging" / "SKILL.md").write_text(
            "---\n"
            "name: systematic-debugging\n"
            "description: 4-phase root cause debugging: understand bugs before fixing.\n"
            "metadata:\n"
            "  hermes:\n"
            "    triggers:\n"
            "      - root cause\n"
            "---\n\nBody.\n",
            encoding="utf-8",
        )
        (skills / "weeknight-cooking").mkdir()
        (skills / "weeknight-cooking" / "SKILL.md").write_text(
            "---\nname: weeknight-cooking\ndescription: Plan grocery lists and weeknight dinners.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        (home / "config.yaml").write_text(
            "skills:\n  selection: shortlist\n  selection_limit: 8\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        getattr(skill_utils, "_raw_config_cache_clear", lambda: None)()
        getattr(skill_utils, "_external_dirs_cache_clear", lambda: None)()

        names = {row["name"] for row in collect_skill_index_entries()}
        assert "systematic-debugging" in names
        assert "weeknight-cooking" in names

        api, persist, candidates = apply_turn_skill_selection(
            "fix this bug, run the gate",
            None,
        )
        assert persist == "fix this bug, run the gate"
        assert "<skill_candidates>" in api
        assert candidates[0]["name"] == "systematic-debugging"
        assert "weeknight-cooking" not in api
