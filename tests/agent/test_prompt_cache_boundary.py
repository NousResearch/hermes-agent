"""Builder-declared stable-prefix cache boundaries (#81867).

Webhook/cron skill invocations concatenate a large static scaffold with a
small volatile tail into one user string; without a declared boundary the
whole message is cached as one atomic block and a changed ticket ID or
timestamp forces a full cache rewrite. These tests cover the registry, the
request-local split, the failover round-trip, and — via the real builders —
that the boundary comes from construction, not from re-parsing marker
strings (so payloads that quote the marker cannot poison the stable prefix).
"""

import copy

import pytest
from unittest.mock import patch

import agent.skill_bundles as skill_bundles
import agent.skill_commands as skill_commands
import tools.skills_tool as skills_tool
from agent.prompt_cache_boundary import (
    clear_stable_prefixes,
    find_stable_prefix,
    register_stable_prefix,
)
from agent.prompt_caching import (
    apply_anthropic_cache_control,
    build_prompt_cache_plan,
    strip_anthropic_cache_control,
)
from agent.skill_commands import _SINGLE_SKILL_INSTRUCTION

MARKER = {"type": "ephemeral"}

SKILL_BODY = "Inspect the report carefully and preserve the stable instructions."


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_stable_prefixes()
    yield
    clear_stable_prefixes()


def _write_skill(skills_dir, name, body=SKILL_BODY):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}\n---\n\n# {name}\n\n{body}\n"
    )
    return skill_dir


@pytest.fixture()
def skills(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "triage")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    monkeypatch.setattr(skill_bundles, "_bundles_cache", {})
    monkeypatch.setattr(skill_bundles, "_bundles_cache_mtime", None)
    skill_commands.scan_skill_commands()
    return skills_dir


class TestRegistry:
    def test_append_user_instruction_prefix_invariant(self):
        """The shared builder helper must return a byte-prefix of the final
        joined message — the invariant every registration site depends on."""
        from agent.skill_commands import append_user_instruction

        parts = ["scaffold line one", "", "skill body", ""]
        instruction = "ticket=42 time=10:00"
        stable_prefix = append_user_instruction(parts, instruction)
        message = "\n".join(parts)

        assert message.startswith(stable_prefix)
        assert message[len(stable_prefix):] == instruction

    def test_requires_proper_prefix(self):
        register_stable_prefix("scaffold")
        assert find_stable_prefix("scaffold volatile") == "scaffold"
        # Exact match would leave an empty volatile block — never split.
        assert find_stable_prefix("scaffold") is None
        assert find_stable_prefix("other") is None

    def test_longest_registered_prefix_wins(self):
        register_stable_prefix("scaffold")
        register_stable_prefix("scaffold extended")
        assert find_stable_prefix("scaffold extended tail") == "scaffold extended"

    def test_lru_eviction_falls_back_safely(self):
        from agent import prompt_cache_boundary

        for index in range(prompt_cache_boundary._MAX_ENTRIES + 1):
            register_stable_prefix(f"scaffold-{index} ")
        assert find_stable_prefix("scaffold-0 volatile") is None
        assert find_stable_prefix("scaffold-1 volatile") == "scaffold-1 "

    def test_empty_prefix_never_registered(self):
        register_stable_prefix("")
        assert find_stable_prefix("anything") is None

    def test_lookup_refreshes_the_entry_lru_position(self):
        """A cron scaffold fired every minute must survive a burst of one-off
        invocations; without the refresh it silently drops back to
        whole-message caching while still being the hottest prefix."""
        from agent import prompt_cache_boundary

        register_stable_prefix("hot-scaffold ")
        for index in range(prompt_cache_boundary._MAX_ENTRIES - 1):
            register_stable_prefix(f"cold-{index} ")

        assert find_stable_prefix("hot-scaffold volatile") == "hot-scaffold "

        register_stable_prefix("newcomer ")

        assert find_stable_prefix("hot-scaffold volatile") == "hot-scaffold "
        assert find_stable_prefix("cold-0 volatile") is None

    def test_total_char_cap_evicts_oldest_and_keeps_newest(self, monkeypatch):
        """Entries retain whole skill bodies, so the entry count alone does
        not bound memory."""
        from agent import prompt_cache_boundary

        monkeypatch.setattr(prompt_cache_boundary, "_MAX_CHARS", 100)

        register_stable_prefix("a" * 80)
        register_stable_prefix("b" * 80)

        assert find_stable_prefix("a" * 80 + "tail") is None
        assert find_stable_prefix("b" * 80 + "tail") == "b" * 80

    def test_single_oversized_prefix_still_registers(self, monkeypatch):
        from agent import prompt_cache_boundary

        monkeypatch.setattr(prompt_cache_boundary, "_MAX_CHARS", 10)
        oversized = "x" * 500

        register_stable_prefix(oversized)

        assert find_stable_prefix(oversized + "tail") == oversized


class TestRequestLocalSplit:
    def test_volatile_tail_does_not_change_marked_prefix(self):
        scaffold = "stable skill scaffold\n\n" + _SINGLE_SKILL_INSTRUCTION
        register_stable_prefix(scaffold)

        first = apply_anthropic_cache_control(
            [{"role": "user", "content": scaffold + "ticket=one time=10:00"}]
        )[0]["content"]
        second = apply_anthropic_cache_control(
            [{"role": "user", "content": scaffold + "ticket=two time=10:01"}]
        )[0]["content"]

        assert len(first) == len(second) == 2
        assert first[0] == second[0]
        assert first[0] == {"type": "text", "text": scaffold, "cache_control": MARKER}
        assert "cache_control" not in first[1]
        assert first[1]["text"] == "ticket=one time=10:00"
        assert second[1]["text"] == "ticket=two time=10:01"

    def test_unregistered_message_keeps_whole_block_layout(self):
        content = "stable-looking scaffold\n\n" + _SINGLE_SKILL_INSTRUCTION + "tail"
        marked = apply_anthropic_cache_control([{"role": "user", "content": content}])
        assert marked[0]["content"] == [
            {"type": "text", "text": content, "cache_control": MARKER}
        ]

    def test_assistant_string_matching_a_prefix_is_not_split(self):
        register_stable_prefix("scaffold ")
        marked = apply_anthropic_cache_control(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "scaffold reply"},
            ]
        )
        assert marked[1]["content"] == [
            {"type": "text", "text": "scaffold reply", "cache_control": MARKER}
        ]

    def test_canonical_message_stays_a_string_in_plan(self):
        scaffold = "stable scaffold "
        register_stable_prefix(scaffold)
        original = scaffold + "volatile"
        messages = [{"role": "user", "content": original}]

        plan = build_prompt_cache_plan(messages, [], native_anthropic=True)

        assert messages == [{"role": "user", "content": original}]
        assert isinstance(plan.messages[0]["content"], list)
        assert plan.messages[0]["content"][0]["cache_control"] == MARKER
        assert "cache_control" not in plan.messages[0]["content"][1]

    def test_strip_reconstructs_exact_string_and_redecorates_identically(self):
        scaffold = "stable scaffold\n\n" + _SINGLE_SKILL_INSTRUCTION
        register_stable_prefix(scaffold)
        original = [{"role": "user", "content": scaffold + "ticket=one"}]

        marked = apply_anthropic_cache_control(copy.deepcopy(original))
        first_wire = copy.deepcopy(marked)
        stripped = strip_anthropic_cache_control(marked)

        assert stripped == original
        assert apply_anthropic_cache_control(copy.deepcopy(stripped)) == first_wire

    def test_strip_flattens_even_after_the_prefix_was_evicted(self):
        """Mid-turn failover re-decorates a request built many messages ago
        (#72626). If flattening depended on the registry still holding the
        entry, a busy gateway that registered _MAX_ENTRIES newer scaffolds in
        between would ship the split shape to the next provider instead of
        the canonical string."""
        from agent import prompt_cache_boundary

        scaffold = "stable scaffold\n\n" + _SINGLE_SKILL_INSTRUCTION
        register_stable_prefix(scaffold)
        original = [{"role": "user", "content": scaffold + "ticket=one"}]
        marked = apply_anthropic_cache_control(copy.deepcopy(original))

        for index in range(prompt_cache_boundary._MAX_ENTRIES + 1):
            register_stable_prefix(f"unrelated-scaffold-{index} ")
        assert find_stable_prefix(scaffold + "ticket=one") is None

        assert strip_anthropic_cache_control(marked) == original

    def test_strip_leaves_organic_two_part_user_content_structured(self):
        organic = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        stripped = strip_anthropic_cache_control(copy.deepcopy(organic))
        assert stripped == organic


class TestRealBuilders:
    def test_two_invocations_share_the_marked_scaffold(self, skills):
        first = skill_commands.build_skill_invocation_message(
            "/triage", user_instruction="ticket=one time=10:00"
        )
        second = skill_commands.build_skill_invocation_message(
            "/triage", user_instruction="ticket=two time=10:01"
        )

        first_blocks = apply_anthropic_cache_control(
            [{"role": "user", "content": first}], native_anthropic=True
        )[0]["content"]
        second_blocks = apply_anthropic_cache_control(
            [{"role": "user", "content": second}], native_anthropic=True
        )[0]["content"]

        assert len(first_blocks) == len(second_blocks) == 2
        assert first_blocks[0] == second_blocks[0]
        assert first_blocks[0]["cache_control"] == MARKER
        assert SKILL_BODY in first_blocks[0]["text"]
        assert "ticket=one" not in first_blocks[0]["text"]
        assert "cache_control" not in first_blocks[1]
        assert first_blocks[1]["text"] == "ticket=one time=10:00"

    def test_payload_quoting_the_marker_cannot_poison_the_stable_prefix(self, skills):
        """The differentiator vs marker-search heuristics: a ticket that quotes
        the instruction marker (e.g. a pasted agent transcript) must stay
        entirely in the volatile tail, or two invocations stop sharing the
        cached prefix."""
        hostile = (
            "ticket=one\n\n" + _SINGLE_SKILL_INSTRUCTION + "quoted transcript line"
        )
        benign = "ticket=two"

        hostile_blocks = apply_anthropic_cache_control(
            [
                {
                    "role": "user",
                    "content": skill_commands.build_skill_invocation_message(
                        "/triage", user_instruction=hostile
                    ),
                }
            ]
        )[0]["content"]
        benign_blocks = apply_anthropic_cache_control(
            [
                {
                    "role": "user",
                    "content": skill_commands.build_skill_invocation_message(
                        "/triage", user_instruction=benign
                    ),
                }
            ]
        )[0]["content"]

        assert hostile_blocks[0] == benign_blocks[0]
        assert hostile_blocks[1]["text"] == hostile
        assert benign_blocks[1]["text"] == benign

    def test_bare_invocation_keeps_whole_block_layout(self, skills):
        message = skill_commands.build_skill_invocation_message("/triage")
        blocks = apply_anthropic_cache_control(
            [{"role": "user", "content": message}]
        )[0]["content"]
        assert blocks == [{"type": "text", "text": message, "cache_control": MARKER}]

    def test_builder_round_trip_survives_failover_strip(self, skills):
        original = [
            {
                "role": "user",
                "content": skill_commands.build_skill_invocation_message(
                    "/triage", user_instruction="ticket=one"
                ),
            }
        ]
        marked = apply_anthropic_cache_control(copy.deepcopy(original))
        assert strip_anthropic_cache_control(marked) == original


class TestCronBuilder:
    def _prompts(self, jobs):
        from cron.scheduler import _build_job_prompt

        with patch(
            "agent.skill_bundles.resolve_bundle_command_key", return_value=None
        ), patch(
            "tools.skills_tool.skill_view", side_effect=self._skill_view
        ), patch(
            "tools.skill_usage.bump_use"
        ):
            return [_build_job_prompt(job) for job in jobs]

    @staticmethod
    def _skill_view(name: str) -> str:
        import json

        if name == "missing":
            return json.dumps({"success": False, "error": "missing"})
        return json.dumps({"success": True, "content": f"Stable content for {name}."})

    def test_cron_runs_share_the_marked_scaffold(self):
        common = {"id": "job-cache", "name": "cache boundary", "skills": ["alpha", "beta"]}
        first, second = self._prompts(
            [
                {**common, "prompt": "ticket=one time=10:00"},
                {**common, "prompt": "ticket=two time=10:01"},
            ]
        )

        first_blocks = apply_anthropic_cache_control(
            [{"role": "user", "content": first}]
        )[0]["content"]
        second_blocks = apply_anthropic_cache_control(
            [{"role": "user", "content": second}]
        )[0]["content"]

        assert isinstance(first, str) and isinstance(second, str)
        assert len(first_blocks) == len(second_blocks) == 2
        assert first_blocks[0] == second_blocks[0]
        assert first_blocks[0]["cache_control"] == MARKER
        assert "Stable content for alpha." in first_blocks[0]["text"]
        assert "ticket=one" not in first_blocks[0]["text"]
        assert first_blocks[1]["text"].endswith("ticket=one time=10:00")

    def test_missing_skill_notice_stays_in_the_stable_prefix(self):
        common = {"id": "job-skip", "name": "skip notice", "skills": ["missing", "alpha"]}
        first, second = self._prompts(
            [{**common, "prompt": "ticket=one"}, {**common, "prompt": "ticket=two"}]
        )

        first_blocks = apply_anthropic_cache_control(
            [{"role": "user", "content": first}]
        )[0]["content"]
        second_blocks = apply_anthropic_cache_control(
            [{"role": "user", "content": second}]
        )[0]["content"]

        assert first_blocks[0] == second_blocks[0]
        assert "could not be found" in first_blocks[0]["text"]
        assert first_blocks[1]["text"].endswith("ticket=one")
