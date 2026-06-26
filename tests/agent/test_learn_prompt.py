"""Tests for /learn — open-ended skill distillation.

Covers the shared prompt builder (agent.learn_prompt.build_learn_prompt) and
the slash-command registry wiring. /learn has no engine and no model tool: it
builds a standards-guided prompt that the live agent runs as a normal turn, so
these are the load-bearing behavior contracts.
"""

from agent.learn_prompt import build_learn_prompt, _AUTHORING_STANDARDS


class TestBuildLearnPrompt:
    def test_embeds_the_user_request_verbatim(self):
        req = "the REST client in ~/projects/acme-sdk, focus on auth"
        prompt = build_learn_prompt(req)
        assert req in prompt

    def test_always_includes_the_authoring_standards(self):
        # The standards are what make distilled skills match house style;
        # they must travel with every prompt regardless of input.
        for req in ["", "a url https://x/y", "what we just did"]:
            assert _AUTHORING_STANDARDS in build_learn_prompt(req)

    def test_instructs_saving_via_skill_manage_not_a_raw_file(self):
        prompt = build_learn_prompt("learn the thing")
        assert "skill_manage" in prompt

    def test_references_gather_tools_for_open_ended_sourcing(self):
        # Open-ended sourcing relies on the agent's own tools, named so it
        # knows dirs/URLs/conversation/paste all route through existing tools.
        prompt = build_learn_prompt("learn from somewhere")
        for tool in ("read_file", "search_files", "web_extract"):
            assert tool in prompt

    def test_empty_request_falls_back_to_the_conversation(self):
        # Bare /learn should distill "what we just did", not error.
        prompt = build_learn_prompt("")
        assert "conversation" in prompt.lower()
        # And still carries the standards + save instruction.
        assert "skill_manage" in prompt

    def test_whitespace_only_request_is_treated_as_empty(self):
        assert build_learn_prompt("   \n  ") == build_learn_prompt("")

    def test_description_length_rule_is_in_the_standards(self):
        # The single most-violated rule must be explicit in the prompt.
        assert "60" in _AUTHORING_STANDARDS

    def test_teaches_the_full_hardline_standards(self):
        # /learn must teach ALL the CONTRIBUTING.md skill rules, not just the
        # description length — otherwise distilled skills miss platform gating,
        # author credit, and the tool-framing table. Lock the coverage in.
        std = _AUTHORING_STANDARDS.lower()
        # #1 description: the count-and-trim self-check (the reported bug).
        assert "count" in std and "60" in std
        # #3 platforms gating against OS-bound primitives.
        assert "platforms" in std
        # author is always the literal Hermes, never the host/OS identity (#52368).
        assert "author: always the literal value `hermes`" in std
        assert "never fill it from the host" in std
        # #2 Hermes-tool framing names the wrapped tools, not shell utilities.
        for tool in ("read_file", "search_files", "patch", "write_file"):
            assert tool in std
        # #6 scripts/references/templates layout.
        assert "scripts/" in _AUTHORING_STANDARDS


class TestLearnRegistryWiring:
    def test_learn_is_registered_and_resolves(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("learn")
        assert cmd is not None
        assert cmd.name == "learn"

    def test_learn_is_in_tools_and_skills_category(self):
        from hermes_cli.commands import resolve_command

        assert resolve_command("learn").category == "Tools & Skills"

    def test_learn_works_on_the_gateway(self):
        # /learn must reach the gateway runner (it's a both-surfaces command),
        # not be CLI-only.
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

        assert "learn" in GATEWAY_KNOWN_COMMANDS

    def test_learn_is_not_cli_only(self):
        from hermes_cli.commands import resolve_command

        assert not resolve_command("learn").cli_only


# ===========================================================================
# Decomposition & delta-update enhancement
#
# The enhancement layers flag-driven modes (decompose / single / update /
# dry-run) onto the open-ended baseline above. build_learn_prompt stays a pure,
# total, deterministic str->str function, so these contracts are expressed as
# assertions over a representative request corpus.
# ===========================================================================

import itertools

import pytest

from agent.learn_prompt import LearnMode, parse_learn_request


def _norm(text: str) -> str:
    """Collapse whitespace so wrapped multi-word phrases match regardless of
    the line-wrap column."""
    return " ".join(text.split())


_BASE_TEXTS = [
    "",
    "   ",
    "document the deployment runbook",
    "summarize our REST API conventions and error handling",
    "cafe deja-vu handle unicode and emoji correctly",
    "discuss --decompose as a concept without triggering it",
    "a" * 40,
]

_FLAG_SETS = [
    [],
    ["--decompose"],
    ["--no-decompose"],
    ["--dry-run"],
    ["--decompose", "--dry-run"],
    ["--no-decompose", "--dry-run"],
    ["--decompose", "--no-decompose"],
    ["--update", "my-skill"],
    ["--update", "my-skill", "--dry-run"],
    ["--update", "parent-skill", "--decompose"],
    ["--update"],
]


def _request_corpus() -> list[str]:
    corpus: list[str] = []
    for base, flags in itertools.product(_BASE_TEXTS, _FLAG_SETS):
        if flags:
            corpus.append((" ".join(flags) + " " + base).strip())
            corpus.append((base + " " + " ".join(flags)).strip())
        else:
            corpus.append(base)
    corpus += [
        "update my-skill with the new authentication docs",
        "update billing-parent with new tax rules --dry-run",
        "UPDATE Cap-Skill WITH mixed case keywords",
        "update",
        "update foo",
        "foo --update",
    ]
    return corpus


CORPUS = _request_corpus()


class TestLearnFlagParsing:
    @pytest.mark.parametrize(
        "req,mode,target,dry,conflict",
        [
            ("just some text", LearnMode.AUTO, None, False, False),
            ("--decompose text", LearnMode.DECOMPOSE, None, False, False),
            ("--no-decompose text", LearnMode.SINGLE, None, False, False),
            ("--decompose --no-decompose text", LearnMode.SINGLE, None, False, True),
            ("--update foo with bar", LearnMode.UPDATE, "foo", False, False),
            ("update foo with bar", LearnMode.UPDATE, "foo", False, False),
            ("--update foo --decompose bar", LearnMode.UPDATE, "foo", False, False),
            ("--dry-run text", LearnMode.AUTO, None, True, False),
            ("--decompose --dry-run text", LearnMode.DECOMPOSE, None, True, False),
            ("--update", LearnMode.AUTO, None, False, False),
        ],
    )
    def test_mode_resolution_table(self, req, mode, target, dry, conflict):
        d = parse_learn_request(req)
        assert d.mode is mode
        assert d.update_target == target
        assert d.dry_run is dry
        assert d.conflicting_flags is conflict

    def test_update_precedence_over_decompose(self):
        d = parse_learn_request("--decompose --update myskill with stuff")
        assert d.mode is LearnMode.UPDATE and d.update_target == "myskill"

    def test_unrecognized_flags_pass_through(self):
        d = parse_learn_request("document this --unknown-flag and --another")
        assert "--unknown-flag" in d.source_text and "--another" in d.source_text
        assert d.mode is LearnMode.AUTO

    def test_directive_is_frozen(self):
        d = parse_learn_request("text")
        with pytest.raises(Exception):
            d.mode = LearnMode.DECOMPOSE  # type: ignore[misc]


class TestLearnPromptModes:
    @pytest.mark.parametrize("req", CORPUS)
    def test_pure_total_deterministic(self, req):
        out1 = build_learn_prompt(req)
        out2 = build_learn_prompt(req)
        assert isinstance(out1, str) and out1 != ""
        assert out1 == out2

    @pytest.mark.parametrize("req", CORPUS)
    def test_invariants_present_in_every_prompt(self, req):
        out = build_learn_prompt(req)
        # Authoring standards travel with every mode.
        assert _AUTHORING_STANDARDS in out
        # Gather tools + skill_manage-only writes named in every mode.
        for tok in ("read_file", "search_files", "web_extract", "skill_manage"):
            assert tok in out

    @pytest.mark.parametrize("req", CORPUS)
    def test_source_text_preserved(self, req):
        d = parse_learn_request(req)
        out = build_learn_prompt(req)
        if d.source_text:
            assert d.source_text in out

    def test_auto_emits_threshold_and_single_default(self):
        auto = [r for r in CORPUS if parse_learn_request(r).mode is LearnMode.AUTO]
        assert auto
        for req in auto:
            norm = _norm(build_learn_prompt(req))
            assert "Decomposition Threshold" in norm
            assert "MAX_SKILL_CONTENT_CHARS" in norm
            assert "100,000 characters" in norm
            assert "distinct single-responsibility topic" in norm
            assert "`create`" in norm

    def test_decompose_forces_hierarchy(self):
        for req in ("--decompose document everything", "build the kit --decompose"):
            assert parse_learn_request(req).mode is LearnMode.DECOMPOSE
            norm = _norm(build_learn_prompt(req))
            assert "Parent Skill" in norm and "Child Skill" in norm
            assert "Progressive Disclosure" in norm
            assert "metadata.hermes.children" in norm
            assert "metadata.hermes.parent" in norm
            assert "metadata.hermes.depends_on" in norm
            assert "do not introduce any new top-level frontmatter" in norm.lower()

    def test_no_decompose_single_and_conflict(self):
        out = build_learn_prompt("--no-decompose keep it one skill")
        assert parse_learn_request("--no-decompose x").mode is LearnMode.SINGLE
        assert "author exactly one skill" in _norm(out).lower()

        conflict = "--decompose --no-decompose ambiguous request"
        d = parse_learn_request(conflict)
        assert d.mode is LearnMode.SINGLE and d.conflicting_flags
        norm = _norm(build_learn_prompt(conflict)).lower()
        assert "these flags conflict" in norm

    def test_update_minimal_patching_and_versioning(self):
        for req, target in [
            ("--update auth-skill with new oauth flow", "auth-skill"),
            ("update billing with tax rules", "billing"),
            ("--update my-skill", "my-skill"),
        ]:
            d = parse_learn_request(req)
            assert d.mode is LearnMode.UPDATE and d.update_target == target
            norm = _norm(build_learn_prompt(req))
            assert f"`{target}`" in norm
            assert "Read the existing skill before modifying" in norm
            assert "`patch`" in norm and "`write_file`" in norm
            assert "Reserve the `edit` action for a full rewrite" in norm
            assert "report that the skill was not found" in norm
            assert "Increment the `version`" in norm
            assert "summary of the change" in norm
            assert "metadata.hermes.children" in norm

    def test_dry_run_previews_and_suppresses_writes(self):
        dry = [r for r in CORPUS if parse_learn_request(r).dry_run]
        assert dry
        for req in dry:
            norm = _norm(build_learn_prompt(req))
            assert "DRY RUN" in norm
            assert "planned skills to create" in norm
            assert "planned patches" in norm
            assert "planned supporting files" in norm
            for action in ("create", "edit", "patch", "delete", "write_file", "remove_file"):
                assert f"`{action}`" in norm

    def test_dry_run_with_decompose_lists_names(self):
        norm = _norm(build_learn_prompt("--decompose --dry-run the big manual"))
        assert "Parent Skill and Child Skill names" in norm


class TestLearnRegistryAdvertisesFlags:
    def test_args_hint_advertises_flags(self):
        from hermes_cli.commands import resolve_command

        hint = resolve_command("learn").args_hint or ""
        for flag in ("--decompose", "--no-decompose", "--update", "--dry-run"):
            assert flag in hint

    def test_no_new_model_tool_registered(self):
        from tools.registry import registry

        assert "learn" not in set(registry.get_all_tool_names())
