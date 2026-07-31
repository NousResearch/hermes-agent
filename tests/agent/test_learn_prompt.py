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




    def test_separates_sources_from_requirements(self):
        # The reported bug (@GrenFX, Jun 2026): when a request leads with a
        # path/URL, the agent fetched it and ignored the trailing prose. The
        # prompt must tell the agent the request can MIX sources and
        # requirements, and that prose after a source is authoring guidance to
        # honor — not noise to drop.
        prompt = build_learn_prompt(
            "https://api.example.com/docs focus on the auth flow, skip deprecated bits"
        )
        low = prompt.lower()
        # Carries the whole request verbatim (no truncation at the URL).
        assert "focus on the auth flow, skip deprecated bits" in prompt
        # Explicitly distinguishes sources from requirements.
        assert "requirement" in low
        # Names the failure mode it's guarding against.
        assert "never fetch the first source" in low




    def test_teaches_the_full_hardline_standards(self):
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



    def test_learn_is_not_cli_only(self):
        from hermes_cli.commands import resolve_command

        assert not resolve_command("learn").cli_only
