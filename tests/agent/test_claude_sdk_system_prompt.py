"""System prompt append — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""


import pytest

from tests.agent.claude_sdk_fakes import (
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestSystemPromptAppend:
    # W2 (composer parity): the append is composed from Hermes' NATIVE
    # builders — memory gauge via MemoryStore.format_for_system_prompt,
    # guidance constants from agent.prompt_builder, the skills index via
    # build_skills_system_prompt — never re-implemented formats. Guidance
    # appears ONLY for tools that are actually callable through the MCP
    # shims. Deliberate pin updates from W1 are annotated inline.

    @staticmethod
    def _home(tmp_path, monkeypatch, *, soul=None, memory=None, user=None, budget=None):
        hermes_home = tmp_path / "hermes"
        memories = hermes_home / "memories"
        memories.mkdir(parents=True)
        if memory is not None:
            (memories / "MEMORY.md").write_text(memory)
        if user is not None:
            (memories / "USER.md").write_text(user)
        import hermes_cli.config as cfg

        append_file = ""
        if soul is not None:
            soul_file = tmp_path / "SOUL.md"
            soul_file.write_text(soul)
            append_file = str(soul_file)
        # config.yaml is the only interface for the persona file
        # (agent.claude_agent_sdk.append_file); the old env var is gone.
        # Patching unconditionally also isolates the suite from a developer's
        # real config.yaml, which would otherwise leak a live append_file in.
        sdk_cfg = {"append_file": append_file}
        # Only inject the budget key when a test asks for one: absent must stay
        # distinguishable from present-but-invalid, which take different paths.
        if budget is not None:
            sdk_cfg["append_total_max_chars"] = budget
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": sdk_cfg}},
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        return hermes_home

    def test_soul_first_and_user_content_present(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(
            tmp_path, monkeypatch,
            soul="# I am the persona under test",
            user="The user prefers concise results",
        )
        out = build_system_prompt_append()
        assert out is not None
        assert out.startswith("# I am the persona under test")
        assert "The user prefers concise results" in out

    def test_native_soul_md_autoloads_when_append_file_unset(
        self, tmp_path, monkeypatch
    ):
        # R2 (#65982, romain-bury): the native composer treats
        # $HERMES_HOME/SOUL.md as identity slot #1; W2 composer parity means
        # this path must load it too when no explicit append_file overrides.
        from agent.claude_sdk_runtime import build_system_prompt_append

        home = self._home(tmp_path, monkeypatch)
        (home / "SOUL.md").write_text("# Native soul identity")
        out = build_system_prompt_append()
        assert out is not None
        assert out.startswith("# Native soul identity")

    def test_append_file_wins_over_native_soul_md(self, tmp_path, monkeypatch):
        # append_file stays the explicit operator override.
        from agent.claude_sdk_runtime import build_system_prompt_append

        home = self._home(
            tmp_path, monkeypatch, soul="# Override persona"
        )
        (home / "SOUL.md").write_text("# Native soul identity")
        out = build_system_prompt_append()
        assert out is not None
        assert out.startswith("# Override persona")
        assert "# Native soul identity" not in out

    def test_workspace_context_file_is_in_sdk_append(self, tmp_path, monkeypatch):
        """SDK turns must receive the same Hermes project instructions as native turns."""
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".hermes.md").write_text(
            "# Workspace contract\nRun the project check before reporting success."
        )

        out = build_system_prompt_append(cwd=str(workspace)) or ""

        assert "# Project Context" in out
        assert "Run the project check before reporting success." in out
        # Context-file discovery uses a process-local warning queue. Drain it
        # so this direct builder test cannot leak state into native prompt tests.
        from agent.prompt_builder import drain_truncation_warnings

        drain_truncation_warnings()

    def test_coding_workspace_snapshot_is_in_sdk_append(self, tmp_path, monkeypatch):
        """SDK turns retain the workspace and operator tail, not false tool guidance."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context

        self._home(tmp_path, monkeypatch)
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: (["Coding posture"], ["Workspace snapshot"], ["Coding tail"]),
        )

        out = build_system_prompt_append(cwd=str(tmp_path), platform="telegram") or ""

        # The native prefix names patch/write_file/terminal/todo, which are not
        # present on the default SDK surface. The claude_code preset plus the
        # SDK-specific inspection guidance own behavior; only the workspace
        # snapshot and configured operator tail are portable here.
        assert "Coding posture" not in out
        assert "Workspace snapshot" in out
        assert "Coding tail" in out
        from agent.prompt_builder import drain_truncation_warnings

        drain_truncation_warnings()

    def test_project_context_preserves_native_fallback_policy(self, tmp_path, monkeypatch):
        """A fallback cwd stays None so prompt_builder can guard install trees."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)
        captured = {}

        def fake_project_context(**kwargs):
            captured.update(kwargs)
            return ""

        monkeypatch.setattr(prompt_builder, "build_context_files_prompt", fake_project_context)
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], [], []),
        )

        build_system_prompt_append(cwd=None, platform="telegram")

        assert captured["cwd"] is None
        assert captured["skip_soul"] is True
        assert captured["allow_install_tree_fallback"] is False

    def test_skip_project_context_keeps_coding_snapshot(self, tmp_path, monkeypatch):
        """skip_context_files does not disable the independent coding snapshot."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)

        def unexpected_project_context(**_kwargs):
            raise AssertionError("project context must stay disabled")

        monkeypatch.setattr(
            prompt_builder,
            "build_context_files_prompt",
            unexpected_project_context,
        )
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], ["WORKSPACE-WITH-PROJECT-FILES-DISABLED"], []),
        )

        out = build_system_prompt_append(
            cwd=str(tmp_path),
            include_project_context=False,
        ) or ""

        assert "WORKSPACE-WITH-PROJECT-FILES-DISABLED" in out

    def test_workspace_snapshot_survives_large_project_context(self, tmp_path, monkeypatch):
        """A large project file must not silently evict the SDK workspace snapshot."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)
        monkeypatch.setattr(
            prompt_builder,
            "build_context_files_prompt",
            lambda **_kwargs: "PROJECT-CONTEXT " + "p" * 16_000,
        )
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], ["WORKSPACE-SNAPSHOT " + "w" * 3_500], []),
        )

        out = build_system_prompt_append(cwd=str(tmp_path)) or ""

        assert "PROJECT-CONTEXT" in out
        assert "WORKSPACE-SNAPSHOT" in out

    def test_oversized_workspace_snapshot_is_capped_not_dropped(self, tmp_path, monkeypatch):
        """An oversized coding snapshot remains represented within the workspace cap."""
        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            build_system_prompt_append,
        )
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)
        monkeypatch.setattr(prompt_builder, "build_context_files_prompt", lambda **_kwargs: "")
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], ["OVERSIZED-WORKSPACE " + "w" * 22_000], []),
        )

        out = build_system_prompt_append(cwd=str(tmp_path)) or ""

        assert "OVERSIZED-WORKSPACE" in out
        assert len(out) <= _APPEND_TOTAL_MAX_CHARS

    def test_workspace_snapshot_survives_capped_soul(self, tmp_path, monkeypatch):
        """The workspace block must fit after the maximum-size SDK soul block."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch, soul="CAPPED-SOUL " + "s" * 8_000)
        monkeypatch.setattr(prompt_builder, "build_context_files_prompt", lambda **_kwargs: "")
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], ["SOUL-COMPATIBLE-WORKSPACE " + "w" * 12_000], []),
        )

        out = build_system_prompt_append(cwd=str(tmp_path)) or ""

        assert "CAPPED-SOUL" in out
        assert "SOUL-COMPATIBLE-WORKSPACE" in out

    def test_project_context_warning_queue_drains_after_builder_error(self, tmp_path, monkeypatch):
        """A failed SDK context build cannot leak warnings into later native prompts."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)

        def record_then_fail(**_kwargs):
            if (warnings := prompt_builder._truncation_warnings.get()) is None:
                prompt_builder._truncation_warnings.set(warnings := [])
            warnings.append("SDK test warning")
            raise RuntimeError("transient context read failure")

        monkeypatch.setattr(prompt_builder, "build_context_files_prompt", record_then_fail)
        build_system_prompt_append(cwd=str(tmp_path))

        assert prompt_builder.drain_truncation_warnings() == []

    def test_gauge_blocks_are_the_native_render(self, tmp_path, monkeypatch):
        # Byte-pin: the memory/user blocks are EXACTLY what the native
        # composer injects (MemoryStore.format_for_system_prompt output,
        # gauge header included) — never a re-implementation.
        from agent.claude_sdk_runtime import build_system_prompt_append
        from tools.memory_tool import load_on_disk_store

        self._home(
            tmp_path, monkeypatch,
            memory="ci runs on the drone server",
            user="prefers squash merges",
        )
        store = load_on_disk_store()
        expected_memory = store.format_for_system_prompt("memory")
        expected_user = store.format_for_system_prompt("user")
        assert "MEMORY (your personal notes) [" in expected_memory  # sanity
        assert "USER PROFILE (who the user is) [" in expected_user

        out = build_system_prompt_append()
        assert expected_memory in out
        assert expected_user in out

    def test_mcp_inspection_preference_is_in_effective_sdk_prompt(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch)
        out = build_system_prompt_append() or ""
        assert "For multi-step tool work, provide a brief user-facing status" in out
        assert "never reveal private reasoning" in out
        assert "prefer the Hermes MCP `read_file` and `search_files` tools before Bash" in out
        assert "database client, process/service state, network operation" in out
        assert "Bash remains subject to normal approval" in out

    def test_consolidated_memory_guidance_is_preserved_verbatim(
        self, tmp_path, monkeypatch
    ):
        # Upstream's MEMORY_GUIDANCE routes procedures to skills and (since its
        # 2026-09 rewording) names the skill_manage write tool, which this runtime
        # cannot call: the sentence that instructs it is dropped, everything else
        # is carried verbatim. The skills-index boilerplate is filtered the same way.
        from agent.claude_sdk_runtime import (
            _strip_uncallable_tool_guidance,
            build_system_prompt_append,
        )
        from agent.prompt_builder import MEMORY_GUIDANCE

        self._home(tmp_path, monkeypatch, memory="uses trunk-based development")
        stripped = _strip_uncallable_tool_guidance(MEMORY_GUIDANCE)
        assert "skill_manage" not in stripped
        assert "Memory is the narrow exception" in stripped
        assert "declarative facts" in stripped
        assert stripped.startswith(MEMORY_GUIDANCE[:60])

        out = build_system_prompt_append()
        assert stripped in out
        assert "skill_manage" not in out
        # Disambiguation addendum (caught live): the claude_code preset has
        # its own file-based memory convention; the append must pin the
        # hermes-tools memory tool as the ONLY durable store.
        assert "ONLY durable memory" in out
        assert "hermes-tools MCP server" in out
        # Reworded after the adversarial review PROVED the preset's memory
        # dir DOES persist per-cwd: the addendum must state true facts
        # (unmanaged/disposable), never the false "will not be injected".
        assert "disposable" in out
        assert "will not be injected" not in out

    def test_skills_guidance_never_injected(self, tmp_path, monkeypatch):
        # SKILLS_GUIDANCE instructs skill_manage — unexposed by design.
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch, memory="a fact")
        out = build_system_prompt_append()
        assert "skill_manage" not in out

    def test_session_search_guidance_always_present(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append
        from agent.prompt_builder import SESSION_SEARCH_GUIDANCE

        self._home(tmp_path, monkeypatch)  # no memory files at all
        out = build_system_prompt_append()
        assert out is not None
        assert SESSION_SEARCH_GUIDANCE in out
        # Query-style addendum (observed live: ANDy multi-term queries miss).
        assert "ALL terms must match" in out

    def test_memory_disabled_removes_blocks_and_guidance(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append
        import hermes_cli.config as cfg

        self._home(tmp_path, monkeypatch, memory="should not appear")
        monkeypatch.setattr(
            cfg, "load_config", lambda *a, **k: {"memory": {"memory_enabled": False}}
        )
        out = build_system_prompt_append()
        assert "should not appear" not in (out or "")
        assert "You have persistent memory" not in (out or "")
        # session_search still works when memory is off — its guidance stays.
        assert "session_search" in (out or "")

    def test_external_memory_provider_removes_tool_guidance(self, tmp_path, monkeypatch):
        # memory.provider: honcho (or ANY external backend) leaves the memory
        # shim UNREGISTERED (hermes_tools_mcp_server_shims.stateless_shim_definitions
        # requires enabled AND no external provider), so the append must not
        # instruct or advertise an absent tool. The on-disk store block stays:
        # external providers run alongside the builtin store, and its facts
        # remain readable. Proven red-first against the enabled-only gate.
        import agent.prompt_builder as pb
        import hermes_cli.config as cfg
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch, memory="a durable fact")
        monkeypatch.setattr(
            cfg,
            "load_config",
            lambda *a, **k: {
                "memory": {"memory_enabled": True, "provider": "honcho"}
            },
        )
        captured = {}

        def fake_index(**kwargs):
            captured.update(kwargs)
            return ""

        monkeypatch.setattr(pb, "build_skills_system_prompt", fake_index)
        out = build_system_prompt_append() or ""
        assert "You have persistent memory" not in out
        assert "ONLY durable memory" not in out
        # The store block itself survives — facts stay readable.
        assert "a durable fact" in out
        # session_search is unaffected.
        assert "session_search" in out
        # And the skills filter is not told the tool exists.
        tools = captured.get("available_tools") or set()
        assert "memory" not in tools
        assert "session_search" in tools

    def test_session_line_and_platform_hint(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append
        from agent.prompt_builder import PLATFORM_HINTS

        self._home(tmp_path, monkeypatch)
        out = build_system_prompt_append(
            platform="telegram", session_id="sess-77", model="claude-opus-4-8"
        )
        assert "Conversation started:" in out  # date-only, native format
        assert "Session ID: sess-77" in out
        assert "Model: claude-opus-4-8" in out
        assert "Provider: claude-agent-sdk" in out
        assert PLATFORM_HINTS["telegram"].strip() in out

    def test_unknown_platform_no_hint_and_none_safe(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch)
        out = build_system_prompt_append(platform="faxmachine")
        assert out is not None  # None-safe, no crash, no bogus hint

    def test_budget_skips_oversized_block_keeps_later_blocks(self, tmp_path, monkeypatch):
        # Whole-block budget policy: a block that does not fit is SKIPPED
        # entirely (never truncated mid-block) and later, smaller blocks
        # still make it in. An oversized hand-edited MEMORY.md must not
        # evict the guidance. (Deliberate pin update from W1's 8000-char
        # raw-file cap: the store renders whole blocks; the budget governs.)
        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            build_system_prompt_append,
        )

        self._home(tmp_path, monkeypatch, memory="y" * (_APPEND_TOTAL_MAX_CHARS + 5000))
        out = build_system_prompt_append()
        assert "yyyyyyyyyy" not in out  # oversized memory block skipped whole
        assert "session_search" in out  # later block survived
        assert len(out) <= _APPEND_TOTAL_MAX_CHARS

    def test_budget_eviction_warns_and_names_the_block(
        self, tmp_path, monkeypatch, caplog
    ):
        # An eviction deletes standing instructions the operator believes are
        # in force, so it must be audible. This was DEBUG — which is how the
        # identity fix could have silently traded SOUL.md in for the
        # MCP-inspection block with nothing in the log to say so.
        import logging

        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            build_system_prompt_append,
        )

        oversized = "# Hand-edited memory\n" + "y" * (_APPEND_TOTAL_MAX_CHARS + 5000)
        self._home(tmp_path, monkeypatch, memory=oversized)
        with caplog.at_level(logging.WARNING):
            out = build_system_prompt_append()

        assert "yyyyyyyyyy" not in out
        evictions = [
            r for r in caplog.records
            if "append budget" in r.getMessage() and r.levelno >= logging.WARNING
        ]
        assert evictions, "an evicted block must be reported above DEBUG"
        message = evictions[0].getMessage()
        # Named by the store's own header, so the loss is actionable...
        assert "MEMORY" in message
        # ...but header ONLY: the body must never reach the log.
        assert "yyyyyyyyyy" not in message

    def test_budget_override_seats_a_block_the_default_evicts(
        self, tmp_path, monkeypatch
    ):
        # The ceiling is a per-box cost decision, so it must be reachable from
        # config without a code edit.
        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            build_system_prompt_append,
        )

        # Separate homes: _home() is not re-entrant on one tmp_path, and the
        # two halves must differ ONLY in the configured ceiling.
        big = "# Big memory\n" + "y" * (_APPEND_TOTAL_MAX_CHARS + 5000)
        self._home(tmp_path / "default", monkeypatch, memory=big)
        assert "yyyyyyyyyy" not in build_system_prompt_append()  # evicted at default

        self._home(
            tmp_path / "raised",
            monkeypatch,
            memory=big,
            budget=_APPEND_TOTAL_MAX_CHARS * 3,
        )
        assert "yyyyyyyyyy" in build_system_prompt_append()  # seated when raised

    @pytest.mark.parametrize(
        "bad", ["not-a-number", 0, -5, True, 1.5, float("inf")]
    )
    def test_invalid_budget_override_falls_back_to_default_with_warning(
        self, tmp_path, monkeypatch, caplog, bad
    ):
        # A typo'd or zero ceiling must never become a silent behaviour
        # change: 0 would strip the entire append, identity included.
        import logging

        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            _append_total_max_chars,
        )

        self._home(tmp_path, monkeypatch, budget=bad)
        with caplog.at_level(logging.WARNING):
            assert _append_total_max_chars() == _APPEND_TOTAL_MAX_CHARS
        assert any(
            "append_total_max_chars" in r.getMessage() for r in caplog.records
        )

    def test_restoring_identity_does_not_evict_a_previously_seated_block(
        self, tmp_path, monkeypatch
    ):
        # The regression the independent review caught, stated as the
        # relationship it actually is: seating SOUL.md must not be paid for by
        # silently dropping a block that fit without it. Sizes are calibrated
        # (memory 8400) so the identity block genuinely pushes the total past
        # the historical 20000 ceiling — at 20000 this test is RED, which is
        # the whole point of the raised default.
        from agent.claude_sdk_runtime import (
            _MCP_INSPECTION_PREFERENCE,
            build_system_prompt_append,
        )

        mcp = _MCP_INSPECTION_PREFERENCE.strip()
        common = dict(user="u" * 4921, memory="m" * 8400)

        # Baseline: no append_file, no hand-written identity file (native
        # load_soul_md() scaffolds a short default), historical ceiling —
        # MCP block fits.
        self._home(tmp_path / "no-soul", monkeypatch, budget=20000, **common)
        assert mcp in build_system_prompt_append()

        # Same load, at the shipped default, with a real identity file
        # written straight into $HERMES_HOME (append_file stays unset, so
        # this exercises the native load_soul_md() path): identity seated
        # AND the MCP block survives. (At budget=20000 the MCP block is
        # evicted — calibrated and verified 2026-08-18 on Main.)
        home = self._home(tmp_path / "with-soul", monkeypatch, **common)
        (home / "SOUL.md").write_text("I am the agent.\n" + "s" * 3200)
        out = build_system_prompt_append()

        assert "I am the agent." in out  # identity seated
        assert "m" * 100 in out and "u" * 100 in out  # memory + profile seated
        assert mcp in out  # ...and NOT paid for with this one

    def test_eviction_warning_never_derives_label_from_block_content(
        self, tmp_path, monkeypatch, caplog
    ):
        import logging

        import agent.prompt_builder as pb
        from agent.claude_sdk_runtime import build_system_prompt_append

        secret = "private skill detail that must never reach logs"
        self._home(tmp_path, monkeypatch, budget=1000)
        monkeypatch.setattr(
            pb,
            "build_skills_system_prompt",
            lambda **_kwargs: secret + "x" * 5000,
        )

        with caplog.at_level(logging.WARNING):
            build_system_prompt_append()

        messages = [record.getMessage() for record in caplog.records]
        assert any("skills index" in message for message in messages)
        assert all(secret not in message for message in messages)

    def test_skills_index_wiring(self, tmp_path, monkeypatch):
        # The index rides the NATIVE builder; we pin OUR wiring — called
        # with the honest MCP-exposed tool set (shims included).
        import agent.prompt_builder as pb
        from agent.claude_sdk_runtime import build_system_prompt_append
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

        self._home(tmp_path, monkeypatch)
        captured = {}

        def fake_index(**kwargs):
            captured.update(kwargs)
            # Includes the index's real unconditional boilerplate sentence —
            # caught LIVE on the deployed box: the native index instructs
            # skill_manage regardless of available_tools, and the strip must
            # remove it (a tmp home's empty index made the old pin vacuous).
            return (
                "## Skills (mandatory)\n"
                "If a skill has issues, fix it with skill_manage(action='patch').\n"
                "- fixture-skill: proves the wiring"
            )

        monkeypatch.setattr(pb, "build_skills_system_prompt", fake_index)
        out = build_system_prompt_append()
        assert "fixture-skill: proves the wiring" in out
        assert "skill_manage" not in out
        tools = captured.get("available_tools") or set()
        assert "memory" in tools and "session_search" in tools
        assert {"read_file", "search_files"} <= tools
        assert not tools & {"terminal", "shell", "write_file", "patch", "process"}
        assert set(EXPOSED_TOOLS) <= tools

    def test_root_files_are_not_read(self, tmp_path, monkeypatch):
        # Negative control (W1): ONE canonical location. Files left at the
        # HERMES_HOME root must NOT be injected.
        from agent.claude_sdk_runtime import build_system_prompt_append

        hermes_home = tmp_path / "hermes"
        (hermes_home / "memories").mkdir(parents=True)
        (hermes_home / "USER.md").write_text("stale root copy")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        assert "stale root copy" not in (build_system_prompt_append() or "")

    def test_memory_shim_write_is_visible_to_next_append(self, tmp_path, monkeypatch):
        # The loop closes: a fact saved through the stateless MCP shim must
        # appear in the next session's system-prompt append.
        from agent.claude_sdk_runtime import build_system_prompt_append
        from agent.transports.hermes_tools_mcp_server_shims import dispatch_memory

        self._home(tmp_path, monkeypatch)
        dispatch_memory(
            {"action": "add", "target": "memory", "content": "the beta build ships friday"}
        )
        out = build_system_prompt_append()
        assert out is not None
        assert "the beta build ships friday" in out

    def test_empty_home_still_provides_guidance(self, tmp_path, monkeypatch):
        # Deliberate pin update (was: no sources → None). Since W2 the
        # append always carries the recall/memory behavior contract — a
        # brand-new box still gets guidance, so the brain knows its tools.
        from agent.claude_sdk_runtime import build_system_prompt_append

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # empty dir
        out = build_system_prompt_append()
        assert out is not None
        assert "session_search" in out


class TestAuxLaneSubscriptionRouting:
    def test_aux_auto_detect_uses_same_sdk_subscription_lane(self, monkeypatch):
        # Auto auxiliary work must not fall through to a metered provider, but
        # it can safely use the same subscription-owned Agent SDK one-shot path.
        from agent.auxiliary_client import _resolve_auto_route
        from agent.claude_sdk_aux_client import ClaudeSdkAuxClient

        monkeypatch.setenv("OPENROUTER_API_KEY", "«redacted:sk-…»")
        client, model, effective_provider = _resolve_auto_route(main_runtime={
            "provider": "claude-agent-sdk",
            "model": "claude-opus-4-8",
            "api_mode": "claude_agent_sdk",
            "base_url": "",
            "api_key": "claude-subscription-oauth",
        })
        assert isinstance(client, ClaudeSdkAuxClient)
        assert model == "claude-opus-4-8"
        assert effective_provider == "claude-agent-sdk"
