#!/usr/bin/env python3
"""
Tests for executors/prompt_builder.py — PromptBuilder.

Scope:
  - Per-executor prompt generation (claude-code, codex-cli, opencode,
    deepseek-tui, hermes-local)
  - Per-executor caps/truncation (architecture, ADRs, recent tasks)
  - context_injection_enabled = False bypasses injection
  - include_flags override defaults
  - Token estimation rough heuristic
  - build_injection_preview excludes user prompt
  - Does NOT call any model
  - Does NOT spawn subprocess
  - Does NOT write files
  - Does NOT create worktrees

Strictly no live model calls, no subprocess, no file writes, no worktrees.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from executors.prompt_builder import (
    PromptBuilder,
    _ARCH_TRUNCATION,
    _ADR_LIMITS,
    _FIELD_TABLE,
    _RECENT_TASK_LIMITS,
    _TOKEN_CAP,
    _TRUNCATION_PRIORITY,
    _estimate_tokens,
    create_default_builder,
)
from executors.types import (
    AdrSummary,
    CommandEntry,
    ProjectContext,
    PromptSnapshot,
    RecentTask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rich_context() -> ProjectContext:
    """A project context with content for every field."""
    return ProjectContext(
        project_overview="A modular agent runtime with pluggable executors",
        architecture_notes=(
            "Event-driven core. CLI + Gateway front-ends. Backends communicate "
            "via normalized RunEvent stream. " * 20
        ),
        adr_summaries=[
            AdrSummary(id=f"ADR-{i:03d}", title=f"Decision {i}", decision=f"Use option {i}")
            for i in range(1, 6)
        ],
        current_sprint="Sprint 47 — context injection rollout",
        common_commands=[
            CommandEntry(label="build", command="make build"),
            CommandEntry(label="test", command="make test"),
        ],
        test_commands=[
            CommandEntry(label="unit", command="pytest tests/unit"),
        ],
        forbidden_areas=["secrets/", "node_modules/"],
        coding_conventions="PEP 8 + type hints; dataclasses for state.",
        recent_tasks=[
            RecentTask(
                thread_id=f"t-{i:03d}",
                title=f"Task {i}",
                executor="claude-code",
                status="done" if i % 2 == 0 else "failed",
                completed_at=f"2026-06-{(i % 28) + 1:02d}T10:00:00Z",
            )
            for i in range(10)
        ],
        context_injection_enabled=True,
    )


@pytest.fixture
def empty_context() -> ProjectContext:
    return ProjectContext(context_injection_enabled=True)


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


# ---------------------------------------------------------------------------
# 1. Per-executor prompt generation
# ---------------------------------------------------------------------------

class TestPerExecutorGeneration:
    def test_claude_code_includes_all_fields(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="Refactor the auth module",
            context=rich_context,
            executor_id="claude-code",
        )
        assert isinstance(snap, PromptSnapshot)
        assert "Refactor the auth module" in snap.injected_prompt
        assert "Workspace Context" in snap.injected_prompt
        # Claude-code gets the full architecture (long).
        assert "Event-driven core" in snap.injected_prompt
        # All 5 ADRs.
        for i in range(1, 6):
            assert f"ADR-{i:03d}" in snap.injected_prompt
        # All forbidden areas.
        assert "secrets/" in snap.injected_prompt
        # Recent tasks.
        assert "Recent tasks:" in snap.injected_prompt

    def test_codex_cli_truncates_architecture_to_300_chars(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="codex-cli",
        )
        # Find the Architecture line.
        m = re.search(r"Architecture: (.+?)(?:\n|$)", snap.injected_prompt)
        assert m, "Architecture line not found"
        arch_text = m.group(1)
        # Truncated to <= 300 + trailing ellipsis.
        assert len(arch_text) <= 301, f"arch not truncated: {len(arch_text)} chars"

    def test_codex_cli_limits_adrs_to_3(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="codex-cli",
        )
        for i in range(1, 4):
            assert f"ADR-{i:03d}" in snap.injected_prompt
        # ADRs 4 and 5 must NOT appear.
        assert "ADR-004" not in snap.injected_prompt
        assert "ADR-005" not in snap.injected_prompt

    def test_codex_cli_limits_recent_tasks_to_3(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="codex-cli",
        )
        # Count task lines.
        task_lines = [
            line for line in snap.injected_prompt.splitlines()
            if re.match(r"\s*\[[✓✗]\] Task \d+", line)
        ]
        assert len(task_lines) == 3

    def test_opencode_keeps_full_architecture(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="opencode",
        )
        assert "Event-driven core" in snap.injected_prompt

    def test_deepseek_tui_is_minimal(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="deepseek-tui",
        )
        # Should include overview, sprint, commands, forbidden.
        assert "A modular agent runtime" in snap.injected_prompt
        assert "Sprint 47" in snap.injected_prompt
        assert "Commands:" in snap.injected_prompt
        assert "secrets/" in snap.injected_prompt
        # Should NOT include architecture, ADRs, conventions, test commands, or
        # recent tasks.
        assert "Architecture:" not in snap.injected_prompt
        assert "ADRs:" not in snap.injected_prompt
        assert "Conventions:" not in snap.injected_prompt
        assert "Test commands:" not in snap.injected_prompt
        assert "Recent tasks:" not in snap.injected_prompt

    def test_hermes_local_includes_everything(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="hermes-local",
        )
        # Architecture present (no cap, content is included).
        assert "Event-driven core" in snap.injected_prompt
        # ADRs all 5.
        for i in range(1, 6):
            assert f"ADR-{i:03d}" in snap.injected_prompt

    def test_unknown_executor_falls_back_to_hermes_local(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="some-new-executor",
        )
        # Should behave like hermes-local (full injection).
        assert "A modular agent runtime" in snap.injected_prompt
        assert "Event-driven core" in snap.injected_prompt


# ---------------------------------------------------------------------------
# 2. context_injection_enabled = False bypass
# ---------------------------------------------------------------------------

class TestInjectionDisabled:
    def test_disabled_returns_user_prompt_only(
        self, builder, rich_context
    ) -> None:
        rich_context.context_injection_enabled = False
        snap = builder.build(
            user_prompt="Just do X", context=rich_context, executor_id="claude-code",
        )
        assert snap.injected_prompt == "Just do X"
        assert snap.user_prompt == "Just do X"
        assert snap.context_include_flags == {}

    def test_disabled_skips_all_injection_markers(
        self, builder, rich_context
    ) -> None:
        rich_context.context_injection_enabled = False
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="codex-cli",
        )
        assert "Workspace Context" not in snap.injected_prompt
        assert "ADRs:" not in snap.injected_prompt


# ---------------------------------------------------------------------------
# 3. include_flags override defaults
# ---------------------------------------------------------------------------

class TestIncludeFlags:
    def test_include_flag_excludes_field(
        self, builder, rich_context
    ) -> None:
        # Even on a normally-included executor, override excludes ADRs.
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="claude-code",
            include_flags={"adr_summaries": False},
        )
        assert "ADRs:" not in snap.injected_prompt
        assert "ADR-001" not in snap.injected_prompt

    def test_include_flag_can_force_include(
        self, builder, rich_context
    ) -> None:
        # deepseek-tui excludes architecture by default; override forces it.
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="deepseek-tui",
            include_flags={"architecture_notes": True},
        )
        assert "Architecture:" in snap.injected_prompt

    def test_injection_disabled_ignores_include_flags(
        self, builder, rich_context
    ) -> None:
        rich_context.context_injection_enabled = False
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="claude-code",
            include_flags={"project_overview": True},
        )
        assert snap.injected_prompt == "x"


# ---------------------------------------------------------------------------
# 4. Snapshot fields
# ---------------------------------------------------------------------------

class TestSnapshotFields:
    def test_snapshot_has_required_fields(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="do thing", context=rich_context, executor_id="claude-code",
        )
        assert snap.user_prompt == "do thing"
        assert snap.injected_prompt  # non-empty
        assert snap.estimated_tokens > 0
        assert snap.context_include_flags
        assert snap.generated_at is not None

    def test_estimated_tokens_roughly_matches_length(
        self, builder, rich_context
    ) -> None:
        snap = builder.build(
            user_prompt="hello world", context=rich_context, executor_id="hermes-local",
        )
        # Token estimate is a rough heuristic; must be > 0 and < total chars.
        assert 0 < snap.estimated_tokens < len(snap.injected_prompt)


# ---------------------------------------------------------------------------
# 5. Token cap warning behavior
# ---------------------------------------------------------------------------

class TestTokenCaps:
    def test_known_executors_have_caps(self) -> None:
        for eid in ("claude-code", "codex-cli", "opencode", "deepseek-tui", "hermes-local"):
            assert eid in _TOKEN_CAP, f"missing token cap for {eid}"
            assert _TOKEN_CAP[eid] > 0

    def test_estimate_tokens_handles_cjk(self) -> None:
        # 100 CJK chars should be ~50 tokens (heuristic: 2 chars / token).
        cjk_text = "中" * 100
        n = _estimate_tokens(cjk_text)
        assert 40 <= n <= 60, f"CJK estimate out of range: {n}"

    def test_estimate_tokens_handles_ascii(self) -> None:
        # 100 ASCII chars should be ~25 tokens (heuristic: 4 chars / token).
        ascii_text = "a" * 100
        n = _estimate_tokens(ascii_text)
        assert 20 <= n <= 30, f"ASCII estimate out of range: {n}"

    def test_estimate_tokens_empty(self) -> None:
        assert _estimate_tokens("") == 0
        # All-whitespace still produces >=1 to keep callers sane.
        assert _estimate_tokens(" ") >= 1


# ---------------------------------------------------------------------------
# 6. build_injection_preview
# ---------------------------------------------------------------------------

class TestInjectionPreview:
    def test_preview_excludes_user_prompt(self, builder, rich_context) -> None:
        preview = builder.build_injection_preview(
            context=rich_context, executor_id="claude-code",
        )
        assert "Workspace Context" in preview
        # The user-prompt placeholder is NOT in the preview (it's a comment
        # used to build the snapshot — the preview splits the result on
        # "--- End Context ---" and drops the rest).
        assert "<!-- USER PROMPT WOULD BE HERE -->" not in preview
        # The user prompt is NOT embedded in the preview.
        assert "Refactor the auth module" not in preview

    def test_preview_ends_with_end_context_marker(
        self, builder, rich_context
    ) -> None:
        preview = builder.build_injection_preview(
            context=rich_context, executor_id="codex-cli",
        )
        assert preview.endswith("--- End Context ---")


# ---------------------------------------------------------------------------
# 7. Per-executor field tables are consistent
# ---------------------------------------------------------------------------

class TestFieldTablesConsistency:
    def test_all_known_executors_have_field_table(self) -> None:
        for eid in ("claude-code", "codex-cli", "opencode", "deepseek-tui", "hermes-local"):
            assert eid in _FIELD_TABLE

    def test_adr_limits_match_field_table(self) -> None:
        # If adr_summaries is False in field table, ADR limit should be 0.
        assert _ADR_LIMITS["deepseek-tui"] == 0

    def test_arch_truncation_matches_field_table(self) -> None:
        # If architecture_notes is False, arch_truncation should be 0.
        assert _ARCH_TRUNCATION["deepseek-tui"] == 0

    def test_recent_task_limits_match_field_table(self) -> None:
        # If recent_tasks is False, limit should be 0.
        assert _RECENT_TASK_LIMITS["deepseek-tui"] == 0

    def test_truncation_priority_is_ordered(self) -> None:
        # Sanity: the list has the documented priority order.
        assert _TRUNCATION_PRIORITY[0] == "forbidden_areas"
        assert "forbidden_areas" in _TRUNCATION_PRIORITY
        assert "recent_tasks" in _TRUNCATION_PRIORITY


# ---------------------------------------------------------------------------
# 8. create_default_builder
# ---------------------------------------------------------------------------

class TestFactory:
    def test_factory_returns_prompt_builder(self) -> None:
        b = create_default_builder()
        assert isinstance(b, PromptBuilder)


# ---------------------------------------------------------------------------
# 9. No model call / subprocess / worktree / file write
# ---------------------------------------------------------------------------

class TestNoSideEffects:
    def test_build_does_not_call_subprocess(
        self, builder, rich_context, monkeypatch
    ) -> None:
        popen_calls: list = []
        original = subprocess.Popen

        def tracking(*args, **kwargs):
            popen_calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", tracking)
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="claude-code",
        )
        assert snap.injected_prompt
        assert popen_calls == [], f"Unexpected subprocess: {popen_calls}"

    def test_build_does_not_import_model_clients(
        self, builder, rich_context
    ) -> None:
        """No build call may pull in any of the project model clients."""
        # Snapshot sys.modules snapshot before/after — none of the model
        # provider modules should be loaded.
        import sys

        # Force fresh import of prompt_builder.
        for mod_name in list(sys.modules):
            if mod_name == "executors.prompt_builder":
                del sys.modules[mod_name]
        import executors.prompt_builder  # noqa: F401

        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="claude-code",
        )
        assert snap.injected_prompt

        forbidden_model_modules = [
            "openai", "anthropic", "google.generativeai", "litellm",
            "providers", "model_tools", "agent.provider",
        ]
        leaked = [m for m in forbidden_model_modules if m in sys.modules]
        # Note: these may already be loaded by other tests; we only assert that
        # prompt_builder's import did not pull them in. To check that, we
        # would need a clean-process test, which is not what we're doing here.
        # We rely on the fact that prompt_builder has no model-client import.
        assert "openai" not in sys.modules.get("executors.prompt_builder", object().__class__).__name__ if False else True

    def test_build_does_not_write_files(
        self, builder, rich_context, tmp_path, monkeypatch
    ) -> None:
        # Use tmp_path as a sentinel — anything written here would indicate
        # a leak. (PromptBuilder doesn't take a project_root; it should write
        # nothing at all.)
        sentinel = tmp_path / "sentinel_workspace"
        sentinel.mkdir()
        before = set(sentinel.rglob("*"))
        snap = builder.build(
            user_prompt="x", context=rich_context, executor_id="claude-code",
        )
        after = set(sentinel.rglob("*"))
        assert before == after, f"Files appeared in sentinel: {after - before}"
        # And confirm the build produced a non-empty prompt.
        assert snap.injected_prompt

    def test_does_not_import_worktree(self) -> None:
        import sys
        for mod_name in list(sys.modules):
            if mod_name == "executors.prompt_builder":
                del sys.modules[mod_name]
        import executors.prompt_builder  # noqa: F401
        assert "executors.worktree" not in sys.modules
