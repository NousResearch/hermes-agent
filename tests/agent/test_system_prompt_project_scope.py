"""Tests for project-scoped memory block injection in build_system_prompt_parts.

Verifies that when ``memory.project_scoping`` is enabled, the memory block
injected into the volatile tier is filtered by the resolved project scope,
and that backward-compat / fail-open paths work.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.system_prompt import build_system_prompt_parts
from tools.memory_tool import MemoryStore


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_agent(**overrides):
    """Minimal agent stub matching the pattern in test_system_prompt.py.

    Must carry all attributes that ``build_system_prompt_parts`` and its
    transitive callers access without guarding.
    """
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _emit_status=lambda *_args, **_kwargs: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _write_memory_entries(path: Path, entries: list[str]) -> None:
    """Write entries separated by the memory-store delimiter ``§``."""
    content = "\n§\n".join(entries)
    path.write_text(content, encoding="utf-8")


def _build(agent):
    """Call build_system_prompt_parts with the usual context-file patches."""
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)


# ── Fixture ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def store_and_agent(monkeypatch, tmp_path):
    """Return (MemoryStore, SimpleNamespace agent) wired to a temp memory dir.

    The store has three entries: an un-tagged global note, a "[project:other]"
    entry, and a "[project:proj]" entry.  The agent has memory + user profile
    enabled.
    """
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)

    store = MemoryStore(memory_char_limit=2000, user_char_limit=500)

    _write_memory_entries(tmp_path / "MEMORY.md", [
        "global note",
        "[project:other] other note",
        "[project:proj] proj note",
    ])
    store.load_from_disk()

    agent = _make_agent(
        _memory_store=store,
        _memory_enabled=True,
        _user_profile_enabled=True,
    )
    return store, agent


# ── Tests ────────────────────────────────────────────────────────────────────


class TestMemoryBlockProjectScope:
    """project-scoped memory block injected via build_system_prompt_parts."""

    def test_scoping_enabled_filters_non_matching(self, store_and_agent, monkeypatch, tmp_path):
        """project_scoping=True + scope='proj' → only global + [project:proj] entries."""
        store, agent = store_and_agent

        # Pin session cwd to a tmp dir with a .git marker so resolve_project_scope
        # returns "proj".
        proj_root = tmp_path / "proj"
        proj_root.mkdir()
        (proj_root / ".git").mkdir()

        monkeypatch.setenv("TERMINAL_CWD", str(proj_root / "sub"))
        (proj_root / "sub").mkdir()

        # Enable project scoping in config
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"memory": {"project_scoping": True}},
        )

        parts = _build(agent)
        volatile = parts.get("volatile", "")

        # Must include global (un-tagged) entries
        assert "global note" in volatile, \
            "global (un-tagged) memory entry must appear when scoping is enabled"
        # Must include matching project entry
        assert "proj note" in volatile, \
            "[project:proj] entry must appear when scope matches"
        # Must NOT include non-matching project entry
        assert "other note" not in volatile, \
            "[project:other] entry must be filtered out with scope='proj'"

    def test_empty_scope_shows_all_entries(self, store_and_agent, monkeypatch):
        """project_scoping=True + resolve_project_scope()=\"\" → all entries unfiltered."""
        store, agent = store_and_agent

        # resolve_project_scope returns "" (e.g. cwd in $HOME or temp root)
        monkeypatch.setattr(
            "agent.runtime_cwd.resolve_project_scope", lambda: ""
        )

        # Enable project scoping in config
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"memory": {"project_scoping": True}},
        )

        parts = _build(agent)
        volatile = parts.get("volatile", "")

        assert "global note" in volatile, \
            "global (un-tagged) entry must appear when scope is empty"
        assert "other note" in volatile, \
            "[project:other] entry must be included when scope is empty (unfiltered)"
        assert "proj note" in volatile, \
            "[project:proj] entry must be included when scope is empty (unfiltered)"

    def test_scoping_disabled_shows_all(self, store_and_agent, monkeypatch):
        """project_scoping=False (default) → all entries unfiltered (backward compat)."""
        store, agent = store_and_agent

        # Config returns no project_scoping (default false)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"memory": {}},  # project_scoping not set → defaults to False
        )

        parts = _build(agent)
        volatile = parts.get("volatile", "")

        assert "global note" in volatile
        assert "other note" in volatile, \
            "[project:other] must be included when scoping is disabled (backward compat)"
        assert "proj note" in volatile

    def test_config_read_error_fails_open(self, store_and_agent, monkeypatch):
        """Config load exception → scope falls back to '' → full unfiltered block."""
        store, agent = store_and_agent

        # load_config_readonly raises
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: (_ for _ in ()).throw(RuntimeError("config unreadable")),
        )

        parts = _build(agent)
        volatile = parts.get("volatile", "")

        assert "global note" in volatile, \
            "global entry must survive config read failure (fail-open)"
        assert "other note" in volatile, \
            "[project:other] entry must survive config read failure (fail-open)"
        assert "proj note" in volatile

    def test_scope_pinned_across_rebuilds(self, store_and_agent, monkeypatch):
        """Scope is pinned on the agent at first build; rebuilds reuse it.

        Build the prompt twice with scoping enabled and resolve_project_scope
        monkeypatched to return different values between calls → the memory
        block is IDENTICAL both times (proves the per-session pin).
        """
        store, agent = store_and_agent

        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"memory": {"project_scoping": True}},
        )

        calls = {"n": 0}

        def oscillating_scope():
            calls["n"] += 1
            # Return "proj" on first call, "other" on second, "third" on third
            return ["proj", "other", "third"][min(calls["n"] - 1, 2)]

        monkeypatch.setattr(
            "agent.runtime_cwd.resolve_project_scope", oscillating_scope,
        )

        # First build — should use oscillating_scope (returns "proj")
        parts1 = _build(agent)
        volatile1 = parts1.get("volatile", "")

        # Second build — should reuse pinned value, NOT oscillating_scope
        parts2 = _build(agent)
        volatile2 = parts2.get("volatile", "")

        assert volatile1 == volatile2, (
            "Memory block changed between rebuilds — scope pin is not working"
        )
        assert "proj note" in volatile1, (
            "First build should contain [project:proj] entry (scope='proj')"
        )
        assert "other note" not in volatile1, (
            "First build should filter out [project:other] entry (scope='proj')"
        )