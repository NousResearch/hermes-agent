"""Project-scoped memory in the system prompt (issue #33638).

Covers the opt-in ``memory.project_scoping`` filter end to end: which entries reach the
volatile memory block, the per-session scope pin (cwd-keyed, dropped at a session
boundary), and the other surfaces that re-derive the same block.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.system_prompt import build_system_prompt_parts, pinned_project_scope
from tools.memory_tool_store import MemoryStore


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_agent(**overrides):
    """Agent stub carrying what build_system_prompt_parts reads without guarding.

    ``skip_context_files`` + ``load_soul_identity=False`` keep the build off the real
    context-file loaders; an empty toolset keeps the coding-posture blocks out of it.
    """
    base = dict(
        _cached_system_prompt=None,
        _cached_system_prompt_static=None,
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        platform="",
        model="",
        provider="",
        session_id="s1",
        pass_session_id=False,
        _memory_store=None,
        _memory_manager=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _execution_guidance=False,
        _environment_probe=False,
        _bot_mode_protocol=False,
        _kanban_worker_guidance="",
        _emit_status=lambda *a, **k: None,
        _transition_context_engine_session=lambda **kw: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _project(root: Path) -> Path:
    """A project root with a .git marker plus a subdir to use as the cwd."""
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    sub = root / "src"
    sub.mkdir()
    return sub


def _enable_scoping(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda: {"memory": {"project_scoping": True}})


def _volatile(agent, cwd: Path) -> str:
    """Build the prompt with the session cwd pinned to *cwd*, return the volatile tier."""
    with patch("agent.system_prompt.resolve_context_cwd", return_value=cwd):
        return build_system_prompt_parts(agent).get("volatile", "")


@pytest.fixture()
def store_and_agent(tmp_path, monkeypatch):
    """MemoryStore (3 entries, one per scope) on a temp memory dir + an agent using it."""
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    store = MemoryStore(memory_char_limit=2000, user_char_limit=500)
    (tmp_path / "MEMORY.md").write_text(
        "\n§\n".join(["global note", "[project:other] other note", "[project:proj] proj note"]),
        encoding="utf-8",
    )
    store.load_from_disk()
    agent = _make_agent(_memory_store=store, _memory_enabled=True, _user_profile_enabled=True)
    return store, agent


# ── Injection ────────────────────────────────────────────────────────────────


class TestMemoryBlockProjectScope:
    def test_matching_scope_keeps_global_and_own_entries(self, store_and_agent, tmp_path, monkeypatch):
        """scoping on + cwd in project 'proj' → global + [project:proj], never [project:other]."""
        _store, agent = store_and_agent
        _enable_scoping(monkeypatch)

        volatile = _volatile(agent, _project(tmp_path / "proj"))

        assert "global note" in volatile, "untagged entries must survive every scope"
        assert "proj note" in volatile, "the matching [project:proj] entry must survive"
        assert "other note" not in volatile, "[project:other] must be filtered out of another project"

    def test_empty_scope_shows_everything(self, store_and_agent, tmp_path, monkeypatch):
        """Scoping on but no project resolved → unfiltered block (fail-open)."""
        _store, agent = store_and_agent
        _enable_scoping(monkeypatch)
        monkeypatch.setattr("agent.runtime_cwd.resolve_project_scope", lambda _cwd=None: "")

        volatile = _volatile(agent, tmp_path)

        assert "global note" in volatile and "other note" in volatile and "proj note" in volatile

    def test_scoping_disabled_shows_everything(self, store_and_agent, tmp_path, monkeypatch):
        """Default (flag absent) is byte-for-byte the pre-feature behavior."""
        _store, agent = store_and_agent
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"memory": {}})

        volatile = _volatile(agent, _project(tmp_path / "proj"))

        assert "global note" in volatile and "other note" in volatile and "proj note" in volatile

    def test_config_read_failure_fails_open(self, store_and_agent, tmp_path, monkeypatch):
        """An unreadable config must not filter anything out."""
        _store, agent = store_and_agent
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: (_ for _ in ()).throw(RuntimeError("config unreadable")),
        )

        volatile = _volatile(agent, _project(tmp_path / "proj"))

        assert "global note" in volatile and "other note" in volatile

    def test_scope_resolution_failure_fails_open(self, store_and_agent, tmp_path, monkeypatch):
        """A raising scope resolver must not filter anything out."""
        _store, agent = store_and_agent
        _enable_scoping(monkeypatch)
        monkeypatch.setattr(
            "agent.runtime_cwd.resolve_project_scope",
            lambda _cwd=None: (_ for _ in ()).throw(OSError("no such dir")),
        )

        volatile = _volatile(agent, _project(tmp_path / "proj"))

        assert "global note" in volatile and "other note" in volatile


# ── The per-session pin ──────────────────────────────────────────────────────


class TestProjectScopePin:
    def test_pin_survives_a_rebuild_with_the_same_cwd(self, store_and_agent, tmp_path, monkeypatch):
        """Compaction rebuilds replay the pinned scope instead of re-resolving it."""
        _store, agent = store_and_agent
        _enable_scoping(monkeypatch)
        cwd = _project(tmp_path / "proj")

        first = _volatile(agent, cwd)
        # A later build would resolve to a DIFFERENT project; the pin must win.
        monkeypatch.setattr("agent.runtime_cwd.resolve_project_scope", lambda _cwd=None: "other")
        second = _volatile(agent, cwd)

        assert first == second, "same session + same cwd must render byte-identical blocks"
        assert "other note" not in second

    def test_cwd_switch_resolves_a_new_scope(self, store_and_agent, tmp_path, monkeypatch):
        """One gateway serves many cwds: a moved cwd must re-resolve, not replay."""
        _store, agent = store_and_agent
        _enable_scoping(monkeypatch)

        in_proj = _volatile(agent, _project(tmp_path / "proj"))
        in_other = _volatile(agent, _project(tmp_path / "other"))

        assert "proj note" in in_proj and "other note" not in in_proj
        assert "other note" in in_other and "proj note" not in in_other

    def test_session_boundary_drops_the_pin(self, store_and_agent, tmp_path, monkeypatch):
        """A /new on the same agent re-resolves for its own session-start cwd."""
        from run_agent import AIAgent

        _store, agent = store_and_agent
        _enable_scoping(monkeypatch)

        in_proj = _volatile(agent, _project(tmp_path / "proj"))
        assert "proj note" in in_proj

        AIAgent.reset_session_state(agent)
        in_other = _volatile(agent, _project(tmp_path / "other"))

        assert "other note" in in_other, "the new session must resolve its own scope"
        assert "proj note" not in in_other

    def test_disabled_flag_returns_empty_scope(self, store_and_agent, monkeypatch):
        _store, agent = store_and_agent
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"memory": {}})

        assert pinned_project_scope(agent) == ""


# ── Header honesty + the surfaces that re-derive the block ────────────────────


class TestScopedBlockConsumers:
    def test_context_breakdown_reports_the_injected_block(self, store_and_agent, tmp_path, monkeypatch):
        """`/context` must attribute the block that is actually in the prompt.

        It re-derives the block and subtracts it from the prompt text, so a
        scope-mismatched derivation leaves the memory text counted twice and reports a
        Memory size that never shipped.
        """
        from agent.context_breakdown import _memory_blocks

        _store, agent = store_and_agent
        _enable_scoping(monkeypatch)
        cwd = _project(tmp_path / "proj")

        volatile = _volatile(agent, cwd)
        with patch("agent.system_prompt.resolve_context_cwd", return_value=cwd):
            reported, _user_block = _memory_blocks(agent)

        assert reported, "the breakdown must find the memory block"
        assert reported in volatile, "the reported block must be the one in the prompt"
        assert "· project: proj" in reported
