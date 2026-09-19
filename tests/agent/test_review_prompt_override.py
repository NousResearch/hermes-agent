"""Behavior tests for ``agent.review_prompts`` config overrides.

Asserts the *resolution chain* (programmatic agent attr > inline config > ``*_file`` config >
module-level constant), not snapshot text. Behavior contracts:

  - unset config                  -> module-level constant (byte-identical default)
  - inline string set             -> that string flows to the review fork
  - inline ""                     -> explicit OFF: resolver -> None, spawn skips (no thread)
  - ``<kind>_file`` set           -> file content replaces the prompt (read once per spawn)
  - file missing/empty/bad-type/oversized/non-regular -> None + actionable warning (skip,
    never silently run the default against user intent); explicit /refine falls back to default
  - programmatic agent attr        -> wins over config; the AIAgent-CLASS default (imported from
    this module) does NOT shadow config (the #57447 review finding)
  - relative ``*_file`` paths      -> resolved against the OWNING profile home, never CWD
  - two profile homes A->B->A       -> each resolves its own config/files (multiplex isolation)
  - the enforced tool-restriction suffix and dispatch whitelist are appended unchanged

The override only changes the user message the background review fork receives. The
main-conversation system prompt and prompt cache are never touched (per AGENTS.md
"Per-conversation prompt caching is sacred").

Original resolver + inline-config tests by @jneeee (#57447); file-backed path, skip
semantics, profile-home resolution and production-shape coverage added in the salvage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from agent.background_review import (
    _MEMORY_REVIEW_PROMPT,
    _SKILL_REVIEW_PROMPT,
    _COMBINED_REVIEW_PROMPT,
    _resolve_review_prompt,
    _REVIEW_PROMPT_BY_KIND,
    _REVIEW_PROMPT_FILE_MAX_BYTES,
    spawn_background_review_thread,
)


_SENTINEL_UNSET = object()

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows")


class _FakeAgent:
    """Minimal stand-in for AIAgent — just enough surface for the resolver."""

    def __init__(self, **overrides: str) -> None:
        for k, v in overrides.items():
            setattr(self, k, v)


class _FakeSessionDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = str(db_path)


def _write_home_config(home: Path, review_prompts: Dict[str, Any]) -> None:
    """Write a minimal config.yaml with an agent.review_prompts block into ``home``."""
    home.mkdir(parents=True, exist_ok=True)
    lines = ["agent:", "  review_prompts:"]
    for key, value in review_prompts.items():
        if value is None or value is _SENTINEL_UNSET:
            continue
        if not isinstance(value, str):
            lines.append(f"    {key}: {value}")  # non-string scalars (error-path tests)
        elif value == "":
            lines.append(f'    {key}: ""')
        elif value.strip() == "" or value != value.strip() or ": " in value or "\n" in value:
            # quote whitespace-only / padded values (YAML trims unquoted spaces);
            # colon/newline-bearing values use a single-quoted scalar
            lines.append(f'    {key}: {value!r}' if "\n" not in value else f"    {key}: |")
            if "\n" in value:
                for ln in value.splitlines():
                    lines.append(f"      {ln}")
        else:
            lines.append(f"    {key}: {value}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clear_config_caches() -> None:
    from hermes_cli import config as config_module
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()


@pytest.fixture
def home_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME; returns (home, activate) where activate(review_prompts) writes
    config and clears the config caches so the next read sees it."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def activate(review_prompts: Dict[str, Any]) -> Path:
        _write_home_config(home, review_prompts)
        _clear_config_caches()
        return home

    activate({})  # start with a clean config (no review_prompts overrides)
    return home, activate


# ---------------------------------------------------------------------------
# 1. Defaults — no change in behaviour when override is unset.
# ---------------------------------------------------------------------------

class TestDefaultsPreserved:
    def test_memory_default_is_module_constant(self, home_env):
        assert _resolve_review_prompt(_FakeAgent(), "memory") == _MEMORY_REVIEW_PROMPT

    def test_skill_default_is_module_constant(self, home_env):
        assert _resolve_review_prompt(_FakeAgent(), "skill") == _SKILL_REVIEW_PROMPT

    def test_combined_default_is_module_constant(self, home_env):
        assert _resolve_review_prompt(_FakeAgent(), "combined") == _COMBINED_REVIEW_PROMPT

    def test_default_config_has_review_prompts_block(self):
        """The DEFAULT_CONFIG dict exposes the six slots as a contract."""
        from hermes_cli.config import DEFAULT_CONFIG
        rp = DEFAULT_CONFIG["agent"]["review_prompts"]
        assert set(rp.keys()) == {
            "memory", "memory_file", "skill", "skill_file", "combined", "combined_file"}
        # Inline slots default to None (unset); file slots to "" (unset).
        assert rp["memory"] is None and rp["skill"] is None and rp["combined"] is None
        assert rp["memory_file"] == "" and rp["skill_file"] == "" and rp["combined_file"] == ""

    def test_config_set_accepts_new_keys(self, home_env):
        """``hermes config set agent.review_prompts.*`` must not be refused as unknown —
        runtime-recognizes-but-config-set-rejects drift is the known failure mode."""
        from hermes_cli.config import _validate_config_key
        for key in ("agent.review_prompts.memory", "agent.review_prompts.memory_file",
                    "agent.review_prompts.skill", "agent.review_prompts.skill_file",
                    "agent.review_prompts.combined", "agent.review_prompts.combined_file"):
            is_known, suggestion = _validate_config_key(key)
            assert is_known, f"{key} rejected (suggestion={suggestion})"


# ---------------------------------------------------------------------------
# 2. Inline config override flows through.
# ---------------------------------------------------------------------------

class TestInlineOverride:
    def test_memory_override_replaces_module_constant(self, home_env):
        _home, activate = home_env
        activate({"memory": "MEMORY OVERRIDE: only save cross-session persona facts."})
        out = _resolve_review_prompt(_FakeAgent(), "memory")
        assert out == "MEMORY OVERRIDE: only save cross-session persona facts."
        assert out != _MEMORY_REVIEW_PROMPT

    def test_skill_override_replaces_module_constant(self, home_env):
        _home, activate = home_env
        activate({"skill": "SKILL OVERRIDE: patch loaded skill first, then references."})
        assert _resolve_review_prompt(_FakeAgent(), "skill") == (
            "SKILL OVERRIDE: patch loaded skill first, then references.")

    def test_combined_override_replaces_module_constant(self, home_env):
        _home, activate = home_env
        activate({"combined": "COMBINED OVERRIDE: nothing qualifies? just say so and exit."})
        assert _resolve_review_prompt(_FakeAgent(), "combined") == (
            "COMBINED OVERRIDE: nothing qualifies? just say so and exit.")

    def test_per_kind_overrides_are_independent(self, home_env):
        _home, activate = home_env
        activate({"memory": "ONLY memory facts."})
        assert _resolve_review_prompt(_FakeAgent(), "memory") == "ONLY memory facts."
        assert _resolve_review_prompt(_FakeAgent(), "skill") == _SKILL_REVIEW_PROMPT
        assert _resolve_review_prompt(_FakeAgent(), "combined") == _COMBINED_REVIEW_PROMPT


# ---------------------------------------------------------------------------
# 3. Empty-string inline override = explicit "off".
# ---------------------------------------------------------------------------

class TestExplicitDisable:
    def test_empty_memory_skips_review(self, home_env):
        _home, activate = home_env
        activate({"memory": ""})
        assert _resolve_review_prompt(_FakeAgent(), "memory") is None

    def test_empty_skill_skips_review(self, home_env):
        _home, activate = home_env
        activate({"skill": ""})
        assert _resolve_review_prompt(_FakeAgent(), "skill") is None

    def test_empty_combined_skips_review(self, home_env):
        _home, activate = home_env
        activate({"combined": ""})
        assert _resolve_review_prompt(_FakeAgent(), "combined") is None

    def test_explicit_refine_uses_default_not_skip(self, home_env):
        """``/refine`` (explicit) is never starved by the disable sentinel — mirrors
        ``auxiliary.background_review.enabled`` (auto off, /refine on)."""
        _home, activate = home_env
        activate({"memory": ""})
        assert _resolve_review_prompt(
            _FakeAgent(), "memory", explicit=True) == _MEMORY_REVIEW_PROMPT

    def test_whitespace_only_inline_is_an_error_not_a_disable(self, home_env, caplog):
        """Mirror of the whitespace-only FILE case: an inline value of spaces is a broken
        override, not the "" disable — error-skip with a warning (MAJOR-1 review finding:
        it previously slipped past the sentinel and spawned a prompt-less review)."""
        _home, activate = home_env
        activate({"memory": "   "})
        with caplog.at_level("WARNING"):
            out = _resolve_review_prompt(_FakeAgent(), "memory")
        assert out is None
        assert any("whitespace-only" in r.message for r in caplog.records)

    def test_whitespace_only_inline_explicit_falls_back_to_default(self, home_env):
        _home, activate = home_env
        activate({"memory": "   "})
        assert _resolve_review_prompt(
            _FakeAgent(), "memory", explicit=True) == _MEMORY_REVIEW_PROMPT

    def test_empty_path_string_is_unset_not_error(self, home_env):
        """DEFAULT_CONFIG seeds "" for *_file slots; an empty path string means unset."""
        _home, activate = home_env
        activate({"memory_file": ""})
        assert _resolve_review_prompt(_FakeAgent(), "memory") == _MEMORY_REVIEW_PROMPT

    def test_spawn_returns_none_none_on_disable(self, home_env):
        _home, activate = home_env
        activate({"memory": ""})
        target, prompt = spawn_background_review_thread(
            _FakeAgent(), [], review_memory=True, review_skills=False)
        assert target is None and prompt is None


# ---------------------------------------------------------------------------
# 4. File-backed overrides.
# ---------------------------------------------------------------------------

class TestFileOverride:
    def test_relative_path_resolves_against_profile_home(self, home_env):
        _home, activate = home_env
        (_home / "prompts").mkdir()
        (_home / "prompts" / "memory.md").write_text(
            "FILE PROMPT: durable facts only.", encoding="utf-8")
        activate({"memory_file": "prompts/memory.md"})
        assert _resolve_review_prompt(_FakeAgent(), "memory") == "FILE PROMPT: durable facts only."

    def test_absolute_path_honored(self, home_env, tmp_path):
        _home, activate = home_env
        external = tmp_path / "external-skill-prompt.txt"
        external.write_text("ABS: skill review via absolute path.", encoding="utf-8")
        activate({"skill_file": str(external)})
        assert _resolve_review_prompt(_FakeAgent(), "skill") == "ABS: skill review via absolute path."

    def test_inline_wins_over_file(self, home_env):
        _home, activate = home_env
        (_home / "p.txt").write_text("FROM FILE", encoding="utf-8")
        activate({"skill": "FROM INLINE", "skill_file": "p.txt"})
        assert _resolve_review_prompt(_FakeAgent(), "skill") == "FROM INLINE"

    def test_content_is_stripped_only_at_edges(self, home_env):
        _home, activate = home_env
        (_home / "p.md").write_text(
            "\n\n  Keep internal   spacing.  \n\t\n", encoding="utf-8")
        activate({"memory_file": "p.md"})
        assert _resolve_review_prompt(_FakeAgent(), "memory") == "Keep internal   spacing."

    def test_edit_between_reviews_is_seen_by_next_spawn(self, home_env):
        """One stable per-review snapshot: each spawn re-reads the file; an edit lands
        on the NEXT review, and a single review never sees mixed content."""
        _home, activate = home_env
        p = _home / "p.md"
        p.write_text("v1 prompt", encoding="utf-8")
        activate({"memory_file": "p.md"})
        _t1, prompt1 = spawn_background_review_thread(
            _FakeAgent(), [], review_memory=True, review_skills=False)
        p.write_text("v2 prompt", encoding="utf-8")
        _t2, prompt2 = spawn_background_review_thread(
            _FakeAgent(), [], review_memory=True, review_skills=False)
        assert prompt1 == "v1 prompt"
        assert prompt2 == "v2 prompt"

    def test_agent_session_db_derives_home_for_file(self, home_env, tmp_path):
        """An agent whose session DB lives under profile B resolves B's file even when the
        ambient HERMES_HOME points elsewhere (idle-queue thread that lost the ContextVar)."""
        home_a, activate = home_env
        home_b = tmp_path / "profile-b-home"
        (home_b / "prompts").mkdir(parents=True)
        (home_b / "prompts" / "memory.md").write_text("B FILE PROMPT", encoding="utf-8")
        _write_home_config(home_b, {"memory_file": "prompts/memory.md"})
        activate({})  # home A: no override

        agent = _FakeAgent()
        agent._session_db = _FakeSessionDB(home_b / "state.db")
        _target, prompt = spawn_background_review_thread(
            agent, [], review_memory=True, review_skills=False)
        assert prompt == "B FILE PROMPT"

    def test_agent_session_db_derives_home_for_config_too(self, home_env, tmp_path):
        """Inline overrides also resolve from the agent's OWN home via the spawn binding."""
        home_a, activate = home_env
        home_b = tmp_path / "profile-b-home"
        _write_home_config(home_b, {"skill": "B INLINE PROMPT"})
        activate({})  # home A: no override

        agent = _FakeAgent()
        agent._session_db = _FakeSessionDB(home_b / "state.db")
        _target, prompt = spawn_background_review_thread(
            agent, [], review_memory=False, review_skills=True)
        assert prompt == "B INLINE PROMPT"


# ---------------------------------------------------------------------------
# 5. Invalid explicit overrides: actionable diagnostics + skip, never silent default.
# ---------------------------------------------------------------------------

class TestInvalidOverrides:
    def test_missing_file_skips_with_warning(self, home_env, caplog):
        _home, activate = home_env
        activate({"memory_file": "prompts/does-not-exist.md"})
        with caplog.at_level("WARNING"):
            out = _resolve_review_prompt(_FakeAgent(), "memory")
        assert out is None
        assert any("does-not-exist.md" in r.message for r in caplog.records)

    def test_empty_file_is_an_error_not_a_disable(self, home_env, caplog):
        _home, activate = home_env
        (_home / "empty.md").write_text("", encoding="utf-8")
        activate({"skill_file": "empty.md"})
        with caplog.at_level("WARNING"):
            out = _resolve_review_prompt(_FakeAgent(), "skill")
        assert out is None  # NOT the default: user intent was an override
        assert any("empty" in r.message.lower() and '""' in r.message for r in caplog.records)

    def test_whitespace_only_file_is_an_error(self, home_env, caplog):
        _home, activate = home_env
        (_home / "ws.md").write_text("   \n\t\n", encoding="utf-8")
        activate({"skill_file": "ws.md"})
        with caplog.at_level("WARNING"):
            out = _resolve_review_prompt(_FakeAgent(), "skill")
        assert out is None

    def test_directory_path_is_rejected(self, home_env, caplog):
        _home, activate = home_env
        (_home / "prompts").mkdir()
        activate({"memory_file": "prompts"})  # a directory
        with caplog.at_level("WARNING"):
            out = _resolve_review_prompt(_FakeAgent(), "memory")
        assert out is None
        assert any("not a regular file" in r.message for r in caplog.records)

    def test_non_utf8_file_is_rejected(self, home_env, caplog):
        _home, activate = home_env
        (_home / "bad.bin").write_bytes(b"\xff\xfe\x00binary")
        activate({"skill_file": "bad.bin"})
        with caplog.at_level("WARNING"):
            out = _resolve_review_prompt(_FakeAgent(), "skill")
        assert out is None

    def test_oversized_file_is_rejected(self, home_env, caplog):
        _home, activate = home_env
        (_home / "huge.md").write_text("x" * (_REVIEW_PROMPT_FILE_MAX_BYTES + 1), encoding="utf-8")
        activate({"memory_file": "huge.md"})
        with caplog.at_level("WARNING"):
            out = _resolve_review_prompt(_FakeAgent(), "memory")
        assert out is None
        assert any("bytes" in r.message for r in caplog.records)

    @_POSIX_ONLY
    def test_unreadable_file_is_rejected(self, home_env, caplog):
        _home, activate = home_env
        secret = _home / "noread.md"
        secret.write_text("chmod me away", encoding="utf-8")
        secret.chmod(0o000)
        activate({"skill_file": "noread.md"})
        try:
            with caplog.at_level("WARNING"):
                out = _resolve_review_prompt(_FakeAgent(), "skill")
            assert out is None
        finally:
            secret.chmod(0o644)

    def test_non_string_inline_value_is_rejected(self, home_env, caplog):
        """A YAML mapping under the inline slot is a config error — skip loudly."""
        _home, activate = home_env
        (_home / "config.yaml").write_text(
            "agent:\n  review_prompts:\n    memory:\n      nested: true\n", encoding="utf-8")
        _clear_config_caches()
        with caplog.at_level("WARNING"):
            out = _resolve_review_prompt(_FakeAgent(), "memory")
        assert out is None
        assert any("expected a string" in r.message for r in caplog.records)

    def test_explicit_refine_falls_back_to_default_on_invalid_file(self, home_env):
        _home, activate = home_env
        activate({"memory_file": "prompts/does-not-exist.md"})
        assert _resolve_review_prompt(
            _FakeAgent(), "memory", explicit=True) == _MEMORY_REVIEW_PROMPT


# ---------------------------------------------------------------------------
# 6. Programmatic overrides and the production AIAgent shape.
# ---------------------------------------------------------------------------

class TestProgrammaticOverride:
    def test_agent_attr_wins_over_config(self, home_env):
        _home, activate = home_env
        activate({"memory": "FROM CONFIG"})
        agent = _FakeAgent()
        agent._MEMORY_REVIEW_PROMPT = "FROM AGENT ATTR"
        assert _resolve_review_prompt(agent, "memory") == "FROM AGENT ATTR"

    def test_inherited_class_default_does_not_shadow_config(self, home_env):
        """The #57447 review finding: run_agent imports the module constants onto the
        AIAgent class namespace, so getattr(agent, ...) ALWAYS finds a truthy value.
        The resolver must treat an attr equal to the module constant as 'no override'
        so config still applies for ordinary AIAgent instances."""
        _home, activate = home_env
        activate({"memory": "CONFIG MUST WIN"})
        # Simulate the real AIAgent shape: class attribute imported from the module.
        class _AgentWithClassDefaults:
            _MEMORY_REVIEW_PROMPT = _MEMORY_REVIEW_PROMPT
            _SKILL_REVIEW_PROMPT = _SKILL_REVIEW_PROMPT
            _COMBINED_REVIEW_PROMPT = _COMBINED_REVIEW_PROMPT
        assert _resolve_review_prompt(_AgentWithClassDefaults(), "memory") == "CONFIG MUST WIN"

    def test_unknown_kind_returns_memory_default_not_crash(self, home_env):
        out = _resolve_review_prompt(_FakeAgent(), "bogus")
        assert isinstance(out, str) and out == _MEMORY_REVIEW_PROMPT


# ---------------------------------------------------------------------------
# 7. Robustness — bad config layouts must not crash the fork.
# ---------------------------------------------------------------------------

class TestBadConfigFallsBack:
    def test_non_dict_review_prompts_falls_through(self, home_env):
        """If a user pastes a string into agent.review_prompts by mistake, we must not
        crash — fall through to the module constant (config unreadable => default)."""
        _home, _activate = home_env
        (_home / "config.yaml").write_text(
            "agent:\n  review_prompts: oops i pasted a string\n", encoding="utf-8")
        _clear_config_caches()
        assert _resolve_review_prompt(_FakeAgent(), "memory") == _MEMORY_REVIEW_PROMPT
        assert _resolve_review_prompt(_FakeAgent(), "skill") == _SKILL_REVIEW_PROMPT
        assert _resolve_review_prompt(_FakeAgent(), "combined") == _COMBINED_REVIEW_PROMPT


# ---------------------------------------------------------------------------
# 8. E2E: real config -> resolution -> review-fork user message, through the
#    run_agent spawn path, with whitelist/suffix integrity.
# ---------------------------------------------------------------------------

class _SyncThread:
    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _make_agent_stub(agent_cls):
    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = "STABLE SYSTEM PROMPT"
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent.enabled_toolsets = ["memory", "skills", "terminal"]
    agent.disabled_toolsets = []
    return agent


class TestEndToEndSpawnPath:
    def test_custom_prompt_flows_to_fork_with_suffix_and_whitelist_intact(self, home_env):
        """Real config -> resolver -> fork user message: the custom prompt REPLACES the base
        prompt, the enforced tool-restriction suffix is STILL appended, the dispatch whitelist
        is unchanged (no new tools), and the foreground system prompt is untouched."""
        _home, activate = home_env
        (_home / "prompts").mkdir()
        (_home / "prompts" / "memory.md").write_text(
            "CUSTOM: evidence-backed lessons only; a no-change outcome is valid.", encoding="utf-8")
        activate({"memory_file": "prompts/memory.md"})

        import run_agent
        from hermes_cli import plugins as _plugins

        captured = {}

        def _capture_whitelist(whitelist, deny_msg_fmt=None):
            captured["whitelist"] = set(whitelist)

        def _capture_run_conversation(self, *, user_message, **kwargs):
            captured["user_message"] = user_message
            return {"final_response": "Nothing to save."}

        agent = _make_agent_stub(run_agent.AIAgent)

        def _no_init(self, *args, **kwargs):
            return None

        with patch.object(run_agent.AIAgent, "__init__", _no_init), \
             patch.object(run_agent.AIAgent, "run_conversation", _capture_run_conversation), \
             patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
             patch.object(run_agent.AIAgent, "close", lambda self: None), \
             patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
             patch("threading.Thread", _SyncThread):
            agent._spawn_background_review(
                messages_snapshot=[], review_memory=True, review_skills=False)

        msg = captured["user_message"]
        assert msg.startswith("CUSTOM: evidence-backed lessons only")
        assert "You can only call" in msg  # enforced suffix still appended
        assert "terminal" not in captured["whitelist"]
        assert agent._cached_system_prompt == "STABLE SYSTEM PROMPT"  # cache untouched

    def test_disable_skips_thread_and_releases_run_token(self, home_env):
        """Caller contract: (None, None) -> no thread, no fork build, run token released."""
        _home, activate = home_env
        activate({"memory": ""})

        import run_agent

        agent = _make_agent_stub(run_agent.AIAgent)
        built = {}

        def _capture_init(self, *args, **kwargs):
            built["yes"] = True
            return None

        with patch.object(run_agent.AIAgent, "__init__", _capture_init), \
             patch("threading.Thread", _SyncThread):
            agent._spawn_background_review(
                messages_snapshot=[], review_memory=True, review_skills=False)

        assert "yes" not in built  # fork never constructed
        assert getattr(agent, "_background_review_run", None) is None  # token released


# ---------------------------------------------------------------------------
# 9. Two profile homes A -> B -> A (multiplex isolation).
# ---------------------------------------------------------------------------

class TestProfileIsolation:
    def _make_home(self, base: Path, name: str, prompt_text: str) -> Path:
        home = base / name
        (home / "prompts").mkdir(parents=True)
        (home / "prompts" / "memory.md").write_text(prompt_text, encoding="utf-8")
        _write_home_config(home, {"memory_file": "prompts/memory.md"})
        return home

    def test_multiplex_a_b_a_resolves_own_home(self, tmp_path, monkeypatch):
        """A multiplex host serving profiles A and B (then A again) must resolve each
        agent's prompt from ITS home — the ContextVar-bound profile scope, exactly as
        the gateway binds per turn."""
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        home_a = self._make_home(tmp_path, "home-a", "PROMPT FROM A")
        home_b = self._make_home(tmp_path, "home-b", "PROMPT FROM B")

        def _resolve_under(home: Path):
            _clear_config_caches()
            token = set_hermes_home_override(str(home))
            try:
                return _resolve_review_prompt(_FakeAgent(), "memory")
            finally:
                reset_hermes_home_override(token)

        assert _resolve_under(home_a) == "PROMPT FROM A"
        assert _resolve_under(home_b) == "PROMPT FROM B"
        assert _resolve_under(home_a) == "PROMPT FROM A"

    def test_spawn_binds_agent_home_without_ambient_override(self, tmp_path, monkeypatch):
        """Without a ContextVar binding (deferred/idle-queue dispatch), the spawn pins the
        agent's session-db home around resolution, so ambient HERMES_HOME cannot leak in."""
        home_a = self._make_home(tmp_path, "home-a", "PROMPT FROM A")
        home_b = self._make_home(tmp_path, "home-b", "PROMPT FROM B")
        monkeypatch.setenv("HERMES_HOME", str(home_a))  # ambient = A (launch profile)

        agent = _FakeAgent()
        agent._session_db = _FakeSessionDB(home_b / "state.db")  # owning = B

        _clear_config_caches()
        _target, prompt = spawn_background_review_thread(
            agent, [], review_memory=True, review_skills=False)
        assert prompt == "PROMPT FROM B"


# ---------------------------------------------------------------------------
# 10. Mapping table — every declared kind has a corresponding prompt constant.
# ---------------------------------------------------------------------------

def test_review_prompt_mapping_covers_all_kinds():
    """The mapping table is the single source of truth for kind->attr.

    Adding a new review kind must require an edit here; if you add a kind but
    forget the mapping, ``_resolve_review_prompt`` quietly falls back to
    _MEMORY_REVIEW_PROMPT, which is wrong-and-silent. Lock it down.
    """
    expected_attrs = {"_MEMORY_REVIEW_PROMPT", "_SKILL_REVIEW_PROMPT", "_COMBINED_REVIEW_PROMPT"}
    actual_attrs = set(_REVIEW_PROMPT_BY_KIND.values())
    assert actual_attrs == expected_attrs
