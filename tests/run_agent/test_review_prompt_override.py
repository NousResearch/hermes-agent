"""Behavior tests for ``agent.review_prompts`` config override.

Asserts the *resolution chain* (agent instance attribute > config.yaml >
module-level constant), not snapshot text. These are *behavior contracts*:

  - None / missing config block       → module-level constant (no behavior change)
  - config override set              → that override string flows through
  - config override = ""             → empty string is preserved (explicit "off")
  - agent instance attr set          → wins over config override
  - unknown kind                     → caller bug does not crash the resolve

The override only changes the user message the background review fork
receives. The main-conversation system prompt and prompt cache are never
touched (per AGENTS.md "Per-conversation prompt caching is sacred").
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agent.background_review import (
    _MEMORY_REVIEW_PROMPT,
    _SKILL_REVIEW_PROMPT,
    _COMBINED_REVIEW_PROMPT,
    _resolve_review_prompt,
    _REVIEW_PROMPT_BY_KIND,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeAgent:
    """Minimal stand-in for AIAgent — just enough surface for the resolver."""

    def __init__(self, **overrides: str) -> None:
        for k, v in overrides.items():
            setattr(self, k, v)


def _set_config_review_prompts(tmp_config, **values):
    """Return a context manager that, on enter, calls ``tmp_config(dict)``
    with an existing real config merged with our review_prompts overrides.
    On exit, monkeypatch (set up by pytest's ``monkeypatch`` fixture inside
    ``tmp_config``) is automatically undone by pytest.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        from hermes_cli.config import load_config
        real = load_config()
        merged = dict(real)
        agent_cfg = dict(merged.get("agent", {}) or {})
        rp = dict(agent_cfg.get("review_prompts", {}) or {})
        for k, v in values.items():
            if v is _SENTINEL_UNSET:
                rp.pop(k, None)
            elif v is None:
                # Explicit None in user config — store as None so the resolver
                # sees "this slot is set but to default". (The resolver
                # treats None the same as missing, so behaviour is preserved.)
                rp[k] = None
            else:
                rp[k] = v
        agent_cfg["review_prompts"] = rp
        merged["agent"] = agent_cfg
        tmp_config(merged)
        yield merged

    return _ctx()


_SENTINEL_UNSET = object()


@pytest.fixture
def tmp_config(monkeypatch):
    """Force ``hermes_cli.config.load_config`` to return a chosen dict.

    Patches the *attribute* ``load_config`` on the module that
    ``_resolve_review_prompt`` lazy-imports. The lazy-import pattern
    (``from hermes_cli.config import load_config`` inside the function)
    means we patch the symbol *on that module*, not the local binding.
    """

    def _apply(value: Dict[str, Any]) -> None:
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: value)

    yield _apply


# ---------------------------------------------------------------------------
# 1. Defaults — no change in behaviour when override is unset.
# ---------------------------------------------------------------------------

class TestDefaultsPreserved:
    """When ``agent.review_prompts`` is None / missing, the resolver must
    return the verbatim module-level constant for each kind. This is the
    regression guard for existing users."""

    def test_memory_default_is_module_constant(self, tmp_config):
        # Block any value from reaching the resolver.
        with _set_config_review_prompts(tmp_config):
            assert _resolve_review_prompt(_FakeAgent(), "memory") == _MEMORY_REVIEW_PROMPT

    def test_skill_default_is_module_constant(self, tmp_config):
        with _set_config_review_prompts(tmp_config):
            assert _resolve_review_prompt(_FakeAgent(), "skill") == _SKILL_REVIEW_PROMPT

    def test_combined_default_is_module_constant(self, tmp_config):
        with _set_config_review_prompts(tmp_config):
            assert _resolve_review_prompt(_FakeAgent(), "combined") == _COMBINED_REVIEW_PROMPT

    def test_default_config_has_review_prompts_block(self):
        """The DEFAULT_CONFIG dict exposes the three slots as a contract."""
        from hermes_cli.config import DEFAULT_CONFIG
        rp = DEFAULT_CONFIG["agent"]["review_prompts"]
        assert set(rp.keys()) == {"memory", "skill", "combined"}
        # All three are None by default (unset).
        assert rp["memory"] is None
        assert rp["skill"] is None
        assert rp["combined"] is None


# ---------------------------------------------------------------------------
# 2. config.yaml override flows through.
# ---------------------------------------------------------------------------

class TestConfigOverride:
    """Setting a field in config.yaml replaces the module constant."""

    def test_memory_override_replaces_module_constant(self, tmp_config):
        custom = "MEMORY OVERRIDE: only save cross-session persona facts."
        with _set_config_review_prompts(tmp_config, memory=custom):
            out = _resolve_review_prompt(_FakeAgent(), "memory")
        assert out == custom
        assert out != _MEMORY_REVIEW_PROMPT  # and IS different from default

    def test_skill_override_replaces_module_constant(self, tmp_config):
        custom = "SKILL OVERRIDE: patch loaded skill first, then references."
        with _set_config_review_prompts(tmp_config, skill=custom):
            out = _resolve_review_prompt(_FakeAgent(), "skill")
        assert out == custom
        assert out != _SKILL_REVIEW_PROMPT  # and IS different from default

    def test_combined_override_replaces_module_constant(self, tmp_config):
        custom = "COMBINED OVERRIDE: nothing qualifies? just say so and exit."
        with _set_config_review_prompts(tmp_config, combined=custom):
            out = _resolve_review_prompt(_FakeAgent(), "combined")
        assert out == custom

    def test_per_kind_overrides_are_independent(self, tmp_config):
        """Overriding one kind must not leak into the others."""
        mem_override = "ONLY memory facts."
        with _set_config_review_prompts(tmp_config, memory=mem_override):
            assert _resolve_review_prompt(_FakeAgent(), "memory") == mem_override
            assert _resolve_review_prompt(_FakeAgent(), "skill") == _SKILL_REVIEW_PROMPT
            assert _resolve_review_prompt(_FakeAgent(), "combined") == _COMBINED_REVIEW_PROMPT


# ---------------------------------------------------------------------------
# 3. Empty-string override = explicit "off" sentinel.
# ---------------------------------------------------------------------------

class TestEmptyStringIsPreserved:
    """An empty-string override is a valid "review off" signal and must not
    be coerced to ``None`` / dropped. The fork exits early when the prompt
    is empty; collapsing it to the default would silently disable the
    override."""

    def test_empty_memory_kept_verbatim(self, tmp_config):
        with _set_config_review_prompts(tmp_config, memory=""):
            assert _resolve_review_prompt(_FakeAgent(), "memory") == ""

    def test_empty_skill_kept_verbatim(self, tmp_config):
        with _set_config_review_prompts(tmp_config, skill=""):
            assert _resolve_review_prompt(_FakeAgent(), "skill") == ""


# ---------------------------------------------------------------------------
# 4. Resolution chain — instance attribute wins over config.
# ---------------------------------------------------------------------------

class TestInstanceAttrWinsOverConfig:
    """The resolver must keep the existing ``getattr(agent, ...)`` back-compat
    path working, AND make it win over the new config block — same precedence
    the original code had."""

    def test_instance_attr_wins_for_memory(self, tmp_config):
        instance_text = "INSTANCE memory override"
        with _set_config_review_prompts(tmp_config, memory="CONFIG memory"):
            agent = _FakeAgent(_MEMORY_REVIEW_PROMPT=instance_text)
            assert _resolve_review_prompt(agent, "memory") == instance_text

    def test_instance_attr_wins_for_skill(self, tmp_config):
        instance_text = "INSTANCE skill override"
        with _set_config_review_prompts(tmp_config, skill="CONFIG skill"):
            agent = _FakeAgent(_SKILL_REVIEW_PROMPT=instance_text)
            assert _resolve_review_prompt(agent, "skill") == instance_text

    def test_instance_attr_wins_for_combined(self, tmp_config):
        instance_text = "INSTANCE combined override"
        with _set_config_review_prompts(tmp_config, combined="CONFIG combined"):
            agent = _FakeAgent(_COMBINED_REVIEW_PROMPT=instance_text)
            assert _resolve_review_prompt(agent, "combined") == instance_text


# ---------------------------------------------------------------------------
# 5. Robustness — bad config layouts must not crash the fork.
# ---------------------------------------------------------------------------

class TestBadConfigFallsBack:
    """The review fork runs in a daemon thread — exceptions here are
    swallowed by the ``except Exception`` guards in turn_finalizer and
    the forked thread silently dies. Better to fall back to the module
    constant than to crash."""

    def test_non_dict_review_prompts_falls_through(self, tmp_config):
        """If a user pastes a string into agent.review_prompts by mistake,
        we must not crash — fall through to the module constant."""
        from hermes_cli.config import DEFAULT_CONFIG
        # Build a dict where agent.review_prompts is a string, not a dict.
        bad_config = dict(DEFAULT_CONFIG)
        bad_config["agent"] = dict(bad_config["agent"])
        bad_config["agent"]["review_prompts"] = "oops i pasted a string"
        tmp_config(bad_config)
        assert _resolve_review_prompt(_FakeAgent(), "memory") == _MEMORY_REVIEW_PROMPT
        assert _resolve_review_prompt(_FakeAgent(), "skill") == _SKILL_REVIEW_PROMPT
        assert _resolve_review_prompt(_FakeAgent(), "combined") == _COMBINED_REVIEW_PROMPT

    def test_unknown_kind_returns_memory_constant(self):
        """Unknown kind should return SOMETHING rather than raise — the
        resolver must not crash the review thread on a caller bug."""
        out = _resolve_review_prompt(_FakeAgent(), "nonsense_kind")
        # Lands on _MEMORY_REVIEW_PROMPT (the resolver's documented fallback
        # for unknown kinds). Whatever it is must be a non-empty string.
        assert isinstance(out, str) and out


# ---------------------------------------------------------------------------
# 6. spawn_background_review_thread wires the resolved prompt through.
# ---------------------------------------------------------------------------

class TestSpawnUsesResolver:
    """The end-to-end behaviour: a non-empty override reaches the spawned
    thread; an unset override preserves the default; an empty override
    passes through unchanged."""

    def test_skill_review_uses_config_override(self, tmp_config):
        from agent.background_review import spawn_background_review_thread
        custom = "SKILL OVERRIDE: prefer patches to skills/"
        with _set_config_review_prompts(tmp_config, skill=custom):
            _, prompt = spawn_background_review_thread(
                agent=_FakeAgent(),  # no instance attr → falls through to config
                messages_snapshot=[],
                review_memory=False,
                review_skills=True,
            )
        assert prompt == custom

    def test_combined_review_uses_config_override(self, tmp_config):
        from agent.background_review import spawn_background_review_thread
        custom = "COMBINED OVERRIDE: none of this is memory-worthy."
        with _set_config_review_prompts(tmp_config, combined=custom):
            _, prompt = spawn_background_review_thread(
                agent=_FakeAgent(),
                messages_snapshot=[],
                review_memory=True,
                review_skills=True,
            )
        assert prompt == custom

    def test_spawn_preserves_empty_override(self, tmp_config):
        """An empty-string override is meaningful (review off). The spawn
        function must not collapse it to the default — that would silently
        re-enable a review the user explicitly disabled."""
        from agent.background_review import spawn_background_review_thread
        with _set_config_review_prompts(tmp_config, memory=""):
            _, prompt = spawn_background_review_thread(
                agent=_FakeAgent(),
                messages_snapshot=[],
                review_memory=True,
                review_skills=False,
            )
        assert prompt == ""

    def test_empty_prompt_short_circuits_review_fork(self, tmp_config):
        """Reviewer fix: an empty prompt must skip the forked agent
        entirely — not just emit a sub-minimal user message. Without
        this guard, ``review_agent.run_conversation`` still fires a
        model call when the user explicitly said "review off".

        The resolver reads ``agent._MEMORY_REVIEW_PROMPT`` (priority 1
        in the resolve chain), so the only way to exercise the actual
        ``_run_review_in_thread`` guard without exploding there is to
        provide a sentinel that returns ``None`` from that attr (so
        the resolver falls through to config → config says "" → empty
        prompt propagates) and then explodes on any *other* attribute
        — the explosion marks "we built the review agent", and the
        test passes iff the explosion never fires (early return ran
        before the agent was ever constructed).
        """
        from agent.background_review import spawn_background_review_thread

        touched = {"after_target": False}

        class _SentinelAgent:
            """Behaves during resolver read, then explodes later.

            The resolver wants ``agent._MEMORY_REVIEW_PROMPT`` —
            returning ``None`` lets it fall through to config. Once the
            resolver returns "" from config, the spawn function builds
            ``_target`` and returns. ``_target()`` calls
            ``_run_review_in_thread``, which checks ``if not prompt:
            return`` BEFORE touching any agent attribute. If the guard
            is in place, no explosion; if it's gone, we'd see
            ``touched["after_target"] = True`` set by _run_review_in_thread
            reaching the agent-construction code path.
            """

            _MEMORY_REVIEW_PROMPT = None

            def __getattr__(self, name):
                # The resolver only reads _MEMORY_REVIEW_PROMPT, which
                # is set as a class attr above. ANY other attribute
                # access means we've reached the run_conversation /
                # review-agent-construction code path. Mark + raise.
                touched["after_target"] = True
                raise AssertionError(
                    f"empty-prompt review reached the build-review-agent "
                    f"code path (attr={name}); the early-return guard in "
                    f"_run_review_in_thread should have skipped this"
                )

        with _set_config_review_prompts(tmp_config, memory=""):
            target, prompt = spawn_background_review_thread(
                agent=_SentinelAgent(),
                messages_snapshot=[],
                review_memory=True,
                review_skills=False,
            )
        assert prompt == ""
        target()  # must early-return without reaching _SentinelAgent attrs
        assert touched["after_target"] is False, (
            "review fork reached the build-review-agent code path — "
            "the empty-prompt short-circuit in _run_review_in_thread "
            "is missing or was bypassed"
        )

    def test_spawn_uses_instance_attr_first(self, tmp_config):
        """When an instance attr is set, it wins over both config and
        module constant — this is the back-compat path for existing tests
        that monkey-patch ``AIAgent._SKILL_REVIEW_PROMPT`` etc."""
        from agent.background_review import spawn_background_review_thread
        instance_text = "INSTANCE skill — back-compat path"
        with _set_config_review_prompts(tmp_config, skill="CONFIG skill"):
            agent = _FakeAgent(_SKILL_REVIEW_PROMPT=instance_text)
            _, prompt = spawn_background_review_thread(
                agent=agent,
                messages_snapshot=[],
                review_memory=False,
                review_skills=True,
            )
        assert prompt == instance_text


# ---------------------------------------------------------------------------
# 7. Mapping table — every declared kind has a corresponding prompt constant.
# ---------------------------------------------------------------------------

def test_review_prompt_mapping_covers_all_kinds():
    """The mapping table is the single source of truth for kind→attr.

    Adding a new review kind must require an edit here; if you add a kind
    but forget the mapping, ``_resolve_review_prompt`` quietly falls back
    to _MEMORY_REVIEW_PROMPT, which is wrong-and-silent. Lock it down.
    """
    expected_attrs = {"_MEMORY_REVIEW_PROMPT", "_SKILL_REVIEW_PROMPT", "_COMBINED_REVIEW_PROMPT"}
    actual_attrs = set(_REVIEW_PROMPT_BY_KIND.values())
    assert actual_attrs == expected_attrs
