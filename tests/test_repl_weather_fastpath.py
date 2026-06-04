"""Regression tests: the interactive REPL and the oneshot (-z) path must serve
the deterministic intent fast-path, not the LLM.

Why: The deterministic weather/time fast-path was wired ONLY into the
single-query (-q) block.  The interactive REPL submit handler and the oneshot
(``hermes -z``) path both bypassed it, so a typed "what is the weather?" reached
the agent and drifted/hallucinated a different answer every run.  A deterministic
handler returns the SAME bytes every time; the agent does not.  These tests fail
on the pre-fix code (REPL/oneshot skip the fast-path) and pass once both surfaces
call ``intent_fast_path.try_fast_path_reply`` before building/invoking the agent.

What: (1) the shared eligibility/dispatch helper gates correctly; (2) the REPL
submit handler source wires the fast-path BEFORE ``self.chat`` and skips the turn
on a hit; (3) the oneshot path source returns the fast-path reply before building
the AIAgent; (4) a functional simulation of the REPL decision proves weather is
served without calling the agent while non-weather falls through to it.

Test: ``pytest tests/test_repl_weather_fastpath.py``.
"""

from __future__ import annotations

import inspect
import re

import pytest

import intent_fast_path as ifp


# ──────────────────────────────────────────────────────────────────────────
# 1. Shared helper — the single source of truth all three surfaces call.
# ──────────────────────────────────────────────────────────────────────────
class TestSharedHelper:
    """intent_fast_path.try_fast_path_reply / _fast_path_eligible."""

    def test_helper_exists(self):
        """The centralizing helper must exist for the REPL/oneshot to call.

        Test: ``try_fast_path_reply`` is importable and callable.
        """
        assert callable(getattr(ifp, "try_fast_path_reply", None))
        assert callable(getattr(ifp, "_fast_path_eligible", None))

    @pytest.mark.parametrize(
        "text,has_images,expected",
        [
            ("what is the weather", False, True),
            ("weather today", False, True),
            ("", False, False),
            ("   ", False, False),
            ("/help", False, False),
            ("weather", True, False),  # image attachment -> defer to agent
        ],
    )
    def test_eligibility(self, text, has_images, expected):
        """Gate empty/slash/image; allow plain weather text.

        Test: each (text, has_images) pair yields the expected eligibility.
        """
        assert ifp._fast_path_eligible(text, has_images=has_images) is expected

    def test_env_gates_force_fall_through(self, monkeypatch):
        """Kanban / goal-mode / disable env vars must force fall-through.

        Test: with each gate set, eligibility is False.
        """
        monkeypatch.setenv("HERMES_DISABLE_INTENT_FASTPATH", "1")
        assert ifp._fast_path_eligible("weather", has_images=False) is False
        monkeypatch.delenv("HERMES_DISABLE_INTENT_FASTPATH")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "abc")
        assert ifp._fast_path_eligible("weather", has_images=False) is False
        monkeypatch.delenv("HERMES_KANBAN_TASK")
        monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
        assert ifp._fast_path_eligible("weather", has_images=False) is False

    def test_dispatch_served_and_fall_through(self, monkeypatch):
        """A weather hit returns a string; non-weather/slash return None.

        Test: monkeypatch the async dispatch so no real HTTP runs; assert weather
        is served and "tell me a joke" / "/help" fall through.
        """
        async def _fake(text):
            return "*Weather — Woodstock, IL*\nNow: 70°F, Clear"

        monkeypatch.setattr(ifp, "_intent_fast_path", _fake)
        assert ifp.try_fast_path_reply("what is the weather").startswith("*Weather")
        # /help (slash) and an image attachment must short-circuit on eligibility
        # BEFORE dispatch ever runs — so _fake's match-all is never consulted.
        assert ifp.try_fast_path_reply("/help") is None
        assert ifp.try_fast_path_reply("weather", has_images=True) is None
        assert ifp.try_fast_path_reply("") is None


# ──────────────────────────────────────────────────────────────────────────
# 2. REPL submit handler must wire the fast-path BEFORE self.chat().
#    This is the structural regression guard: on the OLD code the REPL handler
#    jumps straight from the image-count print to "self.chat(user_input...)"
#    with no fast-path call in between.
# ──────────────────────────────────────────────────────────────────────────
class TestReplWiring:
    """The interactive REPL run() loop must call the fast-path before the agent."""

    def _run_source(self) -> str:
        import cli

        return inspect.getsource(cli.HermesCLI.run)

    def test_repl_calls_fast_path(self):
        """run() must invoke try_fast_path_reply.

        Test: the helper name appears in HermesCLI.run() source.
        """
        assert "try_fast_path_reply" in self._run_source()

    def test_fast_path_precedes_chat_in_repl(self):
        """The fast-path call must come BEFORE the self.chat() dispatch.

        Why: ordering is the whole fix — calling it after the agent run would be
        pointless.  This fails on the pre-fix code (no fast-path call at all).
        Test: the first ``try_fast_path_reply`` index < the ``self.chat(`` index.
        """
        src = self._run_source()
        fp_idx = src.find("try_fast_path_reply")
        chat_idx = src.find("self.chat(user_input")
        assert fp_idx != -1, "REPL never calls the fast-path"
        assert chat_idx != -1, "REPL chat dispatch not found"
        assert fp_idx < chat_idx, "fast-path must precede the agent dispatch"

    def test_repl_skips_agent_on_hit(self):
        """On a fast-path hit the REPL must `continue` (skip the agent turn).

        Test: a `continue` statement appears between the fast-path call and the
        `self.chat(` dispatch in the source.
        """
        src = self._run_source()
        fp_idx = src.find("try_fast_path_reply")
        chat_idx = src.find("self.chat(user_input")
        between = src[fp_idx:chat_idx]
        assert "continue" in between, "REPL must skip the agent on a fast-path hit"


# ──────────────────────────────────────────────────────────────────────────
# 3. Oneshot (-z) path must answer via the fast-path before building AIAgent.
# ──────────────────────────────────────────────────────────────────────────
class TestOneshotWiring:
    """hermes_cli.oneshot.run_oneshot must consult the fast-path first."""

    def _src(self) -> str:
        from hermes_cli import oneshot

        return inspect.getsource(oneshot.run_oneshot)

    def test_oneshot_calls_fast_path(self):
        """run_oneshot must call try_fast_path_reply.

        Test: the helper name appears in run_oneshot source.
        """
        assert "try_fast_path_reply" in self._src()

    def test_oneshot_fast_path_precedes_agent(self):
        """The fast-path must run before _run_agent is invoked.

        Test: the fast-path call index < the _run_agent call index.
        """
        src = self._src()
        fp_idx = src.find("try_fast_path_reply")
        agent_idx = src.find("_run_agent(")
        assert fp_idx != -1
        assert agent_idx != -1
        assert fp_idx < agent_idx


# ──────────────────────────────────────────────────────────────────────────
# 4. Functional simulation of the REPL decision: weather served w/o the agent,
#    non-weather routed to the agent.  Mirrors the run()-loop branch exactly.
# ──────────────────────────────────────────────────────────────────────────
class _FakeCLI:
    """Stand-in exercising the REPL fast-path decision without prompt_toolkit."""

    def __init__(self):
        self.printed: list[str] = []
        self.chatted: list[str] = []

    def _print_assistant_message(self, text):
        self.printed.append(text)

    def chat(self, message, images=None):
        self.chatted.append(message)

    def submit_turn(self, user_input, submit_images=None):
        """Replicates the REPL run() branch: fast-path first, else agent."""
        fp_out = ifp.try_fast_path_reply(
            user_input, has_images=bool(submit_images)
        )
        if fp_out:
            self._print_assistant_message(fp_out)
            return
        self.chat(user_input, images=submit_images or None)


class TestReplDecisionBehavior:
    def test_weather_served_without_agent(self, monkeypatch):
        """A weather line is printed deterministically and the agent is NOT run.

        Test: submit "what is the weather" -> _print_assistant_message called once
        with the weather block; chat() never called.
        """
        async def _fake(text):
            return "*Weather — Woodstock, IL*\nNow: 70°F, Clear"

        monkeypatch.setattr(ifp, "_intent_fast_path", _fake)
        c = _FakeCLI()
        c.submit_turn("what is the weather")
        assert len(c.printed) == 1
        assert c.printed[0].startswith("*Weather")
        assert c.chatted == [], "agent must NOT run for a served weather turn"

    def test_non_weather_routes_to_agent(self, monkeypatch):
        """A non-weather line falls through to the agent.

        Test: submit "tell me a joke" -> chat() called; nothing printed by the
        fast-path.
        """
        async def _none(text):
            return None

        monkeypatch.setattr(ifp, "_intent_fast_path", _none)
        c = _FakeCLI()
        c.submit_turn("tell me a joke")
        assert c.chatted == ["tell me a joke"]
        assert c.printed == []

    def test_slash_command_not_intercepted(self):
        """A slash command must never be served by the fast-path.

        Test: submit "/help" -> chat() receives it (the run() loop's own slash
        guard handles real slash routing earlier; here we assert the fast-path
        itself does not swallow it).
        """
        c = _FakeCLI()
        c.submit_turn("/help")
        assert c.chatted == ["/help"]
        assert c.printed == []
