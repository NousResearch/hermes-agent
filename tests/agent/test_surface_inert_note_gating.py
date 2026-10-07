"""The inert-pinned-tools note must name only genuinely surface-scoped tools.

Live shape (desktop surface, real Bot Chat): the note named ``message_agent`` as inert, yet two
real ``message_agent`` calls immediately afterwards returned ``{status: queued, delivery_id}``.
On the CLI surface the same filter named ``skill_manage``.  One filter producing a false positive
and a true positive is the bug.

``built_for_this_surface`` is what the surface built BEFORE the pin merged a previous surface's
tools back in (``conversation_loop.py::_restore_pinned_tools``).  A canonical Bot Chat never
forks, so its persisted ``tools[]`` is a fossil of session creation
(``_refresh_bot_chat_tools``) and carries names this surface never built but which are fully
callable.  A blanket set-difference therefore names live tools inert; the note must intersect
with the surface-scoped set (``CLIENT_SURFACE_TOOLSETS``) instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.surface_switch import _SURFACE_SWITCH_NOTE_PREFIX

_MARKER = "were not loaded for this interface"
# Carried forward by the persisted pin but NOT surface-scoped, so fully callable here.
# ``message_agent`` is the live reproduction; ``skill_manage`` is what the note named on the CLI.
_CALLABLE_FOSSILS = ("message_agent", "skill_manage")


def _make_agent(session_db):
    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = "sess-inert-note"
    agent.model = "test-model"
    agent.provider = "openrouter"
    agent.api_mode = "chat_completions"
    agent.platform = "cli"
    agent._session_db = session_db
    agent._use_prompt_caching = False
    agent._build_system_prompt = MagicMock(return_value="BUILT_PROMPT")
    agent.enabled_toolsets = agent.disabled_toolsets = None
    return agent


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _restore(*, carried, built_here=("read_file",)):
    """Restore a desktop->cli switch whose pin carries ``carried`` on the wire.

    ``built_here`` is what the CLI surface built before the pin merged the previous surface's
    tools back in; ``carried`` is the full on-wire array after the merge.
    """
    db = MagicMock()
    db.get_session.return_value = {
        "system_prompt": (
            "SYSTEM PROMPT BODY\n\nConversation started: Monday, January 05, 2026\n"
            "Model: test-model\nProvider: openrouter\nPlatform: desktop"
        ),
        "tool_names": __import__("json").dumps([t["function"]["name"] for t in carried]),
    }
    agent = _make_agent(db)
    agent.platform = "cli"
    agent.tools = [_tool(n) for n in built_here]
    agent._platform_hint_overrides = None
    agent._surface_switch_note = ""
    agent._gateway_turn_context_notes = ""

    def _pin(agent_, saved_names):
        agent_.tools = list(carried)
        return True

    # The symbol is late-imported inside _restore_pinned_tools, so patch the defining module —
    # the binding production actually reads (agent/AGENTS.md "patch where production reads";
    # same seam the existing test_system_prompt_restore.py uses).
    with patch("tools.mcp_tool_agent.restore_agent_tool_prefix", _pin):
        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])
    return agent


def _named_inert(note: str) -> list:
    """Parse the tool names the note advertises as inert."""
    assert _MARKER in note, f"note did not name any inert tool:\n{note}"
    tail = note.rsplit(_MARKER, 1)[1]
    tail = tail.split(":", 1)[1] if ":" in tail else tail
    return [n.strip(" .]") for n in tail.split(",") if n.strip(" .]")]


def test_note_does_not_name_a_callable_non_surface_scoped_tool():
    """The reproduction: a carried-forward tool with no surface gate must NOT be called inert.

    ``message_agent`` answers on every surface.  Naming it inert tells the model to avoid a
    working tool — the live false positive, where two calls after the note returned queued.
    """
    carried = [_tool("read_file"), _tool("focus_pane")] + [_tool(n) for n in _CALLABLE_FOSSILS]
    agent = _restore(carried=carried)
    note = agent._surface_switch_note
    assert note, "expected a surface-switch note to be staged"
    named = _named_inert(note)
    for name in _CALLABLE_FOSSILS:
        assert name not in named, (
            f"note named a CALLABLE tool as inert: {named}\nfull note:\n{note}"
        )


def test_note_still_names_the_genuinely_surface_scoped_tool():
    """The true positive survives: ``focus_pane`` IS desktop-only.

    A terminal turn can only answer it with ``tool_error("desktop only")`` — this is the case the
    note exists for, and why the fix is an intersection rather than removing the note.
    """
    carried = [_tool("read_file"), _tool("focus_pane")] + [_tool(n) for n in _CALLABLE_FOSSILS]
    agent = _restore(carried=carried)
    named = _named_inert(agent._surface_switch_note)
    assert "focus_pane" in named, f"the real desktop-only tool was not named: {named}"


def test_note_omits_the_inert_clause_when_nothing_surface_scoped_was_carried():
    """Nothing surface-scoped carried forward -> note stages, but with no inert clause.

    The switch note itself must still be delivered (it retires the stale surface guidance);
    only the inert tail is withheld, because there is nothing genuinely inert to warn about.
    """
    carried = [_tool("read_file")] + [_tool(n) for n in _CALLABLE_FOSSILS]
    agent = _restore(carried=carried)
    note = agent._surface_switch_note
    assert note, "the surface-switch note itself must still stage"
    assert _SURFACE_SWITCH_NOTE_PREFIX in note
    assert _MARKER not in note, f"named live tools inert when nothing gated was carried:\n{note}"
