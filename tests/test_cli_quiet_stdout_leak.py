"""`hermes chat -Q` must not leak presentation output into stdout (#93220).

The quiet single-query branch lives inline in ``cli.py``'s main flow (no
standalone function to call), so these pin the branch's required statements
at the source level — the established convention for behavior with no
runtime mirror (see the install.ps1 source-text tests). Deleting any
neutralization reintroduces a leak:

- ``reasoning_callback``       → streaming ``┌─ Reasoning ─┐`` box
- ``tool_complete_callback``   → full file diffs via render_edit_diff_with_delta
- ``tool_progress_callback``   → MoA reference blocks (printed before the
                                 mode check in _on_tool_progress)
- ``tool_start_callback``      → snapshot capture paired with tool_complete
- ``agent.tool_progress_mode`` → executor's direct progress prints
"""

from __future__ import annotations

from pathlib import Path

import cli as cli_mod

_QUIET_ANCHOR = "# Quiet mode: suppress banner, spinner, tool previews."


def _quiet_branch() -> str:
    """Return the source slice of the quiet single-query branch."""
    source = Path(cli_mod.__file__).read_text(encoding="utf-8")
    start = source.index(_QUIET_ANCHOR)
    # A generous window covers the whole branch body up to the
    # run_conversation call and beyond.
    return source[start : start + 8000]


def test_quiet_branch_clears_reasoning_callback():
    branch = _quiet_branch()
    assert "cli.agent.reasoning_callback = None" in branch, (
        "-Q must clear the reasoning callback on the live agent, or the "
        "streaming Reasoning box renders into captured stdout (#93220)."
    )


def test_quiet_branch_clears_inline_diff_callbacks():
    branch = _quiet_branch()
    for attr in (
        "cli.agent.tool_start_callback = None",
        "cli.agent.tool_complete_callback = None",
    ):
        assert attr in branch, (
            f"-Q must set `{attr}`: the inline-diff callbacks print full "
            "file diffs via render_edit_diff_with_delta and are gated by "
            "neither quiet_mode nor tool_progress_mode (#93220)."
        )


def test_quiet_branch_clears_tool_progress_callback():
    branch = _quiet_branch()
    assert "cli.agent.tool_progress_callback = None" in branch, (
        "-Q must clear tool_progress_callback: _on_tool_progress prints "
        "MoA reference blocks before its tool_progress_mode check (#93220)."
    )


def test_quiet_branch_syncs_tool_progress_off_to_agent():
    branch = _quiet_branch()
    assert 'cli.agent.tool_progress_mode = "off"' in branch, (
        "-Q must sync tool_progress_mode='off' to the live agent so the "
        "tool_executor rendering path stays silent (#93220)."
    )


def test_suppress_status_output_gates_quiet_tool_messages():
    """The executor's [tool]/[done] fallback must stay silent under -Q.

    ``_should_emit_quiet_tool_messages`` is the gate for the quiet-mode
    KawaiiSpinner fallback in agent/tool_executor.py; with the rendering
    callbacks neutralized it would otherwise print ``[tool]``/``[done]``
    lines straight into -Q's captured stdout (#93220).
    """
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.quiet_mode = True
    agent.tool_progress_callback = None
    agent.platform = "cli"

    agent.suppress_status_output = False
    assert agent._should_emit_quiet_tool_messages() is True

    agent.suppress_status_output = True
    assert agent._should_emit_quiet_tool_messages() is False


def test_quiet_branch_neutralizations_precede_run_conversation():
    branch = _quiet_branch()
    run_idx = branch.index("run_conversation(")
    for attr in (
        "cli.agent.reasoning_callback = None",
        "cli.agent.tool_progress_callback = None",
        "cli.agent.tool_start_callback = None",
        "cli.agent.tool_complete_callback = None",
    ):
        assert branch.index(attr) < run_idx, (
            f"`{attr}` must run before run_conversation in the quiet branch"
        )
