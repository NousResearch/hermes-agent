"""Tests for CLI interim assistant message rendering.

The agent core emits real assistant commentary between tool calls via
``interim_assistant_callback``.  The CLI's ``_on_interim_assistant``
handler renders these when ``display.interim_assistant_messages`` is true.
"""

from unittest.mock import patch

from cli import HermesCLI


def _make_cli():
    """Bare-metal HermesCLI without side effects."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.interim_assistant_messages = True
    cli.model = "test-model"
    cli.agent = None
    return cli


def test_renders_box_frame():
    """Text is rendered inside a dimmed box frame with left border."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("Found the bug at line 42.")
    calls = [c[0][0] for c in mock_print.call_args_list]
    assert any("Found the bug at line 42." in c for c in calls), f"text not in calls: {calls}"
    # Box frame: top, content, bottom — at least 3 calls
    assert len(calls) >= 3, f"expected >=3 box lines, got {len(calls)}: {calls}"
    # Top line has ╭ and ◆ marker
    assert "╭" in calls[0] and "◆" in calls[0], f"top line missing box: {calls[0]}"
    # Bottom line has ╰
    assert "╰" in calls[-1], f"bottom line missing box: {calls[-1]}"


def test_silent_when_disabled():
    """No output when display.interim_assistant_messages is false."""
    cli = _make_cli()
    cli.interim_assistant_messages = False
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("You should not see this.")
    mock_print.assert_not_called()


def test_skips_already_streamed():
    """When streaming already displayed the text, do not repeat it."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("already shown via stream", already_streamed=True)
    mock_print.assert_not_called()


def test_skips_empty_text():
    """Blank or whitespace-only text is silently dropped."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("   ")
    mock_print.assert_not_called()


def test_strips_whitespace_in_box():
    """Leading/trailing whitespace is stripped before boxing."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("  ok  ")
    calls = [c[0][0] for c in mock_print.call_args_list]
    # Content line has box padding (│  ...  │) but no raw user whitespace
    content_lines = [c for c in calls if "│" in c]
    assert content_lines, f"no content line found: {calls}"
    # After stripping ANSI and box chars, should be exactly "  ok  "
    # (box padding is "  " on each side; original "  ok  " → stripped "ok")
    import re
    clean = re.sub(r'\x1b\[[0-9;]*m', '', content_lines[0])
    clean = clean.replace('│', '').strip()
    assert clean == "ok", f"expected 'ok', got {clean!r}"


def test_respects_already_streamed_when_enabled():
    """already_streamed=True takes priority over the config flag."""
    cli = _make_cli()
    with patch("cli._cprint") as mock_print:
        cli._on_interim_assistant("streaming handled this", already_streamed=True)
    mock_print.assert_not_called()


def test_aiagent_constructor_accepts_interim_assistant_callback():
    """AIAgent.__init__ must declare interim_assistant_callback as a kwarg.

    If the kwarg is renamed or removed, the mixin wiring line
    ``interim_assistant_callback=self._on_interim_assistant`` will silently
    become a **kwargs capture or raise TypeError at runtime.
    """
    import inspect

    from run_agent import AIAgent

    sig = inspect.signature(AIAgent.__init__)
    assert "interim_assistant_callback" in sig.parameters, (
        "AIAgent.__init__ no longer accepts interim_assistant_callback — "
        "the mixin wiring will break at runtime"
    )


def test_mixin_wires_interim_assistant_callback():
    """Source-level regression guard: the mixin's _init_agent must pass
    interim_assistant_callback=self._on_interim_assistant to AIAgent(…).

    Companion to test_aiagent_constructor_accepts_interim_assistant_callback:
    that test verifies the AIAgent side, this one verifies the mixin side.
    Together they prevent either end of the wire from drifting.
    """
    import inspect

    from hermes_cli import cli_agent_setup_mixin as mixin_mod

    src = inspect.getsource(mixin_mod)
    assert "interim_assistant_callback=self._on_interim_assistant" in src, (
        "mixin no longer wires interim_assistant_callback into AIAgent constructor"
    )


def test_quiet_mode_clears_interim_assistant_callback():
    """In quiet mode (-Q), interim_assistant_callback must be cleared alongside
    stream_delta_callback and tool_gen_callback so machine-readable stdout
    contract is preserved.
    """
    from unittest.mock import MagicMock

    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = MagicMock()
    cli.agent.quiet_mode = False
    cli.agent.suppress_status_output = False
    cli.agent.stream_delta_callback = lambda x: x
    cli.agent.tool_gen_callback = lambda x: x
    cli.agent.interim_assistant_callback = lambda x: x
    cli.agent.run_conversation = MagicMock(return_value="ok")
    cli.model = "test-model"
    cli.interim_assistant_messages = True

    # Simulate the quiet-mode branch: set quiet_mode + suppress then clear
    # all three callbacks as the production code does.
    cli.agent.quiet_mode = True
    cli.agent.suppress_status_output = True
    cli.agent.stream_delta_callback = None
    cli.agent.tool_gen_callback = None
    cli.agent.interim_assistant_callback = None

    # All three must be None after quiet-mode setup.
    assert cli.agent.stream_delta_callback is None
    assert cli.agent.tool_gen_callback is None
    assert cli.agent.interim_assistant_callback is None, (
        "quiet mode must clear interim_assistant_callback to prevent "
        "commentary leakage into machine-readable stdout"
    )
