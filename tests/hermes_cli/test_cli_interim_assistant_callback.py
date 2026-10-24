"""The interactive CLI installs an ``interim_assistant_callback``.

Regression guard for the codex-quiet-CLI gap: the TUI/gateway deliver mid-turn model
commentary as ``message.interim``, but the plain CLI wired no interim callback, so
``agent/codex_runtime.py`` routed Codex ``phase=commentary`` text to the reasoning channel
and the user saw no narration between tool calls (while chat_completions providers such as
DeepSeek showed the same narration live via ``stream_delta_callback``).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_cli_stub(printed):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._captured = printed
    return cli


@pytest.fixture
def captured(monkeypatch):
    """Capture ``_cprint`` without touching the real terminal."""
    import cli as cli_module

    out = []
    monkeypatch.setattr(cli_module, "_cprint", lambda text: out.append(text), raising=False)
    return out


def test_interim_commentary_is_printed(captured):
    cli = _make_cli_stub(captured)
    cli._on_interim_assistant("Voy a revisar la config real antes de concluir.")
    assert captured == ["Voy a revisar la config real antes de concluir."]


def test_already_streamed_commentary_is_not_printed_twice(captured):
    cli = _make_cli_stub(captured)
    # Chat-completions providers stream the same text live (stream_delta_callback); the interim
    # projection must not repeat it in scrollback.
    cli._on_interim_assistant("texto ya pintado por el stream", already_streamed=True)
    assert captured == []


@pytest.mark.parametrize("payload", ["", "   ", "\n", None])
def test_blank_commentary_is_ignored(captured, payload):
    cli = _make_cli_stub(captured)
    cli._on_interim_assistant(payload)
    assert captured == []


def test_callback_signature_matches_agent_call_site(captured):
    """``_deliver_interim`` calls ``cb(visible, already_streamed=...)`` positionally/by keyword."""
    cli = _make_cli_stub(captured)
    cli._on_interim_assistant("hola", already_streamed=False)
    assert captured and "hola" in captured[0]
