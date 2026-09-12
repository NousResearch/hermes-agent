"""Readable conversation rows and notification ordering."""
from queue import Queue
from types import SimpleNamespace

import pytest
from rich.text import Text

from cli import HermesCLI
from hermes_cli.cli_conversation_display import render_user_preview, render_review_notice
from hermes_cli.skin_engine import get_active_skin


def test_user_preview_wraps_before_budget_and_keeps_literal_input(monkeypatch):
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('COLORTERM', 'truecolor')
    monkeypatch.setenv('TERM', 'xterm-256color')
    short = '[bold]literal[/bold] **not Markdown**'
    output = render_user_preview(short, 80)
    assert short in output
    assert 'You' not in output
    assert not any(char in output for char in '╭╰│─')
    assert all(Text.from_ansi(row).cell_len == 80 for row in output.splitlines())
    wide = render_user_preview('Short message', 140)
    assert Text.from_ansi(wide.splitlines()[0]).cell_len == 140
    expected_bg = get_active_skin().get_color('status_bar_bg', '#1a1a2e')
    assert any(span.style.bgcolor and span.style.bgcolor.get_truecolor().hex == expected_bg.lower()
               for span in Text.from_ansi(output).spans)
    long = 'First words ' + 'middle words ' * 50 + 'last words'
    output = render_user_preview(long, 40, first=2, last=1, timestamp='12:34')
    assert 'First words' in output and 'last words' in output
    assert 'more lines' in output and '12:34' in output
    assert all(Text.from_ansi(row).cell_len <= 40 for row in output.splitlines())
    for width in (12, 40, 140):
        band = render_user_preview(long, width, first=1, last=1, timestamp='12:34:56')
        assert all(Text.from_ansi(row).cell_len == width for row in band.splitlines())
    assert long.endswith('last words')


def test_review_notice_waits_for_assistant_and_preserves_all_details(monkeypatch):
    import cli as facade
    emitted = []
    monkeypatch.setattr(facade, '_cprint', emitted.append)
    cli = HermesCLI.__new__(HermesCLI)
    cli.final_response_markdown = 'render'
    cli.show_reasoning = False
    cli._reset_stream_state()
    cli._stream_delta('Assistant is still speaking.')
    detail = "Skill 'example' patched (SKILL.md) · Skill 'example' patched (references/guide.md)"
    cli._agent_status_print('  💾 Self-improvement review: ' + detail)
    assert not emitted
    cli._flush_stream()
    output = '\n'.join(emitted)
    assert output.index('Assistant is still speaking.') < output.index('Self-improvement review')
    for part in ('SKILL.md', 'references/guide.md'):
        assert part in output
    assert '💾' not in output
    cli._agent_status_print('WARNING: example failure')
    assert emitted[-1] == 'WARNING: example failure'
    assert all(Text.from_ansi(row).cell_len <= 30
               for row in render_review_notice(detail, 30).splitlines())


@pytest.mark.parametrize('mode,label', [('queue', 'Queued for next turn'),
                                        ('steer', 'Steered'),
                                        ('interrupt', 'Redirected current turn')])
def test_busy_acknowledgments_share_quiet_layout_and_preserve_payload(monkeypatch, mode, label):
    import cli as facade
    emitted, accepted = [], []
    monkeypatch.setattr(facade, '_cprint', emitted.append)
    monkeypatch.setattr('agent.onboarding.is_seen', lambda *args: True)
    cli = HermesCLI.__new__(HermesCLI)
    cli.final_response_markdown = 'render'
    cli.busy_input_mode = mode
    cli._pending_input = Queue()
    cli._interrupt_queue = Queue()
    cli.agent = SimpleNamespace(steer=lambda text: accepted.append(text) or True,
                                redirect=lambda text: accepted.append(text) or True,
                                _supports_active_turn_redirect=True)
    text = '[bold]keep this literal[/bold]'
    cli._tui_enter_while_busy(text, [], text)
    if mode == 'queue':
        assert cli._pending_input.get_nowait() == text
    else:
        assert accepted == [text]
    output = Text.from_ansi('\n'.join(emitted)).plain
    assert output.startswith('  ' + label + '\n    ')
    assert text in output
    assert not any(mark in output for mark in ['⏩', '↪'])
    assert cli._interrupt_queue.empty()


@pytest.mark.parametrize('mode,label', [('queue', 'Queued for next turn'),
                                        ('steer', 'Steered'),
                                        ('interrupt', 'Redirected current turn')])
def test_busy_acknowledgment_flushes_live_assistant_before_notice(monkeypatch, mode, label):
    import cli as facade
    emitted = []
    monkeypatch.setattr(facade, '_cprint', emitted.append)
    monkeypatch.setattr('agent.onboarding.is_seen', lambda *args: True)
    cli = HermesCLI.__new__(HermesCLI)
    cli.final_response_markdown = 'render'
    cli.show_reasoning = False
    cli.busy_input_mode = mode
    cli._pending_input = Queue()
    cli._interrupt_queue = Queue()
    cli.agent = SimpleNamespace(steer=lambda text: True, redirect=lambda text: True,
                                _supports_active_turn_redirect=True)
    cli._reset_stream_state()
    cli._stream_delta('Assistant text that must land before the acknowledgment.')
    cli._tui_enter_while_busy('follow-up', [], 'follow-up')
    output = Text.from_ansi('\n'.join(emitted)).plain
    assert output.index('Assistant text that must land before the acknowledgment.') < output.index(label)


def test_slash_queue_and_steer_use_same_layout_without_hiding_failure(monkeypatch):
    import cli as facade
    emitted = []
    monkeypatch.setattr(facade, '_cprint', emitted.append)
    cli = HermesCLI.__new__(HermesCLI)
    cli.final_response_markdown = 'render'
    cli._pending_input = Queue()
    cli._agent_running = True
    cli.agent = SimpleNamespace(steer=lambda text: True)
    cli._cmd_queue('/queue Next task')
    cli._cmd_steer('/steer Focus here')
    assert cli._pending_input.get_nowait() == 'Next task'
    output = Text.from_ansi('\n'.join(emitted)).plain
    assert '  Queued for next turn\n    Next task' in output
    assert '  Steering queued after next tool call\n    Focus here' in output
    cli.agent.steer = lambda text: (_ for _ in ()).throw(RuntimeError('example failure'))
    cli._cmd_steer('/steer Again')
    assert emitted[-1] == '  Steer failed: example failure'
