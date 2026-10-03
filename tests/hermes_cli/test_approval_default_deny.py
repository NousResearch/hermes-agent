"""An untouched CLI approval must refuse; explicit approval stays available."""
import pytest

from hermes_cli.cli_modal_mixin import CLIModalMixin


class ApprovalUI(CLIModalMixin):
    """Real modal state/selection/queue with only terminal painting suppressed."""

    def __init__(self, choose=None):
        import threading
        self._approval_lock = threading.Lock()
        self._approval_state = None
        self.choose = choose

    def _paint_now(self):
        if self._approval_state is not None:
            if self.choose is not None:
                self._approval_state['selected'] = self._approval_state['choices'].index(self.choose)
            self._handle_approval_selection()

    def _invalidate(self):
        pass

    def _ring_bell(self, **kwargs):
        pass

    def _persist_prompt_summary(self, *args):
        pass


@pytest.mark.parametrize('options', [{}, {'allow_permanent': False}, {'allow_session': False}, {'smart_denied': True}])
@pytest.mark.parametrize('command', ['git clean -fd', 'git reset --hard ' + 'a' * 80])
def test_untouched_approval_enter_denies(options, command):
    ui = ApprovalUI()
    assert ui._approval_callback(command, 'test only; no command is executed', **options) == 'deny'
    assert ui._approval_state is None


@pytest.mark.parametrize('choice', ['once', 'session', 'always', 'deny'])
def test_explicit_approval_choice_preserved(choice):
    assert ApprovalUI(choice)._approval_callback('git clean -fd', 'test only') == choice


@pytest.mark.parametrize('sequence, expected', [('\r', 'deny'), ('\x1b[A' * 3 + '\r', 'once')])
def test_pipe_keys_reach_real_approval_overlay(sequence, expected):
    """Parse real key bytes and route them through the production modal handlers."""
    import asyncio
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import BufferControl, Layout, Window
    from prompt_toolkit.output import DummyOutput
    from hermes_cli.cli_tui_mixin import CLITuiMixin

    class KeyUI(ApprovalUI, CLITuiMixin):
        _connection_state = _sudo_state = _secret_state = None

        def _paint_now(self):
            pass

        def _poll_modal_queue(self, response_queue, deadline_attr, **kwargs):
            async def drive():
                kb = KeyBindings()
                kb.add('up')(self._tui_approval_up)

                @kb.add('enter')
                def enter(event):
                    assert self._tui_enter_overlay(event)
                    event.app.exit()

                with create_pipe_input() as inp:
                    app = Application(
                        layout=Layout(Window(BufferControl(Buffer()))), key_bindings=kb,
                        input=inp, output=DummyOutput())
                    await asyncio.wait_for(app.run_async(
                        pre_run=lambda: inp.send_text(sequence)), timeout=5)
            asyncio.run(drive())
            return response_queue.get_nowait()

    assert KeyUI()._approval_callback('git clean -fd', 'test only') == expected
