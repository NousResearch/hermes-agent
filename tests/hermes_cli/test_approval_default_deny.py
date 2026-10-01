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
