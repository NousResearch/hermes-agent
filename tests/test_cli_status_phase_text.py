"""T5.2: the tool spinner label must carry a phase word (reading / writing / waiting).

RED premise: tool.started sets ``_spinner_text = emoji + label`` with no phase; the user
cannot tell a read from a write while waiting.
"""
import time

from agent.i18n import t
from hermes_cli.cli_stream_mixin import CLIStreamMixin


class _Stream(CLIStreamMixin):
    def __init__(self):
        self._spinner_text = ""
        self._pet_reasoning = False
        self._pet_turn_error = False
        self._tool_start_time = 0.0
        self.tool_progress_mode = "off"
        self._pending_tool_info = {}
        self._last_scrollback_tool = ""

    def _invalidate(self):
        pass


def _started(tool, preview="payload"):
    s = _Stream()
    s._on_tool_progress("tool.started", tool, preview=preview, function_args={"path": "x"})
    return s


def test_write_tool_uses_write_phase():
    s = _started("write_file")
    assert t("cli.stream.phase_writing") in s._spinner_text
    assert s._tool_start_time <= time.monotonic()


def test_read_tool_uses_read_phase():
    s = _started("read_file")
    assert t("cli.stream.phase_reading") in s._spinner_text


def test_other_tool_uses_wait_phase():
    s = _started("delegate")
    assert t("cli.stream.phase_waiting") in s._spinner_text


def test_cjk_preview_clips_to_configured_cell_budget(monkeypatch):
    from hermes_cli.cli_render import _cells
    import agent.display as disp
    monkeypatch.setattr(disp, "get_tool_preview_max_len", lambda: 30)
    s = _started("write_file", preview="実行: デバイス変更 書き込み 45KB 🚀 長いう long tail suffix")
    phase = t("cli.stream.phase_writing")
    assert phase in s._spinner_text
    label = s._spinner_text.split(" · ", 1)[1]
    assert _cells(label) <= 30, "preview must be clipped to the budget in cells, not chars"
