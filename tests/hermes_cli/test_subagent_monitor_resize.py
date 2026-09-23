"""A queued composer resize must not erase the alternate-screen monitor."""
from types import SimpleNamespace


def test_debounced_resize_does_not_write_over_subagent_monitor():
    from hermes_cli.cli_terminal_mixin import CLITerminalMixin
    writes = []
    cli = SimpleNamespace(
        _subagent_monitor=SimpleNamespace(opening=True),
        _get_tui_terminal_width=lambda: 32,
        _last_resize_width=90,
        _clear_prompt_toolkit_screen=lambda *a, **kw: writes.append('erase'),
        _redraw_rebuilds_scrollback=lambda: False,
        _pet_queue_kitty_frame=lambda: None,
        _schedule_status_bar_unsuppress=lambda app: writes.append('unsuppress'),
    )
    CLITerminalMixin._recover_after_resize(cli, SimpleNamespace(), lambda: writes.append('resize'))
    assert writes == []
