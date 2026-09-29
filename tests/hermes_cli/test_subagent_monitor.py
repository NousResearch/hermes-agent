"""The dock is bounded, session-owned, and only repaints changing live work."""
from types import SimpleNamespace


def test_dock_scopes_children_and_fits_short_narrow_terminal(monkeypatch):
    from hermes_cli import cli_subagent_monitor as monitor
    from tools import delegate_tool_registry as registry
    owner = SimpleNamespace(session_id='owner')
    monkeypatch.setattr(registry, '_active_subagents', {})
    for i in range(8):
        registry._register_subagent(dict(subagent_id=str(i), owner_agent_session_id='owner',
            goal='Check 界' * 20, started_at=10, status='running', last_tool='read_file'))
    registry._register_subagent(dict(subagent_id='foreign', owner_agent_session_id='other', goal='SECRET'))
    dock = monitor.SubagentMonitor(SimpleNamespace(agent=owner))
    assert dock.refresh(now=20)
    text = dock.dock_text(columns=32, rows=14)
    from prompt_toolkit.utils import get_cwidth
    assert '8 live' in text and 'Ctrl+T' in text and 'read_file' in text
    assert 'SECRET' not in text
    assert len(text.splitlines()) <= 3
    assert all(get_cwidth(line) <= 32 for line in text.splitlines())
    assert not dock.refresh(now=20)
    registry._active_subagents.clear()
    assert dock.refresh(now=21)
    assert dock.dock_text(columns=32, rows=14) == ''
    assert not dock.refresh(now=22)
    from hermes_cli.cli_tui_mixin import CLITuiMixin
    cli = SimpleNamespace(_subagent_dock_widget='dock', _get_extra_tui_widgets=lambda: [])
    children = CLITuiMixin._build_tui_layout_children(cli, sudo_widget=None, secret_widget=None,
        approval_widget=None, clarify_widget=None, spacer='spacer', status_bar='status',
        input_rule_top='top', image_bar=None, input_area='composer', input_rule_bot='bottom',
        voice_status_bar=None, completions_menu=None)
    assert children.index('dock') < children.index('status') < children.index('composer')


def test_monitor_controls_recheck_ownership_and_keep_selection(monkeypatch):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from tools import delegate_tool_registry as registry
    monkeypatch.setattr(registry, '_active_subagents', {})
    delivered = []
    child = SimpleNamespace(steer=lambda text: delivered.append(text) or True)
    owner = SimpleNamespace(session_id='owner')
    registry._register_subagent(dict(subagent_id='mine', owner_agent_session_id='owner',
        goal='Check module', started_at=10, status='running', agent=child))
    dock = SubagentMonitor(SimpleNamespace(agent=owner))
    dock.refresh(now=20)
    dock.select(1)
    assert dock.selected['subagent_id'] == 'mine'
    assert dock.control('steer', 'Focus on tests')['status'] == 'queued'
    assert delivered == ['Focus on tests']
    owner.session_id = 'different'
    assert 'error' in dock.control('steer', 'must not arrive')
    assert delivered == ['Focus on tests']
