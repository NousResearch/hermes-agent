"""Background processes share the classic CLI live-work dock with subagents (Processes block)."""
import time
from types import SimpleNamespace

from prompt_toolkit.utils import get_cwidth

from agent.i18n import t


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def test_dock_paints_processes_under_agents_and_retires_finished_rows(monkeypatch):
    from hermes_cli import cli_process_dock
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from tools import delegate_tool_registry as registry
    from tools.process_registry import process_registry

    monkeypatch.setattr(registry, '_active_subagents', {})
    owner = SimpleNamespace(session_id='owner')
    registry._register_subagent(dict(subagent_id='a1', owner_agent_session_id='owner',
        goal='Check module', started_at=time.time() - 5, status='running', last_tool='read_file'))
    quick = process_registry.spawn_local(command="echo hello-dock; exit 3", cwd='.', task_id='t', owner_task_id='t', session_key='')
    slow = process_registry.spawn_local(command="sleep 30", cwd='.', task_id='t', owner_task_id='t', session_key='')
    quick_id, slow_id = quick.id, slow.id
    try:
        _wait(lambda: process_registry.get(quick_id).exited)
        process_registry.list_sessions()  # observes the exit → exited_at stamped
        dock = SubagentMonitor(SimpleNamespace(agent=owner))
        assert dock.refresh()
        text = dock.dock_text(columns=100, rows=30)
        lines = text.splitlines()
        # The pill counts each kind; agents list before processes, each row keeps its activity.
        assert t('cli.subagents.pill_agents_one', count=1) in lines[0]
        assert t('cli.subagents.pill_shells_one', count=1) in lines[0]
        assert t('cli.subagents.count_done', count=1) in lines[0]
        agents_at = next(i for i, line in enumerate(lines) if 'Check module' in line)
        procs_at = next(i for i, line in enumerate(lines) if 'sleep 30' in line)
        assert agents_at < procs_at
        assert any('⚙ sleep 30' in line and t('cli.dock.starting') in line for line in lines)
        assert any('✘ echo hello-dock; exit 3' in line and t('cli.dock.exit_code', code=3) in line for line in lines)
        assert all(get_cwidth(line) <= 100 for line in lines)
        # A short narrow viewport still names the live work in its one row and stays in bounds.
        narrow = dock.dock_text(columns=40, rows=14).splitlines()
        assert any('Check module' in line for line in narrow)
        assert all(get_cwidth(line) <= 40 for line in narrow)
        dock.collapsed = True
        collapsed = dock.dock_text(columns=100, rows=30)
        assert collapsed.count('\n') == 0
        assert t('cli.subagents.pill_agents_one', count=1) in collapsed and t('cli.subagents.pill_shells_one', count=1) in collapsed
        # Finished rows leave after the retention window; running ones stay.
        later = cli_process_dock.process_rows(time.time() + cli_process_dock.RETAIN_SECONDS + 1)
        assert [r['id'] for r in later] == [slow_id]
    finally:
        process_registry.kill_process(slow_id)


def test_monitor_controls_stop_processes_and_never_steer_them():
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from tools.process_registry import process_registry

    slow = process_registry.spawn_local(command="sleep 30", cwd='.', task_id='t', owner_task_id='t', session_key='')
    slow_id = slow.id
    try:
        dock = SubagentMonitor(SimpleNamespace(agent=None))
        dock.refresh()
        dock.selected_id = slow_id
        assert dock.selected_process is not None
        assert 'error' in dock.control('steer', 'nope')
        assert process_registry.get(slow_id).exited is False
        assert dock.control('stop')['status'] == 'killed'
        _wait(lambda: process_registry.get(slow_id).exited)
        dock.refresh()
        assert any(r['id'] == slow_id and r['status'] == 'killed' for r in dock.processes)
    finally:
        process_registry.kill_process(slow_id)
