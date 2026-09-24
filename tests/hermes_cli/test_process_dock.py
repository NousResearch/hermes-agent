"""Background processes share the classic CLI live-work dock with subagents (Processes block)."""
import time
from types import SimpleNamespace



def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)




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
