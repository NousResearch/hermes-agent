import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERMES = str(ROOT)


def run_isolated(tmp_path, code, enabled=True):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: " + ("[subagent-progress]" if enabled else "[]") + "\n")
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": HERMES}
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_real_plugin_discovery_and_hooks(tmp_path):
    output = run_isolated(tmp_path, """
import json
import model_tools
from tools.registry import registry
from hermes_cli.lifecycle import invoke_hook
from hermes_constants import get_hermes_home
entry = registry.get_entry('report_progress')
assert entry is not None
assert entry.toolset == 'subagent_progress'
assert registry.get_entry('review_subagent_progress') is not None
invoke_hook('subagent_start', parent_session_id='p', child_session_id='c', child_subagent_id='s', child_goal='g')
import sqlite3
with sqlite3.connect(get_hermes_home()/'state/subagent-progress.sqlite3') as db:
    assert db.execute('SELECT parent FROM children WHERE subagent=?',('s',)).fetchone()[0] == 'p'
assert not json.loads(entry.handler({'completed':'x','next_step':'y'}))['success']
print('real_discovery_and_hook_ok')
""")
    assert "real_discovery_and_hook_ok" in output


def test_disabled_profile_has_no_tool_or_state(tmp_path):
    output = run_isolated(tmp_path, """
import model_tools
from tools.registry import registry
from hermes_constants import get_hermes_home
assert registry.get_entry('report_progress') is None
assert registry.get_entry('review_subagent_progress') is None
assert not (get_hermes_home()/'state/subagent-progress.sqlite3').exists()
print('disabled_profile_unchanged')
""", enabled=False)
    assert "disabled_profile_unchanged" in output


def test_real_discovery_keeps_profile_state_isolated_a_b_a(tmp_path):
    output = run_isolated(tmp_path, """
import sqlite3
from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
from hermes_cli.plugins import get_plugin_manager
from hermes_cli.lifecycle import invoke_hook

base = get_hermes_home()
profiles = [base / 'profile-a', base / 'profile-b']
managers = []
for home in profiles:
    home.mkdir()
    (home / 'config.yaml').write_text('plugins: {enabled: [subagent-progress]}')
for home, sid in [(profiles[0], 'a'), (profiles[1], 'b'), (profiles[0], 'a-again')]:
    token = set_hermes_home_override(home)
    try:
        manager = get_plugin_manager()
        manager.discover_and_load()
        managers.append(manager)
        invoke_hook('subagent_start', parent_session_id=sid, child_session_id='child-' + sid,
                    child_subagent_id='sub-' + sid, child_goal='test goal')
    finally:
        reset_hermes_home_override(token)
assert managers[0] is managers[2] and managers[0] is not managers[1]
for home, expected in [(profiles[0], {'a', 'a-again'}), (profiles[1], {'b'})]:
    with sqlite3.connect(home / 'state/subagent-progress.sqlite3') as db:
        assert {row[0] for row in db.execute('SELECT parent FROM children')} == expected
print('profile_state_isolated')
""")
    assert 'profile_state_isolated' in output


def test_plugin_injection_queues_busy_adapter_without_interrupt(tmp_path):
    output = run_isolated(tmp_path, """
import asyncio
from types import SimpleNamespace
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.config import Platform
from unittest.mock import AsyncMock

async def main():
    # Exercise the installed adapter's busy-event handler, not a parallel imitation.
    adapter = SimpleNamespace(name='test', _busy_session_handler=None, _pending_messages={},
        _is_queue_text_debounce_candidate=lambda event: False, _canonicalize=lambda source: None)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='1', chat_type='dm', user_id='1')
    event = MessageEvent(text='[tool] checkpoint decision', message_type=MessageType.TEXT,
                        source=source, internal=True, allow_gateway_control=False)
    await BasePlatformAdapter._handle_message_while_active(adapter,event,'session')
    assert event._gateway_accepted is True
    assert adapter._pending_messages['session'].text == event.text
    print('busy_internal_event_queued')
asyncio.run(main())
""")
    assert "busy_internal_event_queued" in output
