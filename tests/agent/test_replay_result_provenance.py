"""Current tool messages distinguish literal output from legacy bare interrupts."""

from pathlib import Path
import subprocess
import sys
import textwrap


def _run_fixture(tmp_path, body):
    root = Path(__file__).resolve().parents[2]
    setup = '''
import json, os, subprocess, sys
from pathlib import Path
home = Path(sys.argv[1])
for key in ('HOME', 'USERPROFILE', 'HERMES_HOME', 'LOCALAPPDATA', 'APPDATA'):
    os.environ[key] = str(home)
os.environ['HERMES_ENABLE_PROJECT_PLUGINS'] = 'false'
sys.path.insert(0, sys.argv[2])
os.chdir(home)
'''
    shared = '''
from agent.replay_cleanup import sanitize_replay_history
from agent.session_persistence import SessionPersistenceMixin
from agent.tool_dispatch_helpers import make_tool_result_message
from agent.vision_message_prep import VisionMessagePrepMixin
from hermes_state import SessionDB
from tools.tool_result_storage import maybe_persist_tool_result

def replay(name, result, session, legacy=False):
    result = maybe_persist_tool_result(result, name, session)
    result = VisionMessagePrepMixin()._tool_result_content_for_active_model(name, result)
    message = make_tool_result_message(name, result, session)
    if legacy:
        message.pop('tool_result_format', None)
    messages = [
        {'role': 'user', 'content': 'Inspect the local result.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': session, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}]},
        message, {'role': 'assistant', 'content': 'Result received.'},
    ]
    store = SessionPersistenceMixin()
    store._session_db = SessionDB(home / (session + '.db'))
    store.session_id = session
    store._session_db.create_session(session, source='cli')
    store._session_db_created = True
    store._last_flushed_db_idx = 0
    try:
        assert store._flush_messages_to_session_db(messages)
        loaded = store._session_db.get_messages_as_conversation(session)
        exported = store._session_db.export_session(session)
        imported = SessionDB(home / (session + '-import.db'))
        try:
            assert imported.import_sessions([exported])['ok']
            restored = imported.get_messages_as_conversation(session)
            assert sanitize_replay_history(restored) == sanitize_replay_history(loaded)
        finally:
            imported.close()
        assert loaded[2]['content'] == result
        cleaned = sanitize_replay_history(loaded)
        return loaded, cleaned
    finally:
        store._session_db.close()
'''
    script = tmp_path / 'fixture.py'
    # Plugin discovery must see the fixture before any product imports.
    before, after = body.split('# PRODUCT IMPORTS', 1)
    script.write_text(textwrap.dedent(setup) + textwrap.dedent(before) + shared + textwrap.dedent(after), encoding='utf-8')
    result = subprocess.run([sys.executable, '-I', str(script), str(tmp_path), str(root)],
                            capture_output=True, text=True, encoding='utf-8', timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_successful_plugin_literal_marker_survives_replay(tmp_path):
    _run_fixture(tmp_path, '''
(home / 'config.yaml').write_text('plugins:\\n  enabled: [replay-fixture]\\n', encoding='utf-8')
p = home / 'plugins' / 'replay-fixture'
p.mkdir(parents=True)
(p / 'plugin.yaml').write_text('name: replay-fixture\\nversion: 1.0.0\\ndescription: Read fixture logs\\n', encoding='utf-8')
(p / '__init__.py').write_text("from pathlib import Path\\ndef register(ctx):\\n    ctx.register_tool(name='fixture_log_read', toolset='replay-fixture', schema={'name':'fixture_log_read','description':'Read log','parameters':{'type':'object','properties':{'path':{'type':'string'}}}}, handler=lambda args, **kw: Path(args['path']).read_text(encoding='utf-8'))\\n", encoding='utf-8')
# PRODUCT IMPORTS
from model_tools import handle_function_call
log = home / 'application.log'
log.write_text('Documented executor marker example:\\n[Command interrupted]', encoding='utf-8')
result = handle_function_call('fixture_log_read', {'path': str(log)}, task_id='fixture', session_id='fixture', tool_call_id='c1')
assert result == log.read_text(encoding='utf-8')
loaded, cleaned = replay('fixture_log_read', result, 'success')
assert cleaned[2]['content'] == result, cleaned
assert cleaned == loaded
from agent.transports.chat_completions import ChatCompletionsTransport
wire = ChatCompletionsTransport().convert_messages(cleaned)
assert all('tool_result_format' not in msg for msg in wire)
''')


def test_real_executor_interrupt_and_legacy_bare_still_recover(tmp_path):
    _run_fixture(tmp_path, '''
# PRODUCT IMPORTS
from tools.environments.local import LocalEnvironment
from tools.interrupt import set_interrupt, clear_current_thread_interrupt
from tools.terminal_tool_result import finalize_foreground_result
# Spawn a real child without shell/session bootstrap or network dependencies.
env = object.__new__(LocalEnvironment)
proc = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(30)'],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        start_new_session=True)
try:
    set_interrupt(True)
    raw = env._wait_for_process(proc, timeout=10)
finally:
    clear_current_thread_interrupt()
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)
assert raw['returncode'] == 130, raw
result = finalize_foreground_result(command='sleep', result=raw, env=env, env_type='local',
    effective_task_id='fixture', task_id='fixture', session_id='fixture', session_key='fixture',
    workdir=str(home), command_cwd=str(home), approval_note=None)
for session, output, legacy in [('current', result, False), ('legacy', raw['output'], True)]:
    loaded, cleaned = replay('terminal', output, session, legacy=legacy)
    assert len(cleaned) == len(loaded)
    assert cleaned[2]['effect_disposition'] == 'unknown'
    assert 'UNKNOWN' in cleaned[2]['content']
# The generic executor has its own cancellation producer, not the legacy command marker.
from types import SimpleNamespace
from agent.tool_executor import _ToolCallRef, _unfinished_tool_result
cancelled, _, _ = _unfinished_tool_result(
    SimpleNamespace(_interrupt_requested=True),
    _ToolCallRef('fixture_log_read', {}, 'fixture', 'cancelled', []),
    timed_out=False, timeout_s=None)
loaded, cleaned = replay('fixture_log_read', cancelled, 'cancelled')
assert cleaned == loaded
assert cleaned[2]['content'] == cancelled
''')
