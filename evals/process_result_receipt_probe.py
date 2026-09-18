"""Live process I/O proof, without model calls: python probe.py REPO OUT_DIR."""
import json
import os
from pathlib import Path
import subprocess
import sys

repo, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
producer = '''
import json, shlex, sys
from gateway.session_context import scoped_current_session_id
from tools.terminal_tool import terminal_tool
from tools.process_registry import process_registry
from hermes_cli.oneshot import _linger_for_background_completions
with scoped_current_session_id("receipt-owner"):
    code = "import sys,time; time.sleep(.2); print('RECEIPT_STDOUT'); print('RECEIPT_STDERR',file=sys.stderr); sys.exit(7)"
    result = json.loads(terminal_tool(shlex.join([sys.executable, '-c', code]), background=True,
        notify_on_complete=True, task_id='receipt-task'))
    _linger_for_background_completions()
    session = process_registry.get(result['session_id'])
    session._reader_thread.join(timeout=10)
    print(json.dumps({'spawn': result, 'live': process_registry.read_log(session.id)}))
'''
consumer = '''
import json,sys
from gateway.session_context import scoped_current_session_id
from tools.process_registry import process_registry
with scoped_current_session_id(sys.argv[2]):
    print(json.dumps({'log':process_registry.read_log(sys.argv[1]),
                     'poll':process_registry.poll(sys.argv[1]),
                     'replayed':process_registry.completion_queue.qsize()}))
'''
env = {key: value for key, value in os.environ.items()
       if not key.startswith(('HERMES_', 'OPENAI_', 'ANTHROPIC_', 'TERMINAL_'))
       and not key.endswith(('_API_KEY', '_TOKEN', '_SECRET'))}
env.update(HOME=str(out), HERMES_HOME=str(out / 'profile'), PYTHONPATH=str(repo),
           TERMINAL_CWD=str(out), PYTHONDONTWRITEBYTECODE='1')

def run(code, *args, profile=None):
    child_env = dict(env)
    if profile:
        child_env['HERMES_HOME'] = str(out / profile)
    result = subprocess.run([sys.executable, '-c', code, *args], cwd=repo,
                            env=child_env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, encoding='utf-8', timeout=45)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout.splitlines()[-1])

before_exit = run(producer)
sid = before_exit['spawn']['session_id']
result = {'repo': str(repo), 'before_exit': before_exit,
          'owner': run(consumer, sid, 'receipt-owner'),
          'stranger': run(consumer, sid, 'receipt-stranger'),
          'unbound': run(consumer, sid, ''),
          'other_profile': run(consumer, sid, 'receipt-owner', profile='other-profile')}
(out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
