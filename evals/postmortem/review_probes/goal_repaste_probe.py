"""#103553: /goal re-paste vs option-selecting fragment

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted here).
It reproduced a defect in the first version of the PR; the fixed head must pass it. Paths are taken
from the command line / environment, never hard-coded. Usage: see the argument parsing at the top of the file.
"""
import asyncio
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import queue
import socket
import sqlite3
import sys
import tempfile
import threading
from unittest.mock import patch

repo, tag = sys.argv[1:3]
sys.path.insert(0, repo)
# Remove all inherited Hermes/config and credential env before real imports.
for key in list(os.environ):
    if key.startswith('HERMES_') or key.endswith(('_API_KEY', '_TOKEN')):
        os.environ.pop(key, None)
home = tempfile.TemporaryDirectory(prefix='review-goaldup-')
os.environ['HERMES_HOME'] = home.name
os.environ['NO_PROXY'] = '*'
os.environ['TZ'] = 'UTC'
# Fail closed: these probes must never invoke a provider or external network.
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('network prohibited in review probe'))
from cli import HermesCLI
from hermes_cli import cli_commands_mixin, goals
from gateway.slash_commands_goals import GatewayGoalCommandsMixin
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from tui_gateway import server

assert str(Path(cli_commands_mixin.__file__).resolve()).startswith(repo)
assert str(Path(goals.__file__).resolve()).startswith(repo)
server._hermes_home = Path(home.name)
goals._DB_CACHE.clear()
goals._get_session_db()
source = sqlite3.connect('file:/tmp/rf/state_copy.db?mode=ro', uri=True)
source.row_factory = sqlite3.Row
original = source.execute('SELECT content FROM messages WHERE id=264820').fetchone()['content']
repeated = source.execute('SELECT content FROM messages WHERE id=267045').fetchone()['content']
assert original == repeated
history = [{'role': 'user', 'content': original}, {'role': 'assistant', 'content': 'The wave is running; I will pick this up when the workers report back.'}]
output = {'tag': tag, 'modules': [cli_commands_mixin.__file__, goals.__file__, server.__file__], 'source_equal': original == repeated, 'original_chars': len(original)}

def make_cli(hist, sid):
    c = HermesCLI.__new__(HermesCLI)
    c.session_id = sid
    c.agent = None
    c.conversation_history = copy.deepcopy(hist)
    c._pending_input = queue.Queue()
    return c

def cli_case(hist, goal, sid):
    c = make_cli(hist, sid)
    before = copy.deepcopy(c.conversation_history)
    with contextlib.redirect_stdout(io.StringIO()):
        assert c.process_command('/goal ' + goal)
    prompt = c._pending_input.get_nowait()
    state = goals.GoalManager(sid).state
    assert c.conversation_history == before
    assert c._pending_input.empty()
    assert state.goal == goal
    return {'prompt': prompt, 'prompt_chars': len(prompt), 'state_goal_preserved': state.goal == goal,
            'history_unchanged': c.conversation_history == before,
            'continuation_still_contains_full_goal': goal in goals.GoalManager(sid).next_continuation_prompt()}

output['cli_literal_witness'] = cli_case(history, original, tag+'-cli')
output['cli_empty_control'] = cli_case([], 'Ship the release', tag+'-empty')
output['cli_unrelated_control'] = cli_case([{'role':'user','content':'Unrelated'}], original, tag+'-unrelated')
output['cli_block_content'] = cli_case([{'role':'user','content':[{'type':'text','text': original}]}], original, tag+'-block')
options = [{'role':'user','content':'We can ship the API or ship the UI. Wait for my choice.'}, {'role':'assistant','content':'Which one should I work on?'}]
output['selection_api'] = cli_case(options, 'ship the API', tag+'-api')
output['selection_ui'] = cli_case(options, 'ship the UI', tag+'-ui')
output['different_selected_goals_same_model_prompt'] = output['selection_api']['prompt'] == output['selection_ui']['prompt']
# /goal draft invokes its only paid dependency as an explicit unavailable stub.
c = make_cli(history, tag+'-draft')
with patch('hermes_cli.goals.draft_contract', return_value=None), contextlib.redirect_stdout(io.StringIO()):
    assert c.process_command('/goal draft ' + original)
output['draft_fallback_prompt'] = c._pending_input.get_nowait()

# Actual registered TUI/Desktop command dispatcher; goal bypasses CLI slash worker.
sid = tag+'-tui'
server._sessions[sid] = {'session_key': sid, 'history': copy.deepcopy(history), 'history_lock': threading.Lock(), 'history_version':0, 'running':False, 'attached_images': [], 'cols': 120}
rpc = server._methods['slash.exec'](1, {'command':'goal '+original, 'session_id':sid})
output['tui_slash_rpc'] = rpc
assert goals.GoalManager(sid).state.goal == original

# Real gateway handler + real enqueue method. Capture only adapter transport FIFO seam.
class GatewayProbe(GatewayGoalCommandsMixin):
    def __init__(self):
        self.mgr = goals.GoalManager(tag+'-gateway')
        self.events = []
        self.conversation_history = copy.deepcopy(history)
    async def _get_goal_manager_for_event(self, event):
        return self.mgr, None
    def _adapter_and_key_for(self, event):
        return object(), 'probe-key'
    def _enqueue_fifo(self, key, event, adapter):
        self.events.append(event)
gw = GatewayProbe()
event = MessageEvent(text='/goal '+original, message_type=MessageType.TEXT,
                     source=SessionSource(platform=Platform.DISCORD, chat_id='probe', chat_type='dm', user_id='probe'), message_id='goal-probe')
asyncio.run(gw._handle_goal_command(event))
assert len(gw.events) == 1
output['gateway_kick'] = {'prompt':gw.events[0].text, 'goal_preserved':gw.mgr.state.goal == original}
# Actual archived replay turn, including useful actions and terminal waits.
next_id = source.execute("SELECT min(id) FROM messages WHERE session_id=? AND id>? AND role='user'", ('20260902_073639_918cf3',267045)).fetchone()[0]
rows = source.execute('SELECT id,role,content,tool_calls,tool_name,timestamp FROM messages WHERE session_id=? AND id>=? AND id<? ORDER BY id', ('20260902_073639_918cf3',267045,next_id)).fetchall()
output['archive_replay_turn'] = {'first_id':rows[0]['id'], 'last_id':rows[-1]['id'], 'elapsed_seconds':rows[-1]['timestamp']-rows[0]['timestamp'], 'assistant_messages':sum(r['role']=='assistant' for r in rows), 'tool_messages':sum(r['role']=='tool' for r in rows), 'first_assistant':next(r['content'] for r in rows if r['role']=='assistant'), 'tools':[{k:r[k] for k in ['id','tool_name','timestamp']} for r in rows if r['role']=='tool']}
try:
    import tiktoken
    enc=tiktoken.get_encoding('cl100k_base')
    output['illustrative_cl100k_tokens']={'original':len(enc.encode(original)), 'kick':len(enc.encode(output['cli_literal_witness']['prompt']))}
except Exception as exc:
    output['tokenizer_unavailable']=str(exc)
out=Path(os.environ.get('PROBE_OUT','.'))/f'goaldup-{tag}-probe.json'
out.write_text(json.dumps(output, indent=2), encoding='utf-8')
print(json.dumps({'artifact':str(out), 'tag':tag, 'cli_chars':output['cli_literal_witness']['prompt_chars'], 'tui_chars':len(rpc['result']['message']), 'gateway_chars':len(gw.events[0].text), 'ambiguous_selection':output['different_selected_goals_same_model_prompt'], 'archive':output['archive_replay_turn'], 'tokens':output.get('illustrative_cl100k_tokens')}, indent=2))
server._sessions.clear()
