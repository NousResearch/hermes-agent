"""One upstream admission and first-response authority over native recovery."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'plugins/model-providers/claude-oauth-directsdk'))
import directsdk

NATIVE = r'''
import json, os, sys, urllib.request, urllib.error
for line in sys.stdin:
    frame=json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type':'result','num_turns':0}),flush=True)
        continue
    break
url=os.environ['ANTHROPIC_BASE_URL']+'/v1/messages'
for _ in range(2):
    try:
        urllib.request.urlopen(urllib.request.Request(url,data=b'{}',headers={'Content-Type':'application/json'}),timeout=5).read()
    except urllib.error.HTTPError:
        break
print(json.dumps({'type':'assistant','message':{'id':'first','role':'assistant','content':[{'type':'text','text':'FIRST'}]}}))
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}))
print(json.dumps({'type':'result','subtype':'success','usage':{'input_tokens':0,'output_tokens':0}}))
'''


@pytest.mark.parametrize('stop', ['end_turn', 'max_tokens', 'model_context_window_exceeded'])
def test_first_response_owns_usage_and_stops_recovery(tmp_path, stop):
    calls = []
    usage = {'input_tokens':0, 'output_tokens':0, 'cache_read_input_tokens':0, 'cache_creation_input_tokens':0}
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            events = [
                {'type':'message_start','message':{'id':'first','role':'assistant','model':'sonnet','content':[], 'usage':usage}},
                {'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}},
                {'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'FIRST'}},
                {'type':'content_block_stop','index':0},
                {'type':'message_delta','delta':{'stop_reason':stop},'usage':usage},
                {'type':'message_stop'},
            ]
            self.wfile.write(''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode())
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE)
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        result=client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}])
        assert len(calls)==1
        assert result.choices[0].message.content=='FIRST'
        assert result.choices[0].finish_reason==('stop' if stop=='end_turn' else 'length')
        assert result.usage.prompt_tokens==0
        assert result.choices[0].message.reasoning_details[0]['messages'][0]['stop_reason']==stop
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_cancel_closes_the_active_upstream_socket(tmp_path):
    entered, disconnected = threading.Event(), threading.Event()
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            entered.set()
            self.rfile.read(1)
            disconnected.set()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    native = tmp_path / 'native.py'
    native.write_text(NATIVE)
    client = directsdk.Client(command=[sys.executable, str(native)], env={'PATH':os.defpath, 'HOME':str(tmp_path), 'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(client.create, model='sonnet', messages=[{'role':'user', 'content':'fixture'}])
            try:
                assert entered.wait(5)
            finally:
                client.cancel()
            with pytest.raises(RuntimeError, match='cancelled'):
                result.result(timeout=3)
            assert disconnected.wait(2)
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()
