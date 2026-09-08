"""Host transforms and streaming deadlines must not strand native sessions."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("directsdk_replay", ROOT / "plugins/model-providers/claude-oauth-directsdk/directsdk.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


def test_host_history_edits_replay_canonical_visible_blocks():
    from agent.context_compressor import ContextCompressor

    msg = {"role": "assistant", "content": "Writing.", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "x", "content": "a" * 3000})}}]}
    blocks = [{"type": "thinking", "thinking": "private", "signature": "opaque"}, {"type": "text", "text": "Writing."}, {"type": "tool_use", "id": "t1", "name": native.PREFIX + "write_file", "input": {"path": "x", "content": "a" * 3000}}]
    msg["reasoning_details"] = [{"type": native.CARRIER, "version": 1, "projection": native.projection(msg), "messages": [{"role": "assistant", "content": blocks}]}]
    history = [{"role": "user", "content": "write"}, msg, {"role": "tool", "tool_call_id": "t1", "content": "ok"}]
    original = copy.deepcopy(history)
    assert native.prepare_history(history)[1][1]["message"]["content"] == blocks
    assert ContextCompressor._truncate_tool_call_args_at(history, 1)
    frame = native.prepare_history(history)[1][1]["message"]
    assert [b["type"] for b in frame["content"]] == ["text", "tool_use"]
    assert frame["content"][-1]["input"] == json.loads(history[1]["tool_calls"][0]["function"]["arguments"])
    history[1]["content"] = "Hook replacement"
    assert native.prepare_history(history)[1][1]["message"]["content"][0]["text"] == "Hook replacement"
    assert history[1]["reasoning_details"] == original[1]["reasoning_details"]
    history[1]["reasoning_details"][0]["version"] = 999
    with pytest.raises(ValueError, match="version"):
        native.prepare_history(history)


@pytest.mark.skipif(os.name != "posix", reason="Native transport is POSIX")
def test_active_stream_outlives_idle_budget_and_large_request_uses_files(tmp_path):
    script = tmp_path / "native.py"
    script.write_text('''import json, pathlib, sys, time
settings = json.loads(pathlib.Path(sys.argv[sys.argv.index('--settings')+1]).read_text())
body = json.loads(settings['env']['CLAUDE_CODE_EXTRA_BODY'])
assert len(body['tools'][0]['description']) == 150000
assert 'temperature' not in body and 'top_p' not in body
assert body['output_config']['effort'] == 'low'
assert len(pathlib.Path(sys.argv[sys.argv.index('--system-prompt-file')+1]).read_text()) == 150000
for line in sys.stdin: pass
for i in range(8):
 print(json.dumps({'type':'stream_event','event':{'type':'content_block_delta','delta':{'type':'text_delta','text':'x'}}}),flush=True)
 time.sleep(.1)
print(json.dumps({'type':'assistant','message':{'role':'assistant','content':[{'type':'text','text':'x'*8}],'id':'m','stop_reason':'end_turn'}}),flush=True)
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}),flush=True)
print(json.dumps({'type':'result','subtype':'success','usage':{'input_tokens':1,'output_tokens':8}}),flush=True)
''')
    client = native.Client(command=[sys.executable, str(script)], env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)})
    request = dict(model="sonnet", messages=[{"role": "system", "content": "s" * 150000}, {"role": "user", "content": "go"}], tools=[{"type": "function", "function": {"name": "probe", "description": "d" * 150000}}], temperature=.3, top_p=.9, extra_body={"reasoning": {"effort": "minimal"}}, timeout=SimpleNamespace(read=.5), stream=True)
    try:
        chunks = list(client.create(**request))
        assert "".join(c.choices[0].delta.content or "" for c in chunks) == "x" * 8
        script.write_text("import time\ntime.sleep(60)\n")
        with pytest.raises(TimeoutError):
            list(client.create(**request))
        assert not client._requests
    finally:
        client.close()
