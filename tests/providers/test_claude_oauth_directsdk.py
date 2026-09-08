import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2] / "plugins" / "model-providers" / "claude-oauth-directsdk"
sys.path.insert(0, str(ROOT))

FAKE = r"""
import json, os, sys, time
if os.environ.get('PID_FILE'):
 open(os.environ['PID_FILE'],'w').write(str(os.getpid()))
if '--version' in sys.argv:
 print('2.1.263 (Claude Code)'); sys.exit()
rows=[]
for line in sys.stdin:
 r=json.loads(line); rows.append(r)
 if r.get('shouldQuery') is False:
  print(json.dumps({'type':'result','num_turns':0,'is_error':False}),flush=True)
if os.environ.get('HANG'):
 print(json.dumps({'type':'stream_event','event':{'type':'content_block_delta','delta':{'type':'text_delta','text':'started'}}}),flush=True)
 time.sleep(60)
wire=json.loads(os.environ['CLAUDE_CODE_EXTRA_BODY'])
assert wire['tools'][0]['description'].endswith('TAIL')
assert '--max-turns' in sys.argv and sys.argv[sys.argv.index('--max-turns')+1]=='1'
assert sys.argv[sys.argv.index('--permission-mode')+1]=='dontAsk'
assert sys.argv[sys.argv.index('--tools')+1]==''
assert rows[-1]['type']=='user'
assert 'metadata' not in wire
blocks=[{'type':'thinking','thinking':'private','signature':'signed-test'}, {'type':'text','text':'hello\n'}, {'type':'tool_use','id':'toolu_test','name':'mcp__hermes__probe','input':{'value':'x'}}]
if len(rows)>1:
 assert rows[1]['message']['content']==blocks
 blocks=[{'type':'text','text':'done'}]
for b in blocks:
 if b['type']=='text':
  print(json.dumps({'type':'stream_event','event':{'type':'content_block_delta','delta':{'type':'text_delta','text':b['text']}}}),flush=True)
print(json.dumps({'type':'assistant','message':{'role':'assistant','content':blocks,'id':'msg_test','model':'sonnet','stop_reason':'tool_use' if len(blocks)>1 else 'end_turn'}}),flush=True)
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}),flush=True)
u={'input_tokens':3,'output_tokens':5,'cache_read_input_tokens':7,'cache_creation_input_tokens':11}
print(json.dumps({'type':'result','num_turns':2 if len(blocks)>1 else 1,'subtype':'error_max_turns' if len(blocks)>1 else 'success','is_error':len(blocks)>1,'usage':u}),flush=True)
sys.exit(1 if len(blocks)>1 else 0)
"""


@unittest.skipUnless(os.name == "posix", "Native process-group transport requires POSIX")
class Contract(unittest.TestCase):
    def client(self, tmp, **kw):
        import directsdk

        script = Path(tmp) / "native.py"
        script.write_text(FAKE)
        return directsdk.Client(
            command=[sys.executable, str(script)],
            env={"PATH": os.environ["PATH"], "HOME": tmp, **kw},
        )

    def request(self) -> dict:
        return dict(
            model="sonnet",
            messages=[{"role": "user", "content": "go"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "probe",
                        "description": "long " * 600 + "TAIL",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                    },
                }
            ],
        )

    def test_canonical_request_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(tmp)
            self.assertEqual(client.api_key, "external-process")
            self.assertEqual(client.base_url, "process://claude-oauth-directsdk")
            for streaming in (False, True):
                req = self.request()
                req["timeout"] = SimpleNamespace(read=10)
                result = client.chat.completions.create(**req, stream=streaming)
                if streaming:
                    chunks = list(result)
                    self.assertEqual(
                        "".join(
                            c.choices[0].delta.content or ""
                            for c in chunks
                            if c.choices
                        ),
                        "hello\n",
                    )
                    final = chunks[-1]
                    msg = {
                        "role": "assistant",
                        "content": "hello",
                        "tool_calls": [
                            {
                                "id": "toolu_test",
                                "type": "function",
                                "function": {
                                    "name": "probe",
                                    "arguments": '{"value":"x"}',
                                },
                            }
                        ],
                        "reasoning_details": final.choices[0].delta.reasoning_details,
                    }
                    self.assertEqual(final.choices[0].finish_reason, "tool_calls")
                else:
                    final = result
                    msg = result.choices[0].message.model_dump()
                    self.assertEqual(msg["tool_calls"][0]["function"]["name"], "probe")
                self.assertEqual(final.usage.prompt_tokens, 21)
                self.assertEqual(final.usage.completion_tokens, 5)
                msg["content"] = (msg.get("content") or "").strip()
                req["messages"] += [
                    msg,
                    {
                        "role": "tool",
                        "tool_call_id": "toolu_test",
                        "content": " host\n result",
                    },
                ]
                self.assertEqual(
                    client.chat.completions.create(**req).choices[0].message.content,
                    "done",
                )
                msg["content"] = "middleware changed"
                with self.assertRaisesRegex(ValueError, "modified"):
                    client.chat.completions.create(**req)
            client.close()

    def test_fail_closed_and_cancellation(self):
        import directsdk

        disabled = json.loads(
            directsdk.request_body(
                {**self.request(), "extra_body": {"reasoning": {"enabled": False}}}
            )[0]
        )
        self.assertEqual(disabled["thinking"], {"type": "disabled"})
        self.assertEqual(disabled["context_management"], {"edits": []})
        effort = json.loads(
            directsdk.request_body(
                {**self.request(), "extra_body": {"reasoning": {"effort": "low"}}}
            )[0]
        )
        self.assertEqual(effort["output_config"], {"effort": "low"})
        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        }
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": "title", "strict": True, "schema": schema},
        }
        formatted = json.loads(
            directsdk.request_body(
                {**self.request(), "extra_body": {"response_format": fmt}}
            )[0]
        )
        self.assertEqual(
            formatted["output_config"]["format"],
            {"type": "json_schema", "schema": schema},
        )
        routed = directsdk.Client(
            env={"CLAUDE_OAUTH_DIRECTSDK_COMMAND": "/native/test"}
        )
        self.assertEqual(routed.command, ["/native/test"])
        from unittest.mock import patch

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-conflicting-key"}):
            with self.assertRaisesRegex(ValueError, "OAuth"):
                directsdk.Client(command="/does/not/exist").create(**self.request())
        client = directsdk.Client(command="/does/not/exist", env={})
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            client.chat.completions.create(**self.request())
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(tmp)
            for bad in (
                {"extra_body": {"metadata": {}}},
                {"extra_body": []},
                {"temperature": float("nan")},
                {"tool_choice": "required"},
                {"n": 2},
            ):
                with self.assertRaises(ValueError):
                    client.chat.completions.create(**self.request(), **bad)

            async def run():
                result = await client.chat.completions.create(**self.request())
                self.assertEqual(result.choices[0].finish_reason, "tool_calls")

            asyncio.run(run())
            hanging = self.client(tmp, HANG="1")
            stream = hanging.chat.completions.create(**self.request(), stream=True)
            self.assertEqual(next(stream).choices[0].delta.content, "started")
            hanging.cancel()
            with self.assertRaisesRegex(RuntimeError, "cancel"):
                list(stream)
            hanging.close()
            # Closing a paused stream must reap without asking for another chunk.
            import time

            pidfile = Path(tmp) / "pid"
            hanging = self.client(tmp, HANG="1", PID_FILE=str(pidfile))
            paused = hanging.chat.completions.create(**self.request(), stream=True)
            next(paused)
            pid = int(pidfile.read_text())
            process = paused.request.process
            paused.close()
            self.assertTrue(process.stdout.closed)
            self.assertEqual(len(hanging._requests), 0)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            else:
                self.fail("Closed paused stream left its native child unreaped")
            hanging.close()
            unstarted = self.client(tmp)
            stream = unstarted.create(**self.request(), stream=True)
            unstarted.close()
            self.assertEqual(len(unstarted._requests), 0)


if __name__ == "__main__":
    unittest.main()
