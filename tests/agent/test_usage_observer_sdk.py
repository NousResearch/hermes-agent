"""Physical HTTP attempts and lazy auxiliary stream completion through the real SDK."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from openai import OpenAI

from agent import auxiliary_hooks as aux
from agent import process_bootstrap as bootstrap


class SDKObserverTests(unittest.TestCase):
    def test_sdk_retry_and_stream_terminal_are_separate_from_logical_usage(self):
        from agent import usage_observer as observer

        events, calls = [], {}
        usage = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
                 'prompt_tokens_details': {'cached_tokens': 40},
                 'completion_tokens_details': {'reasoning_tokens': 7}}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                case = self.path.split('/')[1]
                calls[case] = calls.get(case, 0) + 1
                if case == 'retry' and calls[case] == 1:
                    data = json.dumps({'error': {'message': 'fixture retry', 'type': 'server_error'}}).encode()
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Retry-After', '0')
                elif case == 'retry':
                    data = json.dumps({'id': 'fixture', 'object': 'chat.completion', 'created': 0,
                        'model': 'fixture', 'usage': usage,
                        'choices': [{'index': 0, 'finish_reason': 'stop',
                                     'message': {'role': 'assistant', 'content': 'fixture'}}]}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                else:
                    final = json.dumps({'id': 'fixture', 'object': 'chat.completion.chunk',
                                        'created': 0, 'model': 'fixture', 'choices': [], 'usage': usage})
                    data = ('data: ' + ('{invalid' if case == 'broken' else final)
                            + '\n\ndata: [DONE]\n\n').encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host = 'http://127.0.0.1:' + str(server.server_address[1])
        def fire(event, **payload):
            events.append((event, payload))

        try:
            with patch.object(observer, '_enabled', return_value=True), patch.object(observer, '_fire', side_effect=fire), \
                    patch.object(aux, '_has_hook', return_value=True), patch.object(aux, '_fire', side_effect=fire):
                for case in ('retry', 'stream', 'broken'):
                    with self.subTest(case=case):
                        self._assert_case(case, host, events, calls)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            bootstrap.close_shared_transports()

    def _assert_case(self, case, host, events, calls):
        events.clear()
        with OpenAI(api_key='fixture-no-real-credential', base_url=host+'/'+case+'/v1',
                    max_retries=1 if case=='retry' else 0, timeout=3,
                    http_client=bootstrap.build_keepalive_http_client(host)) as client:
            streamed = case != 'retry'
            result = aux.run_with_aux_hooks(
                lambda: client.chat.completions.create(model='fixture',
                    messages=[{'role': 'user', 'content': 'fixture'}], stream=streamed),
                aux_task='fixture', metadata={'api_request_id': case}, client=client,
                kwargs={}, provider='fixture', model='fixture', api_mode='chat_completions',
                streaming=streamed)
            if streamed:
                self.assertFalse(any(e=='auxiliary_stream_end' for e, _ in events))
                try:
                    if case == 'broken':
                        with self.assertRaises(json.JSONDecodeError):
                            list(result)
                    else:
                        list(result)
                finally:
                    result.close()
        starts = [p for e, p in events if e=='http_attempt_start']
        ends = [p for e, p in events if e=='http_attempt_end']
        self.assertEqual(len(starts), calls[case])
        self.assertEqual(len(ends), calls[case])
        self.assertTrue(all(p['api_request_id']==case for p in starts))
        logical = [p for e, p in events if e=='post_auxiliary_call']
        self.assertEqual(len(logical), 1)
        if case == 'retry':
            self.assertEqual([p['status_code'] for p in ends], [500, 200])
            self.assertEqual(logical[0]['usage']['total_tokens'], 120)
            return
        finals = [p for e, p in events if e=='auxiliary_stream_end']
        self.assertEqual(len(finals), 1)
        if case == 'broken':
            self.assertIsNone(finals[0]['usage'])
            self.assertEqual(finals[0]['error_type'], 'JSONDecodeError')
        else:
            self.assertEqual(finals[0]['usage']['total_tokens'], 120)
