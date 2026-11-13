"""Deterministic loopback model: one tool boundary, then the guided answer."""
from http.server import BaseHTTPRequestHandler
import json


class GuidanceModel(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        payload = b'{"data":[]}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        first = bool(body.get('messages')) and len(self.server.requests) == 1
        if first:
            self.server.blocked.set()
            if not self.server.release.wait(30):
                raise TimeoutError('primary was never released by guidance')
        text = next((m.get('content', '') for m in reversed(body.get('messages', [])) if m['role'] == 'user'), '')
        message = {'role': 'assistant', 'content': getattr(self.server, 'answer', '') or text}
        if first and getattr(self.server, 'case', '') != 'unconsumed':
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'guidance-boundary', 'type': 'function', 'function': {
                    'name': 'terminal', 'arguments': json.dumps({'command': 'printf guidance-boundary', 'timeout': 5})}}]}
        if not first and getattr(self.server, 'hold_final', False) and not self.server.observed.wait(30):
            raise TimeoutError('incorporated receipt was never observed')
        finish = 'tool_calls' if 'tool_calls' in message else 'stop'
        usage = {'prompt_tokens': 10 if first else 17, 'completion_tokens': 3 if first else 5,
                 'total_tokens': 13 if first else 22}
        if body.get('stream'):
            delta = dict(message)
            if 'tool_calls' in delta:
                delta['tool_calls'] = [dict(delta['tool_calls'][0], index=0)]
            frame = {'id': 'guidance-model', 'model': 'controlled-gateway', 'object': 'chat.completion.chunk',
                     'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}], 'usage': usage}
            payload = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            payload = json.dumps({'id': 'guidance-model', 'model': 'controlled-gateway',
                'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': usage}).encode()
            kind = 'application/json'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
