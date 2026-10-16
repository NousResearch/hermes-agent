"""The ordinary ModelPeer with a deterministic goal judge on the same loopback server."""
from io import BytesIO
import json

from tests.gateway.fixtures.shared_authority_peer import ModelPeer as SharedModelPeer


class ModelPeer(SharedModelPeer):
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        request = json.loads(body)
        if request.get('model') != 'local-goal-judge':
            self.rfile = BytesIO(body)
            return super().do_POST()
        self.server.judgments.append(request)
        message = {'role': 'assistant', 'content': json.dumps({
            'verdict': 'continue', 'reason': 'fixture needs another step'})}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'id': 'local-judge', 'object': 'chat.completion',
            'created': 1, 'model': 'local-goal-judge', 'choices': [
                {'index': 0, 'message': message, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode())
