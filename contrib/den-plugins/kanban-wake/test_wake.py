"""Contract tests against the actual webhook verifier, never production DB."""
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from gateway.platforms.webhook import WebhookAdapter

spec = importlib.util.spec_from_file_location('wake_under_test', Path(__file__).with_name('__init__.py'))
wake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wake)

class WakeTests(unittest.TestCase):
    def setUp(self):
        wake._last.clear()

    def request(self):
        seen = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"status":"accepted"}'
        def post(req, **kw):
            seen.append(req)
            return Response()
        with patch.object(wake, '_secret', return_value='unit-secret'), patch.object(wake, '_cfg', return_value={}), patch.object(wake.urllib.request, 'urlopen', side_effect=post):
            wake._post('event data', 't_fixture', 'blocked')
        req = seen[0]
        headers = {k.lower():v for k,v in req.header_items()}
        class Headers(dict):
            def get(self,k,d=None): return super().get(k.lower(),d)
        return SimpleNamespace(headers=Headers(headers),match_info={'route_name':'kanban-wake'}),req.data

    def test_actual_receiver_accepts_emitter(self):
        req, body = self.request()
        self.assertTrue(WebhookAdapter._validate_signature(object(), req, body, 'unit-secret'))

    def test_tamper_rejected(self):
        req, body = self.request()
        self.assertFalse(WebhookAdapter._validate_signature(object(), req, body+b' ', 'unit-secret'))

    def test_request_has_idempotency_id(self):
        req, _ = self.request()
        self.assertTrue(req.headers.get('X-Request-ID'))

    def test_failed_send_can_retry_immediately(self):
        with patch.object(wake,'_cfg',return_value={}), patch.object(wake,'_post',side_effect=[OSError('offline'),None]) as post:
            for _ in range(2): wake._emit('blocked','t_fixture',title='fixture',reason='reason')
            self.assertEqual(post.call_count,2)

    def test_success_debounced(self):
        with patch.object(wake,'_cfg',return_value={}), patch.object(wake,'_post') as post:
            for _ in range(2): wake._emit('blocked','t_fixture',title='fixture',reason='reason')
            self.assertEqual(post.call_count,1)

    def test_redirect_to_current_session(self):
        calls=[]
        gateway=SimpleNamespace(_schedule_plugin_message_injection=lambda **kw: calls.append(kw) or True)
        event=SimpleNamespace(source=SimpleNamespace(platform='webhook',user_id='webhook:kanban-wake'),text='untrusted event data')
        with patch.object(wake,'_cfg',return_value={'session_key':'test-session'}):
            result=wake._pre_gateway_dispatch(event=event,gateway=gateway)
        self.assertEqual(result['action'],'skip')
        self.assertEqual(calls[0]['session_key'],'test-session')

if __name__ == '__main__': unittest.main()
