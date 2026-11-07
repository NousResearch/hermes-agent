"""Real botocore signing tests: scoped bearers never borrow launch identity."""
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from agent import bedrock_adapter as b
from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_context, reset_multiplex_context
from hermes_constants import set_hermes_home_override, reset_hermes_home_override

@contextmanager
def scope(home, values):
    h = set_hermes_home_override(home)
    s = set_secret_scope(values, profile_home=home)
    m = set_multiplex_context(True)
    try:
        yield
    finally:
        reset_multiplex_context(m)
        reset_secret_scope(s)
        reset_hermes_home_override(h)

class Captured(BaseException):
    pass

class ScopedBearerTests(unittest.TestCase):
    def setUp(self):
        b.reset_client_cache()
        self.enterContext(patch("hermes_cli.config.load_config_readonly", return_value={}))
        self.env = patch.dict('os.environ', {
            'AWS_BEARER_TOKEN_BEDROCK': 'WRONG-LAUNCH-TOKEN',
            'AWS_ENDPOINT_URL_BEDROCK_RUNTIME': 'https://wrong-launch.invalid',
            'AWS_ACCESS_KEY_ID': 'WRONG-LAUNCH-KEY',
            'AWS_SECRET_ACCESS_KEY': 'WRONG-LAUNCH-SECRET',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(b.reset_client_cache)

    def values(self, token='profile-A'):
        return {'AWS_BEARER_TOKEN_BEDROCK': token,
                'AWS_ENDPOINT_URL_BEDROCK_RUNTIME': 'https://profile.invalid/bedrock'}

    def capture(self, client):
        captured = {}
        def intercept(request, **kwargs):
            captured['auth'] = request.headers.get('Authorization')
            captured['url'] = request.url
            raise Captured()
        client.meta.events.register('before-send.bedrock-runtime.Converse', intercept)
        try:
            with self.assertRaises(Captured):
                client.converse(modelId='global.anthropic.claude-sonnet-5',
                    messages=[{'role': 'user', 'content': [{'text': 'test'}]}],
                    inferenceConfig={'maxTokens': 1})
        finally:
            client.meta.events.unregister('before-send.bedrock-runtime.Converse', intercept)
        return captured

    def test_profile_bearer_signs_and_uses_own_endpoint(self):
        with scope('/tmp/hermes-profile-a', self.values()):
            c = b._get_bedrock_runtime_client('us-east-1')
            r = self.capture(c)
        self.assertEqual(r['auth'], b'Bearer profile-A')
        self.assertTrue(r['url'].startswith('https://profile.invalid/bedrock/'))

    def test_credentialless_profile_cannot_borrow_launch_token(self):
        with scope('/tmp/hermes-profile-empty', {}):
            with self.assertRaises(RuntimeError):
                b._get_bedrock_runtime_client('us-east-1')

    def test_removed_token_cannot_reuse_cached_client(self):
        with scope('/tmp/hermes-profile-a', self.values()):
            b._get_bedrock_runtime_client('us-east-1')
        with scope('/tmp/hermes-profile-a', {}):
            with self.assertRaises(RuntimeError):
                b._get_bedrock_runtime_client('us-east-1')

    def test_rotation_changes_signed_token(self):
        with scope('/tmp/hermes-profile-a', self.values()):
            a = b._get_bedrock_runtime_client('us-east-1')
        with scope('/tmp/hermes-profile-a', self.values('rotated')):
            c = b._get_bedrock_runtime_client('us-east-1')
            self.assertIsNot(a, c)
            self.assertEqual(self.capture(c)['auth'], b'Bearer rotated')

    def test_profiles_do_not_share_clients(self):
        with scope('/tmp/hermes-profile-a', self.values()):
            a = b._get_bedrock_runtime_client('us-east-1')
        with scope('/tmp/hermes-profile-b', self.values('profile-B')):
            c = b._get_bedrock_runtime_client('us-east-1')
            self.assertIsNot(a, c)
            self.assertEqual(self.capture(c)['auth'], b'Bearer profile-B')

    def test_missing_endpoint_does_not_borrow_launch_endpoint(self):
        with scope('/tmp/hermes-profile-a', {'AWS_BEARER_TOKEN_BEDROCK': 'profile-A'}):
            with self.assertRaises(RuntimeError):
                b._get_bedrock_runtime_client('us-east-1')

    def test_scoped_iam_never_inherits_ambient_bearer(self):
        v = {'AWS_ACCESS_KEY_ID': 'PROFILEKEY', 'AWS_SECRET_ACCESS_KEY': 'PROFILESECRET',
             'AWS_ENDPOINT_URL_BEDROCK_RUNTIME': 'https://profile.invalid/bedrock'}
        with scope('/tmp/hermes-profile-iam', v):
            c = b._get_bedrock_runtime_client('us-east-1')
            r = self.capture(c)
            self.assertTrue(r['auth'].startswith(b'AWS4-HMAC-SHA256 Credential=PROFILEKEY/'))

    def test_insecure_endpoint_is_rejected(self):
        v = self.values()
        v['AWS_ENDPOINT_URL_BEDROCK_RUNTIME'] = 'http://profile.invalid'
        with scope('/tmp/hermes-profile-a', v):
            with self.assertRaises(RuntimeError):
                b._get_bedrock_runtime_client('us-east-1')

if __name__ == '__main__':
    unittest.main()
