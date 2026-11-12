"""Expiry-aware Bedrock key_cmd refresh, without network calls or real helper execution."""
import unittest
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from agent import bedrock_adapter as b
from agent.command_token_source import CommandTokenError
from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_context, reset_multiplex_context
from hermes_constants import set_hermes_home_override, reset_hermes_home_override, hermes_home_key

@contextmanager
def profile(home, values=None):
    h = set_hermes_home_override(home)
    s = set_secret_scope(values or {}, profile_home=home)
    m = set_multiplex_context(True)
    try:
        yield
    finally:
        reset_multiplex_context(m)
        reset_secret_scope(s)
        reset_hermes_home_override(h)

class BedrockTokenRefreshTests(unittest.TestCase):
    def setUp(self):
        b.reset_client_cache()
        self.addCleanup(b.reset_client_cache)
        self.now = 1000.0
        self.commands = {}
        self.config = self.enterContext(patch('hermes_cli.config.load_config_readonly', side_effect=self.load_config))
        self.enterContext(patch('agent.command_token_source.time.monotonic', side_effect=lambda: self.now))
        self.mint = self.enterContext(patch('agent.command_token_source._mint', return_value=('fresh', 120.0)))
        self.enterContext(patch.dict('os.environ', {
            'HERMES_HOME': '/tmp/hermes-refresh-launch',
            'AWS_BEARER_TOKEN_BEDROCK': 'stale-launch',
            'AWS_ENDPOINT_URL_BEDROCK_RUNTIME': 'https://launch.invalid/runtime',
        }))
        self.commands[hermes_home_key()] = 'mint-launch'
        self.enterContext(patch.object(b, '_scoped_boto3_session', side_effect=lambda token, kwargs: token))
        self.new = self.enterContext(patch.object(b, '_new_client', side_effect=lambda factory, *args, **kwargs: SimpleNamespace(token=factory, options=kwargs)))

    def load_config(self):
        return {'providers': {'bedrock': {'key_cmd': self.commands.get(hermes_home_key(), '')}}}

    def client(self):
        return b._cached_client(b._bedrock_runtime_client_cache, 'bedrock-runtime', 'us-east-1')

    def test_explicit_env_does_not_mint_from_profile_config(self):
        self.assertEqual(b.resolve_bedrock_bearer_token({}), "")
        self.assertEqual(b.resolve_bedrock_bearer_token({"AWS_BEARER_TOKEN_BEDROCK": "explicit"}), "explicit")
        self.mint.assert_not_called()

    def test_command_only_profile_is_detected_without_startup_bearer(self):
        with profile("/tmp/hermes-refresh-a"):
            self.commands[hermes_home_key()] = "mint-a"
            self.assertTrue(b.has_aws_credentials())
            self.assertEqual(b.resolve_aws_auth_env_var(), "providers.bedrock.key_cmd")
        self.mint.assert_not_called()

    def test_empty_profile_does_not_probe_launch_credentials(self):
        with profile("/tmp/hermes-refresh-empty"):
            with patch.object(b, "_boto3_chain_has_credentials", side_effect=AssertionError("ambient probe")):
                self.assertFalse(b.has_aws_credentials())

    def test_reuse_then_refresh_before_expiry_unscoped(self):
        self.mint.side_effect = [('one', 120.0), ('two', 120.0)]
        first = self.client()
        self.now = 1059.0
        self.assertIs(self.client(), first)
        self.assertEqual(b.resolve_bedrock_bearer_token(), 'one')
        self.mint.assert_called_once()
        self.now = 1060.0  # 60-second pre-expiry leeway
        renewed = self.client()
        self.assertIsNot(first, renewed)
        self.assertEqual(renewed.token, 'two')
        self.assertEqual(len(b._bedrock_clients_by_home), 1)
        self.assertEqual(b.resolve_bedrock_bearer_token(), 'two')
        self.assertEqual(self.mint.call_count, 2)

    def test_scoped_refresh_uses_profile_endpoint_not_launch(self):
        with profile('/tmp/hermes-refresh-a', {'AWS_ENDPOINT_URL_BEDROCK_RUNTIME': 'https://a.invalid/runtime',
                                             'AWS_BEARER_TOKEN_BEDROCK': 'stale-profile'}):
            self.commands[hermes_home_key()] = 'mint-a'
            self.mint.side_effect = [('a-one', 120.0), ('a-two', 120.0)]
            first = self.client()
            self.now += 61
            second = self.client()
            self.assertEqual(second.token, 'a-two')
            self.assertIsNot(first, second)
            self.assertEqual(second.options['endpoint_url'], 'https://a.invalid/runtime')
            self.assertEqual(second.options['scoped_auth'], 'bearer')
            self.assertEqual(len(b._bedrock_clients_by_home), 1)

    def test_same_command_separate_profiles_do_not_share_tokens(self):
        self.mint.side_effect = [('a', 120.0), ('b', 120.0)]
        for home, expected in [('/tmp/hermes-refresh-a', 'a'), ('/tmp/hermes-refresh-b', 'b')]:
            with profile(home):
                self.commands[hermes_home_key()] = 'same-command'
                self.assertEqual(b.resolve_bedrock_bearer_token(), expected)
        self.assertEqual(self.mint.call_count, 2)
        with profile('/tmp/hermes-refresh-a'):
            self.assertEqual(b.resolve_bedrock_bearer_token(), 'a')

    def test_profile_without_command_does_not_borrow_launch_command(self):
        with profile('/tmp/hermes-refresh-no-command', {'AWS_BEARER_TOKEN_BEDROCK': 'own-static'}):
            self.assertEqual(b.resolve_bedrock_bearer_token(), 'own-static')
        self.mint.assert_not_called()

    def test_refresh_failure_never_uses_stale_bearer_or_cached_client(self):
        self.mint.side_effect = [('fresh', 120.0), CommandTokenError('refresh failed'), CommandTokenError('refresh failed')]
        self.client()
        self.now += 61
        with self.assertRaises(CommandTokenError):
            self.client()
        with self.assertRaises(CommandTokenError):
            b.resolve_bedrock_bearer_token()
        self.assertEqual(self.new.call_count, 1)

    def test_initial_helper_failure_does_not_fall_back(self):
        self.mint.side_effect = CommandTokenError('mint failed')
        with self.assertRaises(CommandTokenError):
            self.client()
        self.new.assert_not_called()

    def test_command_change_and_cache_reset_remint(self):
        b.resolve_bedrock_bearer_token()
        self.commands[hermes_home_key()] = 'replacement-command'
        b.resolve_bedrock_bearer_token()
        self.assertEqual(len(b._bedrock_command_sources), 1)
        b.reset_client_cache()
        b.resolve_bedrock_bearer_token()
        self.assertEqual(self.mint.call_count, 3)

    def test_many_renewals_keep_one_client_per_slot(self):
        self.mint.side_effect = [(str(i), 120.0) for i in range(12)]
        for _ in range(12):
            self.client()
            self.now += 61
        self.assertEqual(len(b._bedrock_clients_by_home), 1)
        self.assertEqual(self.new.call_count, 12)

    def test_concurrent_requests_share_one_mint(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            tokens = list(pool.map(lambda _: b.resolve_bedrock_bearer_token(), range(20)))
        self.assertEqual(tokens, ['fresh'] * 20)
        self.mint.assert_called_once()

    def test_runtime_invalidation_clears_unscoped_command_client(self):
        old = self.client()
        self.assertTrue(b.invalidate_runtime_client('us-east-1'))
        self.assertIsNot(old, self.client())
        self.mint.assert_called_once()

if __name__ == '__main__':
    unittest.main()
