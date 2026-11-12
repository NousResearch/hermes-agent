"""Bedrock runtime URLs must retain profile gateway configuration."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from hermes_cli.runtime_provider_backends import _resolve_bedrock_runtime

class BedrockRuntimeEndpointTests(unittest.TestCase):
    def resolve(self, model=None, providers=None, scope=None):
        def runtime(provider, mode, url, key, **kw):
            return dict(provider=provider, api_mode=mode, base_url=url, api_key=key, **kw)
        with (
            patch('hermes_cli.runtime_provider_backends._rp', return_value=SimpleNamespace(_runtime=runtime)),
            patch('hermes_cli.config.load_config', return_value={'providers': providers or {}}),
            patch('agent.bedrock_adapter.resolve_bedrock_runtime_region', return_value='us-east-1'),
            patch('agent.bedrock_adapter.resolve_aws_auth_env_var', return_value='providers.bedrock.key_cmd'),
            patch('agent.bedrock_adapter.resolve_bedrock_bearer_token', return_value='test-token'),
            patch('agent.bedrock_adapter.bedrock_guardrail_config', return_value={}),
            patch('agent.bedrock_adapter.bedrock_openai_uses_mantle', return_value=False),
            patch('hermes_constants.get_hermes_home_override', return_value='/tmp/profile'),
            patch('agent.secret_scope.current_secret_scope', return_value=scope or {}),
            patch.dict('os.environ', {'AWS_ENDPOINT_URL_BEDROCK_RUNTIME': 'https://wrong-launch.invalid'}),
        ):
            return _resolve_bedrock_runtime('bedrock', model or {}, 'global.anthropic.claude-sonnet-5')

    def test_scoped_sdk_override_wins(self):
        r = self.resolve(model={'base_url':'https://model.invalid'}, providers={'bedrock':{'api':'https://provider.invalid'}},
                         scope={'AWS_ENDPOINT_URL_BEDROCK_RUNTIME':'https://wovey.invalid/bedrock/project/test'})
        self.assertEqual(r['base_url'],'https://wovey.invalid/bedrock/project/test')
        self.assertEqual(r['api_mode'],'bedrock_converse')

    def test_model_url_is_preserved(self):
        self.assertEqual(self.resolve(model={'provider':'bedrock','base_url':'https://model.invalid'})['base_url'], 'https://model.invalid')

    def test_provider_url_is_preserved(self):
        self.assertEqual(self.resolve(providers={'bedrock':{'api':'https://provider.invalid'}})['base_url'], 'https://provider.invalid')

    def test_no_profile_url_uses_regional_default_not_launch_url(self):
        self.assertEqual(self.resolve()['base_url'],'https://bedrock-runtime.us-east-1.amazonaws.com')

    def test_foreign_model_provider_url_is_not_borrowed(self):
        self.assertEqual(self.resolve(model={'provider':'openrouter','base_url':'https://openrouter.ai/api/v1'})['base_url'],
                         'https://bedrock-runtime.us-east-1.amazonaws.com')

if __name__ == '__main__':
    unittest.main()
