"""Claude OAuth DirectSDK — standalone Hermes model-provider registration."""
from providers import register_provider
from providers.base import ProviderProfile


class ClaudeOAuthDirectSDKProfile(ProviderProfile):
    def create_client(self, **client_kwargs):
        from .directsdk import Client
        return Client(**client_kwargs)

    def fetch_models(self, **_):
        return None

    def build_api_kwargs_extras(self, *, reasoning_config=None, **_):
        return ({'reasoning': dict(reasoning_config)} if reasoning_config else {}), {}


profile = ClaudeOAuthDirectSDKProfile(
    name='claude-oauth-directsdk',
    display_name='Claude OAuth DirectSDK',
    description='Request-scoped official Claude Code; Hermes owns tool execution',
    api_mode='chat_completions',
    auth_type='external_process',
    steering_as_user_message=True,
    env_vars=(),
    base_url='process://claude-oauth-directsdk',
    process_command='claude',
    process_args=(),
    process_command_env_vars=('CLAUDE_OAUTH_DIRECTSDK_COMMAND',),
    default_aux_model='sonnet',
    fallback_models=('sonnet', 'opus', 'haiku'),
)
register_provider(profile)
