"""Claude OAuth DirectSDK — standalone Hermes model-provider registration."""
from providers import register_provider
from providers.base import ProviderProfile


class ClaudeOAuthDirectSDKProfile(ProviderProfile):
    # Claude Code 2.1.263 baked first-party catalog. Alias resolution and
    # entitlement remain native-owned; these are not account discovery results.
    model_metadata = {
        'sonnet': {'canonical_model': 'claude-sonnet-5', 'context_window': 1_000_000},
        'opus': {'canonical_model': 'claude-opus-5', 'context_window': 1_000_000},
        'haiku': {'canonical_model': 'claude-haiku-4-5', 'context_window': 200_000},
    }

    def get_model_context_length(self, model):
        # Native ML/NEn clamp even 1M catalog models to 200K without the
        # matching entitlement. Do not infer an account's allowance from a name.
        if model in self.model_metadata or any(
            model == item['canonical_model'] for item in self.model_metadata.values()
        ):
            return 200_000
        return None

    def get_usage_cost(self, model, usage):
        from decimal import Decimal, InvalidOperation
        from agent.usage_pricing import CostResult, format_cost_label

        native = (usage.raw_usage or {}).get('native_cost') or {}
        unknown = CostResult(amount_usd=None, status='unknown', source='none', label='n/a',
                             notes=('native final list-price accounting unavailable; subscription invoice unknown',))
        amount = native.get('total_cost_usd')
        models = native.get('modelUsage') or {}
        if isinstance(amount, bool) or not models or any(row.get('costBasis') != 'list' for row in models.values()):
            return unknown
        try:
            amount = Decimal(str(amount))
        except InvalidOperation:
            return unknown
        if not amount.is_finite() or amount < 0:
            return unknown
        return CostResult(amount_usd=amount, status='estimated', source='provider_cost_api',
                          label=format_cost_label(amount), notes=('native API list-price equivalent; not subscription invoice; extra usage unknown',))

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
    supports_health_check=False,
    steering_as_user_message=True,
    native_reasoning_details_type='claude-oauth-directsdk.native_assistant',
    env_vars=(),
    base_url='process://claude-oauth-directsdk',
    process_command='claude',
    process_args=(),
    process_command_env_vars=('CLAUDE_OAUTH_DIRECTSDK_COMMAND',),
    default_aux_model='sonnet',
    fallback_models=('sonnet', 'opus', 'haiku'),
)
register_provider(profile)
