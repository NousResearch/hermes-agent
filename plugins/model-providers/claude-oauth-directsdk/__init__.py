"""Claude OAuth DirectSDK — standalone Hermes model-provider registration."""
from providers import register_provider
from providers.base import ProviderProfile
from .model_catalog import MODEL_METADATA, native_model


class ClaudeOAuthDirectSDKProfile(ProviderProfile):
    model_metadata = MODEL_METADATA

    def get_model_context_length(self, model):
        return self.model_metadata.get(native_model(model), {}).get('context_window')

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
    default_aux_model='claude-sonnet-5[1m]',
    fallback_models=tuple(MODEL_METADATA),
)
register_provider(profile)
