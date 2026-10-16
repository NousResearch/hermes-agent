from typing import Any
from providers import register_provider
from providers.base import ProviderProfile

class ClaudeCodeACPProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        from agent.external_acp_client import ExternalACPClient
        return ExternalACPClient(**client_kwargs)

claude_code_acp = ClaudeCodeACPProfile(
    name='claude-code-acp', aliases=('claude-code',), api_mode='chat_completions',
    env_vars=(), base_url='acp://claude-code', auth_type='external_process',
    process_command='claude', process_args=('--acp',),
    process_command_env_vars=('HERMES_CLAUDE_CODE_COMMAND',),
    process_args_env_var='HERMES_CLAUDE_CODE_ARGS')
register_provider(claude_code_acp)
