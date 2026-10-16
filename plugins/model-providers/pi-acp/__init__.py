from typing import Any
from providers import register_provider
from providers.base import ProviderProfile

class PiACPProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        from agent.external_acp_client import ExternalACPClient
        return ExternalACPClient(**client_kwargs)

pi_acp = PiACPProfile(
    name='pi-acp', aliases=('pi-agent',), api_mode='chat_completions',
    env_vars=(), base_url='acp://pi', auth_type='external_process',
    process_command='pi-acp', process_args=(),
    process_command_env_vars=('HERMES_PI_ACP_COMMAND',),
    process_args_env_var='HERMES_PI_ACP_ARGS')
register_provider(pi_acp)
