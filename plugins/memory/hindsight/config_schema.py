"""Operational Hindsight settings in the native dashboard."""
from plugins.memory.config_schema import (
    KIND_SECRET, KIND_TEXT, STORAGE_CONFIG_YAML, ProviderConfigSchema, ProviderField,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name='hindsight', label='Hindsight', storage=STORAGE_CONFIG_YAML,
    fields=(
        ProviderField(key='llm_model', label='Memory model', default='gpt-5.6-luna',
                      description='Codex model used by the Railway memory service.'),
        ProviderField(key='llm_reasoning_effort', label='Learning effort', default='low'),
        ProviderField(key='reflect_llm_reasoning_effort', label='Recall effort', default='medium'),
        ProviderField(key='url', label='API URL', kind=KIND_TEXT,
                      default='http://127.0.0.1:8888', inline=True),
        ProviderField(key='bank_id', label='Bank ID', kind=KIND_TEXT,
                      description='Blank selects a unique bank per profile.', inline=True),
        ProviderField(key='api_key', label='API key', kind=KIND_SECRET,
                      env_key='HINDSIGHT_API_KEY', inline=True),
    ),
)
