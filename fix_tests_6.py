import re
import os

files_to_fix = [
    'tests/run_agent/test_anthropic_prompt_cache_policy.py',
    'tests/run_agent/test_compressor_fallback_update.py',
    'tests/run_agent/test_switch_model_pool_reload_52727.py',
    'tests/run_agent/test_anthropic_third_party_oauth_guard.py',
    'tests/run_agent/test_callable_api_key.py'
]

for path in files_to_fix:
    if not os.path.exists(path): continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix anthropic_prompt_cache_policy
    if '_anthropic_prompt_cache_policy' in content:
        content = content.replace('agent._anthropic_prompt_cache_policy(', 'anthropic_prompt_cache_policy(agent, ')
        if 'from agent.agent_runtime_helpers import anthropic_prompt_cache_policy' not in content:
            content = content.replace('from run_agent import AIAgent', 'from run_agent import AIAgent\nfrom agent.agent_runtime_helpers import anthropic_prompt_cache_policy')

    # Fix _is_direct_openai_url
    if '._is_direct_openai_url' in content:
        content = content.replace('agent._is_direct_openai_url(', 'is_direct_openai_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""), getattr(agent, "_base_url_lower", ""))')
        if 'from agent.providers.openai_adapter import is_direct_openai_url' not in content:
            content = content.replace('from run_agent import AIAgent', 'from run_agent import AIAgent\nfrom agent.providers.openai_adapter import is_direct_openai_url')

    # Fix agent.anthropic_adapter -> agent.providers.anthropic_adapter
    if 'agent.anthropic_adapter' in content:
        content = content.replace('agent.anthropic_adapter', 'agent.providers.anthropic_adapter')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Fixed {path}')
