import os, re

target_dir = '.'
for root, _, files in os.walk(target_dir):
    if '.venv' in root or 'venv' in root or '.git' in root:
        continue
    for file in files:
        if not file.endswith('.py'):
            continue
        path = os.path.join(root, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = content
            
            new_content = new_content.replace('is_direct_openai_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""))', 'is_direct_openai_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""))')
            new_content = new_content.replace('is_azure_openai_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_lower", ""))', 'is_azure_openai_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_lower", ""))')
            new_content = new_content.replace('is_github_copilot_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""))', 'is_github_copilot_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""))')
            new_content = new_content.replace('is_openrouter_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""))', 'is_openrouter_url(getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""))')
            new_content = re.sub(r'agent\._max_tokens_param\((.*?)\)', r'max_tokens_param(agent.model, getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""), getattr(agent, "_base_url_lower", ""), \1)', new_content)
            
            new_content = re.sub(r'agent\._anthropic_prompt_cache_policy\((.*?)\)', r'anthropic_prompt_cache_policy(agent, \1)', new_content)
            new_content = new_content.replace('anthropic_prompt_cache_policy(agent, )', 'anthropic_prompt_cache_policy(agent)')
            
            if new_content != content:
                imports_openai = 'from agent.providers.openai_adapter import is_direct_openai_url, is_azure_openai_url, is_github_copilot_url, max_tokens_param\n'
                imports_openrouter = 'from agent.providers.openrouter_adapter import is_openrouter_url\n'
                imports_anthropic = 'from agent.agent_runtime_helpers import anthropic_prompt_cache_policy\n'
                
                # Check what we need to import
                needs_openai = 'is_direct_openai_url' in new_content or 'max_tokens_param' in new_content
                needs_openrouter = 'is_openrouter_url' in new_content
                needs_anthropic = 'anthropic_prompt_cache_policy' in new_content
                
                # Skip importing anthropic inside agent_runtime_helpers
                if path.endswith('agent_runtime_helpers.py'):
                    needs_anthropic = False
                
                imports_to_add = ''
                if needs_openai and 'from agent.providers.openai_adapter import is_direct_openai_url' not in new_content:
                    imports_to_add += imports_openai
                if needs_openrouter and 'from agent.providers.openrouter_adapter import is_openrouter_url' not in new_content:
                    imports_to_add += imports_openrouter
                if needs_anthropic and 'from agent.agent_runtime_helpers import anthropic_prompt_cache_policy' not in new_content:
                    imports_to_add += imports_anthropic
                
                if imports_to_add:
                    lines = new_content.split('\n')
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        if 'from __future__' in line:
                            insert_idx = i + 1
                    lines.insert(insert_idx, imports_to_add.strip())
                    new_content = '\n'.join(lines)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f'Updated usages in {path}')
        except Exception as e:
            pass
