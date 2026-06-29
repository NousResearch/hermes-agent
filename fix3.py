import os
import re

path = r'agent\chat_completion_helpers.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace agent._max_tokens_param
content = content.replace(
    'max_tokens_param_fn=agent._max_tokens_param',
    'max_tokens_param_fn=lambda model, val, base: max_tokens_param(model, val, base, getattr(agent, "_base_url_hostname", ""), getattr(agent, "_base_url_lower", ""))'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed _max_tokens_param')

path2 = r'agent\conversation_loop.py'
with open(path2, 'r', encoding='utf-8') as f:
    content2 = f.read()

# Replace agent._requested_output_cap_from_api_kwargs
content2 = content2.replace(
    'agent._requested_output_cap_from_api_kwargs(api_kwargs)',
    'requested_output_cap_from_api_kwargs(api_kwargs)'
)

# And add the import if missing
if 'requested_output_cap_from_api_kwargs' not in content2 and 'from agent.providers.openai_adapter' not in content2:
    pass # Wait, let me check how to import it properly first.

with open(path2, 'w', encoding='utf-8') as f:
    f.write(content2)

print('Fixed conversation_loop.py partially')
