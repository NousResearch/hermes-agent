import os
path = r'agent\chat_completion_helpers.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'max_tokens_param_fn=agent._max_tokens_param,',
    'max_tokens_param_fn=lambda model, val, base: max_tokens_param(model, val, base, getattr(agent, "_base_url_hostname", ""), getattr(agent, "_base_url_lower", "")),'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed max_tokens_param_fn properly.')
