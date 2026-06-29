import os

path = r'agent\chat_completion_helpers.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('agent._is_direct_openai_url(fb_base_url)', 'is_direct_openai_url(fb_base_url, getattr(agent, "_base_url_hostname", ""))')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed chat_completion_helpers.py direct url')
