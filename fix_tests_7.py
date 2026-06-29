import os

# 1. Fix patch paths in test_anthropic_third_party_oauth_guard.py
path = r'tests\run_agent\test_anthropic_third_party_oauth_guard.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('patch("agent.anthropic_adapter.', 'patch("agent.providers.anthropic_adapter.')
content = content.replace('patch(\'agent.anthropic_adapter.', 'patch(\'agent.providers.anthropic_adapter.')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)


# 2. Fix read_text() encoding in test_callable_api_key.py
path = r'tests\run_agent\test_callable_api_key.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('.read_text()', '.read_text(encoding="utf-8")')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed both test files')
