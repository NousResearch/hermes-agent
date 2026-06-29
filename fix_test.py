import re
path = r'tests\run_agent\test_run_agent.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'max_tokens_param\(agent\.model, getattr\(agent, "base_url", None\), getattr\(agent, "_base_url_hostname", ""\), getattr\(agent, "_base_url_lower", ""\), 4096\)'
repl = r'max_tokens_param(agent.model, 4096, getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""), getattr(agent, "_base_url_lower", ""))'
content = re.sub(pattern, repl, content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed max_tokens_param param order for 4096')
