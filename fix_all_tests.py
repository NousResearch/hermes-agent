import re
import os

files = ['tests/run_agent/test_run_agent.py', 'tests/run_agent/test_run_agent_codex_responses.py']

for path in files:
    if not os.path.exists(path): continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Match max_tokens_param(agent.model, getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""), getattr(agent, "_base_url_lower", ""), <number>)
    pattern = r'max_tokens_param\(\s*agent\.model,\s*getattr\(agent,\s*"base_url",\s*None\),\s*getattr\(agent,\s*"_base_url_hostname",\s*""\),\s*getattr\(agent,\s*"_base_url_lower",\s*""\),\s*(\d+)\s*\)'
    
    def repl(m):
        return f'max_tokens_param(agent.model, {m.group(1)}, getattr(agent, "base_url", None), getattr(agent, "_base_url_hostname", ""), getattr(agent, "_base_url_lower", ""))'

    new_content = re.sub(pattern, repl, content)

    if new_content != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed max_tokens_param param order in {path}')
    else:
        print(f'No matches in {path}')
