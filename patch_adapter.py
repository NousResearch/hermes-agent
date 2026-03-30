import re

with open("agent/google_adapter.py", "r") as f:
    code = f.read()

old_usage = """    # Usage tracking
    prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0"""

new_usage = """    # Usage tracking
    prompt_tokens = 0
    completion_tokens = 0
    if getattr(response, 'usage_metadata', None):
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        completion_tokens = response.usage_metadata.candidates_token_count or 0"""

if old_usage in code:
    code = code.replace(old_usage, new_usage)
    with open("agent/google_adapter.py", "w") as f:
        f.write(code)
    print("Patched agent/google_adapter.py")
