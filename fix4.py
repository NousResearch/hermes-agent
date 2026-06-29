import os
import re

# 1. Add requested_output_cap_from_api_kwargs to openai_adapter
openai_adapter_path = r'agent\providers\openai_adapter.py'
with open(openai_adapter_path, 'r', encoding='utf-8') as f:
    openai_adapter = f.read()

cap_func = """
from typing import Optional, Any

def requested_output_cap_from_api_kwargs(api_kwargs: Any) -> Optional[int]:
    \"\"\"Extract the outgoing response token cap from a prepared request.\"\"\"
    if not isinstance(api_kwargs, dict):
        return None
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        raw = api_kwargs.get(key)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None
"""

if 'def requested_output_cap_from_api_kwargs' not in openai_adapter:
    with open(openai_adapter_path, 'a', encoding='utf-8') as f:
        f.write(cap_func)

# 2. Fix conversation_loop.py
conv_loop_path = r'agent\conversation_loop.py'
with open(conv_loop_path, 'r', encoding='utf-8') as f:
    conv_loop = f.read()

conv_loop = conv_loop.replace('agent._requested_output_cap_from_api_kwargs(api_kwargs)', 'requested_output_cap_from_api_kwargs(api_kwargs)')
if 'requested_output_cap_from_api_kwargs' not in conv_loop:
    # Need to import it
    import_stmt = "from agent.providers.openai_adapter import requested_output_cap_from_api_kwargs\n"
    conv_loop = import_stmt + conv_loop

with open(conv_loop_path, 'w', encoding='utf-8') as f:
    f.write(conv_loop)


# 3. Fix tests/run_agent/test_run_agent.py
test_run_agent = r'tests\run_agent\test_run_agent.py'
with open(test_run_agent, 'r', encoding='utf-8') as f:
    tr = f.read()

tr = tr.replace('agent._is_azure_openai_url', 'is_azure_openai_url')
tr = tr.replace('agent._is_direct_openai_url', 'is_direct_openai_url')

# Add imports
tr = tr.replace('from agent.chat_completion_helpers import FailoverReason', 'from agent.chat_completion_helpers import FailoverReason\nfrom agent.providers.openai_adapter import is_azure_openai_url, is_direct_openai_url')

with open(test_run_agent, 'w', encoding='utf-8') as f:
    f.write(tr)

# 4. Fix tests/run_agent/test_fallback_credential_isolation.py
tfc_path = r'tests\run_agent\test_fallback_credential_isolation.py'
with open(tfc_path, 'r', encoding='utf-8') as f:
    tfc = f.read()

tfc = tfc.replace('agent._is_azure_openai_url.return_value', 'mock_is_azure.return_value')

# I need to mock the actual module instead of the agent property.
# I'll just change the patch inside the test if needed.
# For now, let's just write this script and then check test_fallback_credential_isolation.py manually.

print("Done fix4")
