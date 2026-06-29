import os
import sys

# add hermes to path
sys.path.insert(0, os.path.abspath('.'))

from tests.run_agent.test_context_token_tracking import test_anthropic_no_cache_fields

class DummyMonkeypatch:
    def setattr(self, target, val, *args, **kwargs):
        # We know we're patching:
        # "agent.providers.anthropic_adapter.build_anthropic_client"
        # "agent.auxiliary_client.resolve_provider_client"
        pass
    def undo(self): pass

import sys
import threading
from unittest.mock import MagicMock

def trace_calls(frame, event, arg):
    if event == 'call':
        func_name = frame.f_code.co_name
        file_name = frame.f_code.co_filename
        if 'hermes-agent' in file_name and ('agent_init' in file_name or 'context_compressor' in file_name or 'model_metadata' in file_name):
            print(f"CALL: {func_name} in {file_name}")
    return trace_calls

sys.settrace(trace_calls)
threading.settrace(trace_calls)

print("Running test...", flush=True)
try:
    # test_anthropic_cache_read_and_creation_added(DummyMonkeypatch())
    print("Test 1 finished", flush=True)
    test_anthropic_no_cache_fields(DummyMonkeypatch())
    print("Test 2 finished", flush=True)
    print("Test finished successfully!", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Test failed with exception!", flush=True)
    sys.exit(1)
