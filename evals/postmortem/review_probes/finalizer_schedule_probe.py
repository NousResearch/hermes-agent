"""#103507: pytest plugin forcing the consumer-first schedule (-p finalizer_schedule_probe --finalizer-probe=consumer-first)

Independent-review probe (written by the /review subagent for tracking issue #103563, adapted here).
It reproduced a defect in the first version of the PR; the fixed head must pass it. Paths are taken
from the command line / environment, never hard-coded. Usage: see the argument parsing at the top of the file.
"""
import asyncio
import json
import pytest


def pytest_addoption(parser):
    parser.addoption('--finalizer-probe', default='off')


@pytest.fixture(autouse=True)
def finalizer_schedule_probe(request, monkeypatch):
    mode = request.config.getoption('--finalizer-probe')
    if mode == 'off':
        yield
        return
    from agent import relay_llm, chat_completion_helpers
    print('PROBE_IMPORT', relay_llm.__file__, chat_completion_helpers.__file__)
    original_provider = relay_llm.ManagedLlmStream._provider_stream
    original_count = chat_completion_helpers._StreamingCall._count_chunk
    stats = {'gated_terminal_chunks': 0, 'consumer_releases': 0}

    def terminal(chunk):
        choices = chunk.get('choices') or []
        return (not choices and chunk.get('usage') is not None) or (
            bool(choices) and choices[0].get('finish_reason') == 'tool_calls')

    async def gated_provider(stream, *args):
        async for chunk in original_provider(stream, *args):
            if terminal(chunk):
                stream._review_gate = asyncio.Event()
                stats['gated_terminal_chunks'] += 1
                yield chunk
                await asyncio.wait_for(stream._review_gate.wait(), timeout=10)
            else:
                yield chunk

    def count_and_release(call, diag, chunk):
        stream = call.managed_stream_holder.get('stream')
        gate = getattr(stream, '_review_gate', None)
        raw = chunk.model_dump() if hasattr(chunk, 'model_dump') else vars(chunk)
        if terminal(raw) and gate is not None and not gate.is_set():
            gate.set()
            stats['consumer_releases'] += 1
        return original_count(call, diag, chunk)

    assert mode == 'consumer-first'
    monkeypatch.setattr(relay_llm.ManagedLlmStream, '_provider_stream', gated_provider)
    monkeypatch.setattr(chat_completion_helpers._StreamingCall, '_count_chunk', count_and_release)
    yield
    print('PROBE_STATS', json.dumps(stats, sort_keys=True))
