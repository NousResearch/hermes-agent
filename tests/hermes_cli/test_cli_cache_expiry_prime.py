"""The classic CLI's idle hook (``CLISessionMixin._maybe_compact_before_cache_expiry``) primes the chat
and grants a fresh keep-alive budget only after a compaction that made it smaller: ``compress_now``
reports ``compressed`` for an unchanged, aborted or refused history too."""

import contextlib
import queue
from types import SimpleNamespace

import pytest

from agent.conversation_compression_manual import CompressRequest, CompressResult
from hermes_cli.cli_session_mixin import CLISessionMixin

HISTORY = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(8)]


def _cli():
    adopted = []
    cli = SimpleNamespace(
        agent=SimpleNamespace(), _agent_running=False, _pending_input=queue.Queue(), conversation_history=list(HISTORY),
        session_id="sid", _busy_command=lambda *a, **k: contextlib.nullcontext(),
        _adopt_compress_result=lambda result: adopted.append(result),
    )
    return cli, adopted


@pytest.mark.parametrize("flags, primes", [
    ({}, True), ({"noop": True}, False), ({"aborted": True}, False), ({"refused_would_grow": True}, False),
])
def test_only_a_committed_compaction_primes_the_chat(monkeypatch, flags, primes):
    import agent.conversation_compression_manual as manual
    import agent.prompt_cache_lifecycle as decision

    primed = []
    summary = {"noop": False, "aborted": False, "refused_would_grow": False, **flags}
    after = HISTORY if flags else HISTORY[:2]
    monkeypatch.setattr(decision, "claim_compaction", lambda agent, **k: True)
    monkeypatch.setattr(decision, "prime_after_compaction", lambda agent, history: primed.append(history))
    monkeypatch.setattr(manual, "compress_now", lambda *a, **k: CompressResult(
        "compressed", list(HISTORY), list(after), 100, 50, CompressRequest(), summary=summary))
    cli, adopted = _cli()

    CLISessionMixin._maybe_compact_before_cache_expiry(cli)

    assert len(adopted) == 1
    assert bool(primed) is primes
