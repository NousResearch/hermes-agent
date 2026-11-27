"""The non-required pre-compress checkpoint must never fail silently.

``_pre_compress_memory_context()`` wrapped the non-required ``on_pre_compress()``
call in ``contextlib.suppress(Exception)``. A provider that raised there -- an
unreachable durable store, an archiving backend that snapshots the raw transcript
before lossy compression -- lost its insights *and* its snapshot with no log line
anywhere, so "this compression was archived" and "archiving silently broke" were
indistinguishable after the fact.

These pin the observable contract: the failure stays non-fatal, every outcome
(failure / no provider / success) leaves a trace, and a working provider's
context is still handed to the summary unchanged.
"""

import logging
from types import SimpleNamespace

from agent.conversation_compression import _pre_compress_memory_context

_MODULE_LOGGER = "agent.conversation_compression"
_EVIDENCE = [
    {"role": "user", "content": "remember this"},
    {"role": "assistant", "content": "noted"},
]


class _RaisingProvider:
    def on_pre_compress(self, messages, evidence_messages=None):
        raise RuntimeError("durable store unreachable")


class _WorkingProvider:
    def __init__(self, context="provider insights"):
        self.calls = 0
        self._context = context

    def on_pre_compress(self, messages, evidence_messages=None):
        self.calls += 1
        return self._context


def _agent(memory_manager):
    return SimpleNamespace(_memory_manager=memory_manager)


def _module_records(caplog):
    return [r for r in caplog.records if r.name == _MODULE_LOGGER]


def test_failing_non_required_checkpoint_warns_and_compression_continues(caplog):
    """Non-fatal stays non-fatal -- but the failure is now observable."""
    with caplog.at_level(logging.WARNING, logger=_MODULE_LOGGER):
        context = _pre_compress_memory_context(_agent(_RaisingProvider()), _EVIDENCE, False)

    assert context == "", "a non-required checkpoint failure must not block compression"
    warnings = [r for r in _module_records(caplog) if r.levelno >= logging.WARNING]
    assert warnings, "a failing non-required checkpoint must leave a warning, not vanish"
    assert "pre-compress" in warnings[0].getMessage().lower()
    assert warnings[0].exc_info is not None, "the warning must carry the provider traceback"


def test_every_checkpoint_outcome_is_distinguishable_in_the_log(caplog):
    """Success (archived), absence (not archived) and failure (broke) read differently."""
    with caplog.at_level(logging.DEBUG, logger=_MODULE_LOGGER):
        caplog.clear()
        provider = _WorkingProvider()
        context = _pre_compress_memory_context(_agent(provider), _EVIDENCE, False)
        archived_records = list(_module_records(caplog))

        caplog.clear()
        assert _pre_compress_memory_context(_agent(None), _EVIDENCE, False) == ""
        no_provider_records = list(_module_records(caplog))

    assert provider.calls == 1
    assert context == "provider insights"
    assert not [r for r in archived_records if r.levelno >= logging.WARNING]
    assert no_provider_records, "compressing with no memory provider must leave a trace too"
    assert not [r for r in no_provider_records if r.levelno >= logging.WARNING]
