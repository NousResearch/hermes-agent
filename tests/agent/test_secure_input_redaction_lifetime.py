"""Exact-value redaction lifetime regressions; synthetic credentials only."""
import pytest

from agent import secure_input_broker as broker

SENTINEL = '[REDACTED SECURE INPUT]'


@pytest.fixture(autouse=True)
def isolated_broker(monkeypatch):
    broker.clear_all()
    clock = [1000.0]
    monkeypatch.setattr(broker, '_now', lambda: clock[0])
    yield clock
    broker.clear_all()


def test_shorter_duplicate_capture_does_not_unprotect_live_reference(isolated_broker):
    value = 'synthetic-duplicate-capture-value'
    first = broker.register_secret(value, purpose='fixture', ttl_seconds=600)
    broker.register_secret(value, purpose='fixture', ttl_seconds=1)
    isolated_broker[0] += 2
    assert broker.describe_secret(first['secret_ref']) is not None
    assert broker.redact_known_secrets(value) == SENTINEL


@pytest.mark.parametrize('reverse', [False, True])
def test_nested_values_are_fully_masked_independently_of_capture_order(reverse):
    short = 'synthetic-nested'
    long = short + '-sensitive-suffix'
    values = [short, long]
    for value in reversed(values) if reverse else values:
        broker.register_secret(value, purpose='fixture')
    assert broker.redact_known_secrets(long) == SENTINEL
