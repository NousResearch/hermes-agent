"""A managed worker that dies reports WHY in its own frame.

Observed defect (t_925a816f): every managed worker that failed at start sent
``{'type': 'error', 'reason': 'managed_worker_failed'}`` and nothing else, so the card's crash
record could only quote that reason code. The real cause (``ValueError:
kanban_skills_unavailable`` in ``gateway/session_kanban.py:388``) sat in the profile's
managed-worker.log, four logs away from the board.

The child now sends a bounded, redacted cause with the failing STEP, the exception type, its
one-line message and the source location. The raw traceback never crosses the pipe, so a
credential in an exception message cannot leak through it.
"""
import pytest


class FakeChannel:
    """Captures the frames a worker would send over its stdout pipe."""

    def __init__(self):
        self.frames = []

    def send(self, kind, **fields):
        self.frames.append((kind, fields))


@pytest.fixture
def bootstrap(monkeypatch):
    """Minimal pipe: hello, a validated bootstrap frame, then whatever the test's execute does."""
    from agent import managed_worker

    monkeypatch.setattr(managed_worker, 'hello', lambda: {'pid': 1, 'birth': 2})
    monkeypatch.setattr(managed_worker, 'read_frame', lambda stream: {'bootstrap': True})
    monkeypatch.setattr(managed_worker, 'validate_bootstrap', lambda frame: {'api_key': 'sk-live-SECRET-123',
                                                                            'route': 'r'})
    return managed_worker


def test_a_failing_execute_sends_the_step_and_the_exception(bootstrap, monkeypatch):
    def execute(frame, channel):
        raise ValueError('kanban_skills_unavailable')

    monkeypatch.setattr(bootstrap, 'execute', execute)
    channel = FakeChannel()
    assert bootstrap.serve(channel) == 1
    kinds = [kind for kind, _ in channel.frames]
    assert kinds == ['hello', 'error']
    error = dict(channel.frames[-1][1])
    assert error['reason'] == 'managed_worker_failed'
    cause = error['cause']
    assert cause['step'] == 'execute'
    assert cause['exc_type'] == 'ValueError'
    assert cause['detail'] == 'ValueError: kanban_skills_unavailable'
    assert 'test_managed_worker_loss_cause.py' in cause['where'] and 'in execute' in cause['where']


def test_a_failing_bootstrap_names_that_step(bootstrap, monkeypatch):
    def validate(frame):
        raise KeyError('home')

    monkeypatch.setattr(bootstrap, 'validate_bootstrap', validate)
    channel = FakeChannel()
    assert bootstrap.serve(channel) == 1
    cause = channel.frames[-1][1]['cause']
    assert cause['step'] == 'bootstrap' and cause['exc_type'] == 'KeyError'


def test_the_bootstrap_key_never_reaches_the_owner_frame(bootstrap, monkeypatch):
    """A provider key echoed inside an exception message is the exact leak the pipe must not carry."""
    def execute(frame, channel):
        raise RuntimeError('upstream rejected key sk-live-SECRET-123')

    monkeypatch.setattr(bootstrap, 'execute', execute)
    channel = FakeChannel()
    bootstrap.serve(channel)
    detail = channel.frames[-1][1]['cause']['detail']
    assert 'sk-live-SECRET-123' not in detail and '***' in detail


def test_a_clean_turn_sends_no_error_frame(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap, 'execute',
                        lambda frame, channel: channel.send('result', result={'final_response': 'done'}))
    channel = FakeChannel()
    assert bootstrap.serve(channel) == 0
    assert [kind for kind, _ in channel.frames] == ['hello', 'result']


def test_loss_cause_bounds_what_it_returns(bootstrap):
    """The owner's frame stays small: a huge message is cut, its traceback is not attached."""
    long = 'x' * 5000
    cause = bootstrap.loss_cause(RuntimeError(long), 'execute', None)
    assert len(cause['detail']) == 300
    assert 'Traceback' not in str(cause)
