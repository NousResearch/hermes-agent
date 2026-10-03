"""Bounded real-signal comparison spanning stream creation and consumption."""
import signal
import threading
import time
import pytest
from agent import relay_llm
from tests.agent.test_relay_llm import relay_turn


def test_signal_during_blocked_provider(relay_turn):
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    workers, close_errors = [], []
    def provider(request):
        try:
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(5), 'fixture release failed'
            yield {'choices':[{'index':0,'delta':{'content':'done'}}]}
        finally:
            exited.set()
    def interrupt(signum, frame):
        assert entered.is_set(), 'provider never started'
        raise KeyboardInterrupt('interrupt during provider read')
    previous = signal.signal(signal.SIGALRM, interrupt)
    timer = threading.Timer(1.5, release.set)
    timer.start()
    stream = None
    start = time.monotonic()
    try:
        signal.setitimer(signal.ITIMER_REAL, .5)
        with pytest.raises(KeyboardInterrupt):
            stream = relay_llm.stream({'model':'test','messages':[{'role':'user','content':'hi'}]},provider,
                session_id='session-1',name='custom',model_name='test',metadata={'api_mode':'chat_completions'},finalizer=lambda:{})
            next(stream)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        release.set()
        timer.join(3)
        if stream is not None:
            try:
                stream.close()
            except BaseException as exc:
                close_errors.append(repr(exc))
    assert time.monotonic() - start < 4
    assert not close_errors, close_errors
    assert exited.wait(3), 'provider did not finish'
    for worker in workers:
        if worker is not threading.current_thread():
            worker.join(3)
    assert all(worker.is_alive() is False for worker in workers if worker is not threading.current_thread())
