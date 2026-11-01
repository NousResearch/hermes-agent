"""Load-progress watcher: the SSE byte stream, disconnect clearing and reconnect.

``_watch`` is driven synchronously with the response faked only at
``urllib.request.urlopen``. The real line parser, JSON decoding, ``_apply_event``,
percent calculation, ``get_loading_progress`` and snapshot clearing all run.
Observations are recorded inside the fakes and asserted afterwards, because
``_watch`` swallows ``Exception`` raised from within its read loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_cli.local_runtime.load_progress as lp

_MODEL_A = "Qwën-模型-A"
_MODEL_B = "Other-B"


class _Stop(BaseException):
    """Private sentinel: escapes _watch's broad ``except Exception``."""


def _record(model: str, value: float) -> bytes:
    import json

    msg = {"model": model, "event": "status_change",
           "data": {"status": "loading",
                    "progress": {"stages": ["text_model"], "current": "text_model", "value": value}}}
    return b"data: " + json.dumps(msg, ensure_ascii=False).encode("utf-8") + b"\n\n"


class _Response:
    """Context-manager reader; ends with EOF or OSError once its chunks run out."""

    def __init__(self, chunks, end, observe):
        self._chunks = list(chunks)
        self._end = end
        self._observe = observe
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def read1(self, _size):
        self._observe()
        if self._chunks:
            return self._chunks.pop(0)
        if self._end == "oserror":
            raise OSError("connection reset")
        return b""


@pytest.mark.parametrize("end", ["eof", "oserror"])
def test_stream_parses_split_events_and_disconnect_clears_then_reconnect_is_fresh(monkeypatch, end):
    monkeypatch.setattr(lp, "_snapshot", {})
    monkeypatch.setattr(lp, "_endpoint", lambda: ("http://synthetic.invalid", "k"))
    monkeypatch.setattr(lp, "_ensure_watcher", lambda: None)

    # Split the first record inside the 2-byte "ë" and across line boundaries.
    first = _record(_MODEL_A, 0.5)
    cut = first.index("ë".encode()) + 1
    chunks_1 = [
        b": keepalive\ndata: {not json\n",
        first[:cut],
        first[cut:-2],           # complete JSON, but no newline yet
        first[-2:-1],            # newline completing the data line
        first[-1:],              # blank separator line
    ]
    chunks_2 = [_record(_MODEL_B, 0.75)]

    seen_reads: list[list[dict]] = [[], []]
    responses: list[_Response] = []
    queue = [(chunks_1, seen_reads[0]), (chunks_2, seen_reads[1])]

    def fake_urlopen(req, timeout=None):
        if not queue:
            raise _Stop("unexpected extra connection")
        chunks, sink = queue.pop(0)
        resp = _Response(chunks, end, lambda: sink.append(lp.get_loading_progress()))
        responses.append(resp)
        return resp

    at_backoff: list[dict] = []

    def fake_sleep(_seconds):
        at_backoff.append(lp.get_loading_progress())
        if len(at_backoff) >= 2:
            raise _Stop()

    monkeypatch.setattr(lp.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(lp, "time", SimpleNamespace(monotonic=lp.time.monotonic, sleep=fake_sleep))

    with pytest.raises(_Stop):
        lp._watch()

    # First connection: nothing until the newline completes the record, then model A at 50%.
    first_reads = seen_reads[0]
    assert len(first_reads) == len(chunks_1) + 1
    loaded_a = {_MODEL_A: {"stage": "text_model", "value": 0.5, "percent": 50}}
    assert first_reads[:4] == [{}, {}, {}, {}]
    assert first_reads[4:] == [loaded_a, loaded_a]  # after the newline, and at the terminating read

    # Dead connection: snapshot cleared before backoff.
    assert at_backoff[0] == {}

    # Reconnect: fresh model only, no leftover from the old stream.
    second_reads = seen_reads[1]
    assert second_reads[-1] == {_MODEL_B: {"stage": "text_model", "value": 0.75, "percent": 75}}
    assert all(_MODEL_A not in snap for snap in second_reads)
    assert at_backoff[1] == {}

    assert len(responses) == 2
    assert all(r.closed for r in responses)
