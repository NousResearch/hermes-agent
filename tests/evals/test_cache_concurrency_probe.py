"""The live cache-concurrency probe's native-wire recorder must run.

``patched_stream`` wraps ``anthropic`` ``Messages.stream`` for the
``anthropic_messages`` wire and records one row per API call, keyed by the
``[probe-session N]`` tag the probe puts in each session's first message. The
probe is a script (argument parsing and patching happen at import), so it is
loaded with ``runpy`` and zero workers: nothing reaches the network.
"""

import runpy
import sys
import tempfile

import pytest

pytest.importorskip("anthropic")

import agent.prompt_caching as prompt_caching  # noqa: E402
from anthropic.resources.messages import Messages  # noqa: E402

MODULE = "evals.postmortem.live_ab.cache_concurrency_probe"


class _Stream:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def probe(tmp_path, monkeypatch):
    # The script patches these process-wide; monkeypatch restores them.
    monkeypatch.setattr(Messages, "stream", Messages.stream)
    monkeypatch.setattr(prompt_caching, "effective_cache_ttl", prompt_caching.effective_cache_ttl)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the probe seeds a mkdtemp workdir
    monkeypatch.setattr(sys, "argv", [
        MODULE, "--repo", ".", "--provider", "anthropic", "--api-key", "test-key",
        "--workers", "0", "--out", str(tmp_path / "probe.jsonl"),
    ])
    return runpy.run_module(MODULE, run_name="__probe__")


def test_native_stream_recorder_tags_the_worker_and_hashes_the_request(probe):
    probe["_orig_stream"] = lambda self, **kw: _Stream()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "[probe-session 7] read the files"}]},
        {"role": "assistant", "content": "ok"},
    ]

    ctx = probe["patched_stream"](None, messages=messages, system="sys", tools=[{"name": "read_file"}])

    rec = ctx.rec
    assert rec["worker"] == 7
    assert rec["n_msgs"] == len(messages)
    assert len(rec["msg_shas"]) == len(messages)
    other = probe["patched_stream"](None, messages=messages, system="other", tools=[{"name": "read_file"}]).rec
    assert other["system_sha"] != rec["system_sha"]
    assert other["tools_sha"] == rec["tools_sha"]
