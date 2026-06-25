"""Compaction-summarizer timeout/retry resilience (PRD 2026-06-25).

Root cause (Phase-0): the anthropic transport adapter
(`_AnthropicCompletionsAdapter.create`) SILENTLY DROPPED the `timeout` kwarg, so
the aux `compression` timeout never applied on the claude-app compaction route —
the native client used anthropic's 900s default read timeout, and with the SDK's
default `max_retries=2` a held-open relay turned ONE bounded 504 into a ~30-min
wedge that the gateway idle-timeout then killed.

Fix: thread the caller's `timeout` through the anthropic adapter, and for the
COMPACTION ROUTE ONLY apply `max_retries=0` (route-scoped, non-mutating, via the
SDK's `.with_options()`), so a wedged relay fails fast and LCM's existing L3
degrade is reached. Other aux tasks keep their default SDK retries.
"""
import time
from types import SimpleNamespace

import pytest

from agent.auxiliary_client import _AnthropicCompletionsAdapter


class _FakeMessages:
    """Records the per-request options the adapter applied via with_options()."""

    def __init__(self, owner):
        self._owner = owner

    def create(self, **anthropic_kwargs):
        self._owner.create_calls.append(anthropic_kwargs)
        # Return an Anthropic-shaped response the adapter can normalize.
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12),
        )


class _FakeAnthropic:
    """Minimal stand-in for anthropic.Anthropic with a non-mutating with_options."""

    def __init__(self, max_retries=2, timeout=900.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.with_options_calls = []
        self.create_calls = []
        self.messages = _FakeMessages(self)

    def with_options(self, **opts):
        self.with_options_calls.append(opts)
        # SDK semantics: returns a COPY with overrides; original unchanged.
        copy = _FakeAnthropic(
            max_retries=opts.get("max_retries", self.max_retries),
            timeout=opts.get("timeout", self.timeout),
        )
        # share the call-recording lists so the test can inspect either
        copy.create_calls = self.create_calls
        copy.with_options_calls = self.with_options_calls
        return copy


def _make_adapter(real):
    return _AnthropicCompletionsAdapter(real, model="claude-opus-4-8", is_oauth=False)


def test_anthropic_adapter_threads_timeout_for_compaction():
    """RED: the adapter currently DROPS timeout. After the fix, a compaction call
    must apply the caller's timeout to the native client (via with_options or a
    per-request timeout), not silently use the 900s default."""
    real = _FakeAnthropic()
    adapter = _make_adapter(real)
    adapter.create(
        messages=[{"role": "user", "content": "summarize"}],
        max_tokens=512,
        timeout=60.0,
        _aux_task="compression",
    )
    # the caller's 60s timeout must reach the native client somehow:
    applied_timeout = any(
        opts.get("timeout") == 60.0 for opts in real.with_options_calls
    ) or any(
        ak.get("timeout") == 60.0 for ak in real.create_calls
    )
    assert applied_timeout, (
        "compaction timeout was dropped — adapter must thread it to the native client"
    )


def test_anthropic_adapter_sets_max_retries_zero_for_compaction():
    """RED: compaction route must run with max_retries=0 so a held-open relay
    fails fast instead of the SDK retrying (×2) and wedging the turn."""
    real = _FakeAnthropic()
    adapter = _make_adapter(real)
    adapter.create(
        messages=[{"role": "user", "content": "summarize"}],
        max_tokens=512,
        timeout=60.0,
        _aux_task="compression",
    )
    assert any(opts.get("max_retries") == 0 for opts in real.with_options_calls), (
        "compaction route must apply max_retries=0 (route-scoped, via with_options)"
    )


def test_anthropic_adapter_does_NOT_zero_retries_for_other_tasks():
    """AC-13: route-scoped. A NON-compaction aux task (vision/title-gen) must NOT
    get max_retries=0 — it keeps the SDK's transient-retry resilience."""
    real = _FakeAnthropic()
    adapter = _make_adapter(real)
    adapter.create(
        messages=[{"role": "user", "content": "describe"}],
        max_tokens=512,
        timeout=60.0,
        _aux_task="vision",
    )
    assert not any(opts.get("max_retries") == 0 for opts in real.with_options_calls), (
        "non-compaction aux tasks must NOT be forced to max_retries=0 (AC-13)"
    )


def test_with_options_is_non_mutating():
    """AC-13 by construction: with_options returns a copy; the shared cached
    client's max_retries is unchanged after a compaction call."""
    real = _FakeAnthropic(max_retries=2)
    adapter = _make_adapter(real)
    adapter.create(
        messages=[{"role": "user", "content": "summarize"}],
        max_tokens=512,
        timeout=60.0,
        _aux_task="compression",
    )
    assert real.max_retries == 2, "shared client must be unchanged (non-mutating)"


def test_no_task_means_no_retry_override():
    """Back-compat: a call with no _aux_task behaves like a normal aux call
    (no max_retries override forced)."""
    real = _FakeAnthropic()
    adapter = _make_adapter(real)
    adapter.create(
        messages=[{"role": "user", "content": "x"}],
        max_tokens=64,
        timeout=30.0,
    )
    assert not any(opts.get("max_retries") == 0 for opts in real.with_options_calls)


# ── _scope_client_for_task: the route-scoping dispatcher (call-site half) ──

class _FakeOpenAIClient:
    """OpenAI-wire stand-in with a non-mutating with_options."""

    def __init__(self, max_retries=2):
        self.max_retries = max_retries
        self.with_options_calls = []

    def with_options(self, **opts):
        self.with_options_calls.append(opts)
        c = _FakeOpenAIClient(max_retries=opts.get("max_retries", self.max_retries))
        c.with_options_calls = self.with_options_calls
        return c


def test_scope_openai_client_left_unchanged():
    """The OpenAI-wire path is intentionally NOT client-scoped (it already honors
    `timeout` from kwargs, the incident is anthropic-transport-only, and swapping
    the client identity would break the shared retry path). It returns unchanged
    with no extra kwargs."""
    from agent.auxiliary_client import _scope_client_for_task
    c = _FakeOpenAIClient(max_retries=2)
    scoped, extra = _scope_client_for_task(c, "compression", 60.0)
    assert scoped is c, "OpenAI-wire client must be returned unchanged"
    assert not c.with_options_calls, "must not swap the OpenAI client identity"
    assert extra == {}, "no _aux_task for the OpenAI path"


def test_scope_openai_client_other_task_untouched():
    """AC-13: a non-compaction task must not be scoped on OpenAI either."""
    from agent.auxiliary_client import _scope_client_for_task
    c = _FakeOpenAIClient(max_retries=2)
    scoped, extra = _scope_client_for_task(c, "vision", 60.0)
    assert scoped is c, "non-compaction task returns the client unchanged"
    assert not c.with_options_calls
    assert extra == {}


def test_scope_anthropic_wrapper_injects_aux_task_not_with_options():
    """The Anthropic wrapper hides its inner client, so the dispatcher passes
    `_aux_task` (the adapter scopes the inner client); it must NOT call
    with_options on the wrapper."""
    from agent.auxiliary_client import _scope_client_for_task, AnthropicAuxiliaryClient
    real = _FakeAnthropic()
    wrapper = AnthropicAuxiliaryClient(real, model="claude-opus-4-8",
                                       api_key="k", base_url="http://x/anthropic")
    scoped, extra = _scope_client_for_task(wrapper, "compression", 60.0)
    assert scoped is wrapper
    assert extra == {"_aux_task": "compression"}


def test_scope_never_raises_on_quirky_client():
    """Robustness: a client whose with_options raises must degrade to unchanged,
    never break the summary call."""
    from agent.auxiliary_client import _scope_client_for_task

    class _Boom:
        def with_options(self, **o):
            raise RuntimeError("nope")
    c = _Boom()
    scoped, extra = _scope_client_for_task(c, "compression", 60.0)
    assert scoped is c
    assert extra == {}


# ── AC-12: the cancel primitive aborts a held-open socket + no fd leak ──

# (no external timeout plugin; the test self-bounds)
def test_ac12_httpx_timeout_aborts_held_open_socket_no_fd_leak():
    """The real relay wedge holds the socket open sending no bytes. Prove the
    per-request timeout= ABORTS it within budget and leaks no fds across N calls."""
    import os
    import socket
    import threading
    from openai import OpenAI, APITimeoutError

    stop = threading.Event()
    ready = threading.Event()
    port_box: list = []
    accepts: list = []

    def _server():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        port_box.append(srv.getsockname()[1])
        ready.set()
        srv.settimeout(0.3)
        while not stop.is_set():
            try:
                c, _ = srv.accept()
            except socket.timeout:
                continue
            accepts.append(1)
            try:
                c.settimeout(0.2)
                try:
                    c.recv(65536)
                except Exception:
                    pass
            except Exception:
                pass
            # Hold just long enough that the CLIENT must time out (4s > client's 3s
            # budget), then close server-side so the test's fd delta reflects only
            # any CLIENT-side leak, not the rig's accumulating server sockets.
            def _hold_then_close(conn):
                time.sleep(4.0)
                try:
                    conn.close()
                except Exception:
                    pass
            threading.Thread(target=_hold_then_close, args=(c,), daemon=True).start()
        srv.close()

    t = threading.Thread(target=_server, daemon=True)
    t.start()
    ready.wait(5)
    base_url = f"http://127.0.0.1:{port_box[0]}/v1"

    def _fds():
        try:
            return len(os.listdir("/dev/fd"))
        except Exception:
            return -1

    client = OpenAI(api_key="sk-test", base_url=base_url, max_retries=0, timeout=3)
    N = 4
    fds_before = _fds()
    for _ in range(N):
        t0 = time.monotonic()
        with pytest.raises(APITimeoutError):
            client.chat.completions.create(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4,
            )
        assert time.monotonic() - t0 <= 3 + 2.5, "request must abort within budget+ε"
    import gc
    gc.collect()
    time.sleep(5.0)  # let the server's held sockets (4s) close so the delta is client-only
    fds_after = _fds()
    stop.set()
    t.join(timeout=2)

    assert len(accepts) == N, f"max_retries=0 => exactly N accepts, got {len(accepts)}"
    # fd-count must be stable across N fallovers (no per-call socket leak).
    assert fds_after <= fds_before + 2, f"fd leak: {fds_before} -> {fds_after} across {N} calls"

