"""Regression tests for the Contexto memory provider plugin.

Requires a running self-hosted Contexto stack on CONTEXTO_BASE_URL
(default http://localhost:4010). Skips if unreachable.

Covers:
  - subagent / cron / flush contexts must not ingest, prefetch, or register
  - primary context still ingests normally
  - on_delegation captures (task, result) pairs as curated parent turns
  - recall finds delegation content
  - nested-subagent (subagent calling on_delegation) is also no-op
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

from plugins.memory.contexto import ContextoMemoryProvider


_BASE = os.environ.get("CONTEXTO_BASE_URL", "http://localhost:4010")


def _selfhost_alive() -> bool:
    try:
        httpx.get(f"{_BASE}/v1/agents", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _selfhost_alive(),
    reason=f"Contexto selfhost not reachable at {_BASE}",
)


@pytest.fixture
def slug() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def primary(slug: str) -> ContextoMemoryProvider:
    p = ContextoMemoryProvider()
    p.initialize(session_id="prim", agent_context="primary",
                 agent_identity=slug, user_id="alex")
    yield p
    p.shutdown()


@pytest.fixture
def subagent(slug: str) -> ContextoMemoryProvider:
    s = ContextoMemoryProvider()
    s.initialize(session_id="sub", agent_context="subagent",
                 agent_identity=slug, user_id="alex")
    yield s
    s.shutdown()


def test_subagent_skips_all_writes(subagent: ContextoMemoryProvider):
    """Subagent context must not spawn any threads or fire any HTTP."""
    subagent.queue_prefetch("anything")
    subagent.sync_turn("massive subagent output " * 100, "result " * 100)
    subagent.on_delegation(task="t", result="r")
    assert subagent._sync_thread is None
    assert subagent._prefetch_thread is None


@pytest.mark.parametrize("ctx", ["cron", "flush"])
def test_other_nonprimary_contexts_skip(slug: str, ctx: str):
    p = ContextoMemoryProvider()
    p.initialize(session_id="x", agent_context=ctx,
                 agent_identity=slug, user_id="alex")
    p.sync_turn("hi", "hello")
    assert p._sync_thread is None
    p.shutdown()


def test_primary_ingests(primary: ContextoMemoryProvider):
    primary.sync_turn("My favorite color is octarine.", "Got it — octarine.")
    assert primary._sync_thread is not None
    primary._sync_thread.join(timeout=120)
    assert not primary._sync_thread.is_alive()


def test_on_delegation_captures_subagent_result(primary: ContextoMemoryProvider, slug: str):
    primary.on_delegation(
        task="research how the rate limiter handles clock skew",
        result="Use time.monotonic() locally and Redis TTL for cross-pod coordination.",
        child_session_id="sub-session-id",
    )
    primary._sync_thread.join(timeout=120)
    time.sleep(2)  # let server index

    result = primary._client.search(
        "rate limiter clock skew", agent=slug, user_id="alex",
    )
    wm = result.get("workingMemory", [])
    assert wm, "delegation result should be recallable"
    contents = " ".join(it.get("content", "") for it in wm)
    assert "clock skew" in contents.lower() or "monotonic" in contents.lower()
