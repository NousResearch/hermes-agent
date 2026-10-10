"""Holographic prefetch() must carry each fact's id so fact_feedback can close the trust loop.

The plugin's own ``fact_feedback`` tool requires a ``fact_id`` and its ``system_prompt_block`` is
the only place that tells the model to call it, yet the lines ``prefetch()`` injected before each
turn were ``- [trust] content`` — the id was dropped. A fact the model used through passive
recall could therefore never be rated, so ``record_feedback`` (+0.05 helpful / -0.10 unhelpful)
only ever ran for facts the model had first fetched through the ``fact_store`` tool, whose JSON
results do carry ``fact_id``. This is an internal inconsistency of the plugin, not a
``MemoryProvider`` contract change: ``prefetch()`` still returns formatted recall text.
"""

import json
import re

import pytest

from plugins.memory.holographic import HolographicMemoryProvider

_INJECTED_ID_RE = re.compile(r"\[id (\d+)")


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64})
    p.initialize(session_id="test-session")
    yield p
    p.shutdown()


def _trust_of(provider, fact_id):
    return next(f["trust_score"] for f in provider._store.list_facts(limit=100) if f["fact_id"] == fact_id)


def test_prefetch_carries_fact_id_that_fact_feedback_accepts(provider):
    added = json.loads(provider.handle_tool_call(
        "fact_store", {"action": "add", "content": "Rob prefers the Kanagawa colour scheme in Neovim", "category": "user_pref"}))
    fact_id = added["fact_id"]
    trust_before = _trust_of(provider, fact_id)

    injected = provider.prefetch("which Neovim colour scheme does Rob prefer")

    assert "Kanagawa" in injected, injected
    recalled_ids = [int(m) for m in _INJECTED_ID_RE.findall(injected)]
    assert fact_id in recalled_ids, f"injected recall does not carry the fact id:\n{injected}"

    # The id parsed from the injected text is exactly what the plugin's own tool needs.
    feedback = json.loads(provider.handle_tool_call(
        "fact_feedback", {"action": "helpful", "fact_id": recalled_ids[recalled_ids.index(fact_id)]}))

    assert feedback["fact_id"] == fact_id
    assert feedback["new_trust"] > feedback["old_trust"] == trust_before
    assert _trust_of(provider, fact_id) == feedback["new_trust"]


def test_system_prompt_block_names_the_id_marker_prefetch_emits(provider):
    provider.handle_tool_call("fact_store", {"action": "add", "content": "Rob prefers the Kanagawa colour scheme in Neovim"})

    injected = provider.prefetch("Neovim colour scheme")
    block = provider.system_prompt_block()

    marker = _INJECTED_ID_RE.search(injected)
    assert marker is not None, injected
    # The instruction that tells the model to call fact_feedback must describe the marker the
    # injected lines actually carry, otherwise the model has no id to pass.
    assert "fact_feedback" in block
    assert "[id" in block, block
