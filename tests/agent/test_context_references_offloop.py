"""@-reference root resolution must not stat the filesystem on the event loop.

``Path.expanduser().resolve()`` walks the filesystem; on a slow or hung mount it stalls
the caller. ``preprocess_context_references_async`` therefore resolves the cwd and the
allowed root through ``asyncio.to_thread``.
"""

import threading

import pytest

from agent import context_references as refs_mod


@pytest.mark.asyncio
async def test_reference_roots_resolve_off_event_loop(monkeypatch, tmp_path):
    calls = []
    real = refs_mod._expand_and_resolve

    def _tracking(path):
        calls.append(threading.current_thread())
        return real(path)

    monkeypatch.setattr(refs_mod, "_expand_and_resolve", _tracking)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    loop_thread = threading.current_thread()

    result = await refs_mod.preprocess_context_references_async(
        "see @file:notes.txt", cwd=tmp_path, context_length=8192, allowed_root=tmp_path,
    )

    assert calls, "the root resolution never ran"
    assert all(thread is not loop_thread for thread in calls)
    assert "hello" in result.message
