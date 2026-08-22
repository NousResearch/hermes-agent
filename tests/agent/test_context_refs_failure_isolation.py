"""One failing @-reference must not abort the whole concurrent expansion (#91221).

Two layers of failure handling are pinned here:

1. Per-ref failures inside ``_expand_reference`` (a raising url fetcher, an
   unreadable file) are already converted to warnings by its own except-block;
   this test additionally pins that the SIBLING refs still land — under a bare
   gather any escape would abort the batch.
2. Anything that escapes ``_expand_reference`` itself (a non-standard
   exception from a helper) is contained by the gather's
   ``return_exceptions=True``: the failed ref surfaces as a warning, siblings
   complete, and the caller sees no exception. Before #91221 the gather was
   bare — one escape crashed the whole message AND leaked the still-running
   sibling coroutines (never awaited).
"""

from __future__ import annotations

import pytest

import agent.context_references as cr
from agent.context_references import preprocess_context_references_async


@pytest.mark.asyncio
async def test_failing_fetcher_surfaces_warning_siblings_land(tmp_path):
    """Three url refs, the middle fetcher raises: the other two still land
    and the failure surfaces as a per-ref warning (inner except-block)."""
    good = "https://good.example/x"

    async def fetcher(url: str) -> str:
        if "bad" in url:
            raise RuntimeError("simulated fetch failure")
        return f"CONTENT[{url}]"

    msg = f"see @url:{good} @url:https://bad.example/y @url:{good} please"
    res = await preprocess_context_references_async(
        msg, cwd=tmp_path, context_length=100_000, url_fetcher=fetcher
    )

    assert any("simulated fetch failure" in w for w in res.warnings), (
        f"the failing ref must surface as a warning, got {res.warnings!r}"
    )
    assert res.message.count("CONTENT[https://good.example/x]") == 2


@pytest.mark.asyncio
async def test_escaping_exception_is_contained_by_gather(tmp_path, monkeypatch):
    """An exception that escapes _expand_reference entirely (monkeypatched to
    raise) must not abort the batch: the failed ref becomes a warning, the
    sibling ref's block still lands, and no exception reaches the caller."""
    refs_seen: list[cr.ContextReference] = []
    real_expand = cr._expand_reference

    async def exploding_expand(ref, cwd, **kwargs):
        refs_seen.append(ref)
        if "bad" in (ref.target or ""):
            raise RuntimeError("escaped the inner except-block")
        return await real_expand(ref, cwd, **kwargs)

    monkeypatch.setattr(cr, "_expand_reference", exploding_expand)

    async def fetcher(url: str) -> str:
        return f"CONTENT[{url}]"

    good = "https://good.example/x"
    res = await preprocess_context_references_async(
        f"@url:{good} @url:https://bad.example/y",
        cwd=tmp_path,
        context_length=100_000,
        url_fetcher=fetcher,
    )

    assert len(refs_seen) == 2, "both sibling coroutines must have been awaited"
    assert any("expansion failed" in w for w in res.warnings)
    assert "CONTENT[https://good.example/x]" in res.message


@pytest.mark.asyncio
async def test_all_refs_failing_still_returns_result(tmp_path):
    """Every ref failing is not an error either — a warning for each."""
    async def fetcher(url: str) -> str:
        raise ValueError("boom")

    res = await preprocess_context_references_async(
        "@url:https://a.example @url:https://b.example",
        cwd=tmp_path,
        context_length=100_000,
        url_fetcher=fetcher,
    )
    assert sum("boom" in w for w in res.warnings) == 2


@pytest.mark.asyncio
async def test_cancelled_ref_propagates_cancellation(tmp_path, monkeypatch):
    """A CancelledError outcome (BaseException, not Exception on 3.8+) must
    propagate instead of falling into the (warning, block) unpack — which
    would raise an unrelated TypeError and corrupt the batch."""
    import asyncio

    real_expand = cr._expand_reference

    async def cancelled_expand(ref, cwd, **kwargs):
        if "bad" in (ref.target or ""):
            raise asyncio.CancelledError()
        return await real_expand(ref, cwd, **kwargs)

    monkeypatch.setattr(cr, "_expand_reference", cancelled_expand)

    async def fetcher(url: str) -> str:
        return f"CONTENT[{url}]"

    with pytest.raises(asyncio.CancelledError):
        await preprocess_context_references_async(
            "@url:https://good.example/x @url:https://bad.example/y",
            cwd=tmp_path,
            context_length=100_000,
            url_fetcher=fetcher,
        )
