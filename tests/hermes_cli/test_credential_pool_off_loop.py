"""Regression tests for the #91912 salvage — credential-pool handlers off-loop.

The 2026-08-22 incident: ``GET /api/credentials/pool`` ran ``load_pool()`` on
the uvicorn event loop; for a token-exchange provider that reaches
``urllib.request.urlopen`` whose ``timeout`` does not bound ``getaddrinfo``,
so a networkless host froze the whole dashboard backend for 17 minutes.
"""

from __future__ import annotations

import threading

import pytest

import hermes_cli.web_routers.ops as _rt_ops

# ---------------------------------------------------------------------------
# web_server credential-pool handlers off the loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_credential_pool_runs_off_event_loop(monkeypatch):
    import hermes_cli.auth as auth_mod

    loop_thread = threading.get_ident()
    seen = {}

    def fake_read_pool(*args, **kwargs):
        seen["thread"] = threading.get_ident()
        return {}

    monkeypatch.setattr(auth_mod, "read_credential_pool", fake_read_pool)
    result = await _rt_ops.list_credential_pool()

    assert result == {"providers": []}
    assert seen["thread"] != loop_thread
