"""AsyncSessionDB offload facade for SessionDB.

Extracted verbatim from hermes_state.py (slice R5-C9, epic #78647) so the
async door onto SessionDB can live without importing hermes_state (which
would be a cycle). hermes_state re-imports AsyncSessionDB from here for
backward compatibility.

Contract: every method call is offloaded via asyncio.to_thread so a
blocking SQLite call never freezes the event loop. Generic forwarder —
the audit confirms no method returns a live cursor/generator.
"""

import asyncio

class AsyncSessionDB:
    """Async door onto SessionDB: offloads each call via asyncio.to_thread so a blocking SQLite call never freezes the event loop. Generic forwarder — the audit confirms no method returns a live cursor/generator."""

    def __init__(self, db: "SessionDB") -> None:
        self._db = db

    def __getattr__(self, name: str):
        attr = getattr(self._db, name)
        if not callable(attr):
            return attr

        async def _offloaded(*args, **kwargs):
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _offloaded
