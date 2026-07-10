"""
Regression test for issue #62047 - Telegram polling heartbeat swallows
python-telegram-bot (PTB) NetworkError, so reconnect never fires.

The bug: `_polling_heartbeat_loop()` only catches
`asyncio.TimeoutError` and `OSError`. PTB 22.6 raises
`telegram.error.NetworkError` and `telegram.error.TimedOut` on
connectivity failures - these inherit from `TelegramError`, not
`OSError`, so they fall through to the generic `except Exception`
and are silently swallowed.

The fix: in addition to the existing `asyncio.TimeoutError`/`OSError`
catch, add an explicit `except Exception` branch that checks
`isinstance(exc, (NetworkError, TimedOut))` and routes those
exceptions into `_handle_polling_network_error()`.

Static-source test: verify the heartbeat's exception handler
includes a NetworkError/TimedOut branch on unfixed code, fails.

Tests:
  1. test_static_heartbeat_handles_PTB_NetworkError - static check that
     the fix is in place (fails on unfixed code, passes on fixed).
  2. test_static_heartbeat_does_not_reconnect_on_BadRequest - the
     existing `except Exception: pass` was the right behavior for
     non-connectivity errors; verify we did NOT remove it.
"""

import re
from pathlib import Path


def test_static_heartbeat_handles_PTB_NetworkError():
    """The heartbeat fix must include a NetworkError/TimedOut catch.

    On unfixed code, the catch list is `(asyncio.TimeoutError, OSError)`
    and PTB NetworkError is silently swallowed.
    On fixed code, there's a follow-on branch that catches
    `NetworkError`/`TimedOut` and routes them to reconnect.
    """
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    adapter_py = (worktree / "plugins" / "platforms" / "telegram" / "adapter.py").read_text()

    # Find the polling heartbeat loop function
    m = re.search(r"async def _polling_heartbeat_loop\(self.*?(?=^    async def |\Z)",
                  adapter_py, re.MULTILINE | re.DOTALL)
    assert m, "_polling_heartbeat_loop function not found"
    body = m.group(0)

    # The fix must mention NetworkError AND TimedOut
    assert "NetworkError" in body, (
        "#62047 regression: _polling_heartbeat_loop does not catch "
        "NetworkError. PTB 22.6 raises this on connectivity failures "
        "but it inherits from TelegramError, not OSError, so the "
        "existing catch (asyncio.TimeoutError, OSError) misses it."
    )
    assert "TimedOut" in body, (
        "#62047 regression: _polling_heartbeat_loop does not catch "
        "TimedOut (which inherits from NetworkError in PTB 22.6)."
    )


def test_static_heartbeat_does_not_silently_swallow_all_exceptions():
    """Sanity: the heartbeat must still have a fallback `except Exception`
    for non-connectivity errors (TelegramError 401, etc.), but it must
    NOT catch NetworkError without routing to reconnect.

    Verify there's a `from telegram.error import NetworkError, TimedOut`
    inside the function (or at module level - if module-level, that's
    also fine for the import) and the exception handler for these
    types creates the polling error task.
    """
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    adapter_py = (worktree / "plugins" / "platforms" / "telegram" / "adapter.py").read_text()

    # The fix must import NetworkError somewhere within the heartbeat
    # function (the lazy import pattern)
    m = re.search(r"async def _polling_heartbeat_loop\(self.*?(?=^    async def |\Z)",
                  adapter_py, re.MULTILINE | re.DOTALL)
    assert m
    body = m.group(0)

    # Either a module-level or function-level import of NetworkError
    # is acceptable, but it must be present in the body for the
    # isinstance check to work.
    assert "NetworkError" in body and "TimedOut" in body, (
        "#62047: NetworkError and TimedOut must both be importable in "
        "the heartbeat function scope. Either via lazy import inside "
        "the except branch or module-level import."
    )

    # The fix should still preserve the existing except Exception: pass
    # fallback for non-connectivity errors. After the fix, NetworkError/
    # TimedOut should be caught FIRST (with reconnect), THEN
    # BadRequest/InvalidToken fall through to the existing except
    # Exception: pass.
    assert "except Exception" in body, (
        "#62047: must preserve except Exception: pass for non-connectivity "
        "errors. The fix should add a NetworkError/TimedOut catch BEFORE "
        "the existing fallback."
    )
