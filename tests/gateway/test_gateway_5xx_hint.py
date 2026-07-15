"""Hermetic contract guard for the gateway 502/503/504 error hint.

When the agent turn fails with an upstream 5xx (502/503/504), the gateway
error handler appends a transient-unavailability hint so the user knows to
retry rather than reading a raw traceback. The mapping lives inline in the
GatewayRunner error handler; this pins the code→hint contract without driving
the full inbound-message pipeline.
"""

import inspect

from gateway.run import GatewayRunner


def test_5xx_codes_map_to_transient_hint():
    src = inspect.getsource(GatewayRunner._handle_message_with_agent)
    # The three gateway/proxy 5xx codes are handled together.
    assert "status_code in {502, 503, 504}" in src
    # ...and produce a user-facing "transient, retry" hint, not a raw error.
    assert "temporarily unavailable (5xx)" in src


def test_5xx_hint_is_distinct_from_overload_hint():
    """529 (overloaded) and 5xx (unavailable) stay separate branches."""
    src = inspect.getsource(GatewayRunner._handle_message_with_agent)
    assert "status_code == 529" in src
    assert src.index("status_code == 529") < src.index("status_code in {502, 503, 504}")
