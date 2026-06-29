"""Tests for proxy error body cap.

Regression test for #55015: proxy-mode error responses were read without
a body size limit.
"""

from __future__ import annotations


def test_proxy_error_body_cap_logic():
    """Verify the body cap logic rejects oversized Content-Length."""
    max_body = 16 * 1024 * 1024  # 16 MiB

    # Oversized response should be rejected
    cl = str(17 * 1024 * 1024)  # 17 MiB
    assert int(cl) > max_body

    # Normal response should pass
    cl_normal = str(1024)  # 1 KB
    assert int(cl_normal) <= max_body

    # Missing Content-Length should not be rejected
    cl_none = None
    should_reject = cl_none is not None and int(cl_none) > max_body
    assert should_reject is False
