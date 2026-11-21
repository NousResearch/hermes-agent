from __future__ import annotations

from pm.environment import _base_environment
from pm.index_config import is_forwarded, is_index_redirect


def test_uv_transport_concurrency_and_retries_are_forwarded():
    assert is_forwarded("UV_CONCURRENT_DOWNLOADS")
    assert is_forwarded("UV_HTTP_RETRIES")
    assert not is_index_redirect("UV_CONCURRENT_DOWNLOADS")
    assert not is_index_redirect("UV_HTTP_RETRIES")

    env = {
        "UV_CONCURRENT_DOWNLOADS": "2",
        "UV_HTTP_RETRIES": "5",
        "UV_HTTP_TIMEOUT": "60",
        "UV_OTHER_UNTRUSTED": "ignore_me",
    }
    base = _base_environment(env)
    assert base.get("UV_CONCURRENT_DOWNLOADS") == "2"
    assert base.get("UV_HTTP_RETRIES") == "5"
    assert base.get("UV_HTTP_TIMEOUT") == "60"
    assert "UV_OTHER_UNTRUSTED" not in base
