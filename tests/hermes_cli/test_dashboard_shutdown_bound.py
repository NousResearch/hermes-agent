"""The dashboard must bound uvicorn's graceful shutdown (#58005).

uvicorn defaults ``timeout_graceful_shutdown`` to ``None``, which waits
forever. A single still-open WebSocket then holds the process through SIGTERM
until the supervisor's kill timer fires, which looks like a hung dashboard to
systemd/Docker.
"""

from __future__ import annotations

import pytest


class _StopBeforeServing(Exception):
    """Abort start_server once the uvicorn config has been captured."""


def test_start_server_bounds_graceful_shutdown(monkeypatch):
    import uvicorn

    from hermes_cli import web_server

    captured = {}
    real_config = uvicorn.Config

    def _capture_config(*args, **kwargs):
        captured.update(kwargs)
        # Build the real object so any downstream attribute access stays honest,
        # then stop before a socket is ever bound.
        real_config(*args, **kwargs)
        raise _StopBeforeServing

    monkeypatch.setattr(uvicorn, "Config", _capture_config)

    with pytest.raises(_StopBeforeServing):
        web_server.start_server(host="127.0.0.1", port=0, open_browser=False, headless=True)

    timeout = captured.get("timeout_graceful_shutdown")
    assert timeout is not None, (
        "uvicorn.Config was built without timeout_graceful_shutdown; "
        "the dashboard would wait forever on SIGTERM"
    )
    assert 0 < timeout <= 30, f"graceful shutdown bound is not sane: {timeout!r}"
