"""The gateway makes its process environment its own before anything reads it.

``hermes serve`` sets ``HERMES_SERVE_HEADLESS=1`` in its own process, so every shell under it inherits the flag.
A gateway started from such a shell mounted its Webapp headless and answered every page "web UI disabled".
"""

import os

import pytest


def test_an_inherited_serve_headless_flag_is_dropped(monkeypatch):
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    from gateway.run_bootstrap import _start_gateway_own_process_env

    _start_gateway_own_process_env()

    assert "HERMES_SERVE_HEADLESS" not in os.environ
    assert os.environ["HERMES_EXEC_ASK"] == "1"


def test_without_the_flag_nothing_else_changes(monkeypatch):
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    before = {k: v for k, v in os.environ.items() if k != "HERMES_EXEC_ASK"}
    from gateway.run_bootstrap import _start_gateway_own_process_env

    _start_gateway_own_process_env()

    assert {k: v for k, v in os.environ.items() if k != "HERMES_EXEC_ASK"} == before


class _Called(Exception):
    pass


class _TooLate(Exception):
    pass


@pytest.mark.asyncio
async def test_start_gateway_settles_its_environment_first(monkeypatch):
    """Every entry point (``hermes gateway run``, ``python -m gateway.run``, the CLI) runs through
    ``start_gateway``; it must settle the environment before config loading, the listener or the
    dashboard mount can read it. The next startup step raises its own sentinel, so a start that skips
    the helper fails at once instead of booting a real gateway inside the test."""
    import gateway.run_bootstrap as bootstrap
    import hermes_cli.update_proxy_env as proxy_env

    def called():
        raise _Called

    def too_late():
        raise _TooLate

    monkeypatch.setattr(bootstrap, "_start_gateway_own_process_env", called)
    monkeypatch.setattr(proxy_env, "restore_user_proxy_env", too_late)
    with pytest.raises(_Called):
        await bootstrap.start_gateway()
