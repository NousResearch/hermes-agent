"""Regression test for #102592: ``serve``/``dashboard`` must run plugin
discovery at startup.

The web/serve runtime never routes through ``_prepare_agent_startup()``'s
``_AGENT_COMMANDS`` gate, so ``discover_plugins()`` never ran in this process
and every ``ctx.register_hook(...)`` plugin (``pre_llm_call``,
observability, ...) was silently inert on web-chat and desktop-backend turns
while working fine in the CLI. ``start_server()`` now calls the idempotent
``discover_plugins()`` as a startup step.
"""

from __future__ import annotations

import hermes_cli.web_server as web_server
from tests.hermes_cli.test_dashboard_auth_gate import _stub_uvicorn_run


def test_serve_startup_runs_plugin_discovery(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        "hermes_cli.plugins.discover_plugins",
        lambda *args, **kwargs: calls.append(True),
    )
    _stub_uvicorn_run(monkeypatch)

    web_server.start_server(
        host="127.0.0.1", port=0, open_browser=False, headless=True
    )

    assert calls, (
        "serve/dashboard startup must run discover_plugins() — without it "
        "every ctx.register_hook(...) plugin is silently inert on web-chat "
        "and desktop-backend turns (#102592)"
    )
