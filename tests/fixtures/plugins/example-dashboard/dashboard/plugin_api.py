"""Example dashboard plugin — backend API routes (test fixture).

This plugin lives under ``tests/fixtures/plugins/`` so it is NOT shipped as
part of the bundled-plugins set; a stock hermes-agent install does not see
an "Example" tab in its sidebar. The ``_install_example_plugin`` pytest
fixture in ``tests/hermes_cli/test_web_server.py`` copies this directory
into ``$HERMES_HOME/plugins/example-dashboard/`` and forces the dashboard
plugin discovery cache to rescan, so tests that need a stable, side-effect-
free GET endpoint to verify plugin API auth + static-asset behaviour can
hit ``/api/plugins/example/hello`` (and ``/dashboard-plugins/example/
manifest.json``) without depending on any production-facing plugin.

Mounted at /api/plugins/example/ by the dashboard plugin system.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/hello")
async def hello():
    """Simple greeting endpoint to demonstrate plugin API routes."""
    return {"message": "Hello from the example plugin!", "plugin": "example", "version": "1.0.0"}


@router.get("/whoami")
async def whoami():
    """Side-effect-free credential probe: read a secret and fold any failure into the
    plugin "no data" contract (a plugin handler must never raise out to the client).

    Tests use this to prove the *production* mount path (discovery → import →
    ``_mount_plugin_api_routes`` → scoped handler) resolves the caller's profile
    credentials under multi-profile hosting (#120310). It reads a made-up key, so it
    never touches real credentials or the network.
    """
    from agent.secret_scope import get_secret

    try:
        return {"ok": True, "key": get_secret("EXAMPLE_PLUGIN_PROBE_KEY")}
    except Exception as exc:  # plugin contract: never raise out of the handler
        return {"ok": False, "error": type(exc).__name__}
