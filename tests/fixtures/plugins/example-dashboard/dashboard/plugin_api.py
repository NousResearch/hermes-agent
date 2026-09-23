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
