"""ClawTeam dashboard plugin — backend API routes.

Mounted at /api/plugins/clawteam/ by the dashboard plugin system.

Thin shell over the `clawteam` CLI. Shared CLI driver lives at
`..._clawteam_cli` so the agent-facing tools and the dashboard
endpoints cannot drift on argv handling, name validation, or error
shape.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

from fastapi import APIRouter, HTTPException

# Hermes' dashboard loader (web_server.py:_mount_plugin_api_routes) loads
# this file via importlib.util.spec_from_file_location as a standalone
# module — there is no enclosing package, so `from .._clawteam_cli` would
# raise ImportError and the route would never mount. Load the sibling
# helper by file path instead.
_HELPER_PATH = pathlib.Path(__file__).resolve().parent.parent / "_clawteam_cli.py"
_spec = importlib.util.spec_from_file_location("_hermes_clawteam_cli", _HELPER_PATH)
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
CliError = _helper.CliError
run_clawteam_json = _helper.run_clawteam_json
validate_name = _helper.validate_name

router = APIRouter()


def _wrap(exc: CliError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.get("/teams")
def list_teams() -> dict[str, Any]:
    """List all discoverable teams."""
    try:
        teams = run_clawteam_json("team", "discover")
    except CliError as exc:
        raise _wrap(exc)
    return {"teams": teams or []}


@router.get("/teams/{name}")
def team_status(name: str) -> dict[str, Any]:
    """Return full status for a single team (members, recent activity)."""
    try:
        name = validate_name(name, field="team name")
        # `--` sentinel: belt-and-braces against future clawteam flag
        # parsers treating a positional that slips past the regex as
        # an option.
        status = run_clawteam_json("team", "status", "--", name)
    except CliError as exc:
        raise _wrap(exc)
    return {"team": status}
