"""Extracted APIRouter modules for the dashboard web server.

Each module exposes ``router = APIRouter()`` (profiles additionally exposes
``sessions_router``) and is mounted by ``hermes_cli.web_server`` at the exact
point in module execution where the routes were originally registered, so
route-matching order is unchanged.  Shared web_server helpers/state are
reached through the late-binding seam in ``hermes_cli.web_deps``.
"""
