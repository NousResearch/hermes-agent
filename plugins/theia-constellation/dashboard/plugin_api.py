"""Theia Constellation — backend API routes.

Mounted at /api/plugins/theia-constellation/ by the dashboard plugin system.

Provides graph data for the constellation panel. Currently reads from a
static graph.json file; future versions will build the graph dynamically
from Hermes session data.
"""

import json
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

# Look for graph data in these locations (first found wins):
#   1. Plugin data dir: ~/.hermes/plugins/theia-constellation/data/graph.json
#   2. Bundled example:  <plugin_dir>/../data/graph.json
#   3. Theia project:    ~/projects/hermes-hackathon-seshviz/theia/examples/graph.json

_GRAPH_SEARCH_PATHS = [
    Path.home() / ".hermes" / "plugins" / "theia-constellation" / "data" / "graph.json",
    Path(__file__).parent.parent / "data" / "graph.json",
    Path.home() / "projects" / "hermes-hackathon-seshviz" / "theia" / "examples" / "graph.json",
]


def _find_graph() -> dict | None:
    """Locate and load the graph JSON file."""
    for path in _GRAPH_SEARCH_PATHS:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


@router.get("/graph")
async def get_graph(stats: int = 0):
    """Return the constellation graph data.

    If ?stats=1 is passed, returns the full graph (used by the plugin
    to show node/edge counts). Otherwise returns the full graph for
    the panel iframe.
    """
    graph = _find_graph()
    if graph is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No graph data found. Place graph.json in ~/.hermes/plugins/theia-constellation/data/"},
        )
    return graph
