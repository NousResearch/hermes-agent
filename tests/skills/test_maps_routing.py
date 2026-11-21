"""Verify that both routing commands query the graph for the requested mode."""
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "skills/productivity/maps/scripts/maps_client.py"


@pytest.mark.parametrize("command,steps", [("distance", "false"), ("directions", "true")])
@pytest.mark.parametrize("mode,endpoint", [
    ("driving", "https://router.project-osrm.org/route/v1/driving"),
    ("walking", "https://routing.openstreetmap.de/routed-foot/route/v1/driving"),
    ("cycling", "https://routing.openstreetmap.de/routed-bike/route/v1/driving"),
])
def test_route_uses_requested_mode(command, steps, mode, endpoint, monkeypatch):
    handler = runpy.run_path(str(SCRIPT))[f"cmd_{command}"]
    geocode = Mock(side_effect=[(51.5, -0.1, "Start"), (51.6, -0.2, "End")])
    http_get = Mock(return_value={"code": "Ok", "routes": [{"distance": 100, "duration": 60}]})
    monkeypatch.setitem(handler.__globals__, "geocode_single", geocode)
    monkeypatch.setitem(handler.__globals__, "http_get", http_get)

    handler(SimpleNamespace(origin=["Start"], to=["End"], mode=mode))

    http_get.assert_called_once_with(f"{endpoint}/-0.1,51.5;-0.2,51.6?overview=false&steps={steps}")
