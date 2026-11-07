"""Tests for skills/productivity/maps/scripts/maps_client.py (no network)."""

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "maps"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import maps_client as mc


# ---------------------------------------------------------------------------
# OSRM routing: every mode goes to its own FOSSGIS server
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode, server, profile", [
    ("driving", "routed-car", "driving"),
    ("walking", "routed-foot", "foot"),
    ("cycling", "routed-bike", "bike"),
])
def test_osrm_route_url_uses_per_mode_server(mode, server, profile):
    url = mc.osrm_route_url(mode, 52.5, 13.4, 52.6, 13.5, steps=False)
    assert url == (
        f"https://routing.openstreetmap.de/{server}/route/v1/{profile}/"
        "13.4,52.5;13.5,52.6?overview=false&steps=false"
    )


def test_osrm_route_url_steps_flag():
    url = mc.osrm_route_url("walking", 1, 2, 3, 4, steps=True)
    assert url.endswith("steps=true")


def test_distance_walking_requests_foot_server(monkeypatch, capsys):
    monkeypatch.setattr(mc, "geocode_single",
                        lambda q: (52.5, 13.4, q) if q == "A" else (52.51, 13.41, q))
    seen = []

    def fake_get(url, *a, **kw):
        seen.append(url)
        return {"code": "Ok", "routes": [{"distance": 1900.0, "duration": 1500.0}]}

    monkeypatch.setattr(mc, "http_get", fake_get)
    mc.cmd_distance(argparse.Namespace(origin=["A"], to=["B"], mode="walking"))

    assert len(seen) == 1
    assert "/routed-foot/route/v1/foot/" in seen[0]
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "walking"
    assert out["duration_minutes"] == 25.0
