"""Offline regression tests for the maps skill's geographic calculations."""

import importlib.util
import math
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills" / "productivity" / "maps" / "scripts" / "maps_client.py"
)
spec = importlib.util.spec_from_file_location("maps_client", SCRIPT)
maps_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maps_client)


def test_antipodal_distance_is_half_the_earth_circumference():
    expected = math.pi * 6_371_000
    for latitude in (-82, -70, 0, 70, 82):
        distance = maps_client.haversine_m(latitude, 0, -latitude, 180)
        assert math.isfinite(distance)
        assert distance == pytest.approx(expected, rel=0, abs=1)


def test_coincident_and_ordinary_distances_are_unchanged():
    assert maps_client.haversine_m(51.5, -0.1, 51.5, -0.1) == 0
    expected = 6_371_000 * math.pi / 180
    forward = maps_client.haversine_m(0, 0, 0, 1)
    reverse = maps_client.haversine_m(0, 1, 0, 0)
    assert forward == pytest.approx(expected)
    assert reverse == pytest.approx(forward)
