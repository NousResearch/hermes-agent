"""Unit tests for maps_client.py pure functions."""

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add the scripts directory to the path so we can import maps_client
SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import maps_client as mc


# ── Haversine ────────────────────────────────────────────────────────────


class TestHaversine:
    def test_same_point_is_zero(self):
        assert mc.haversine_m(48.8584, 2.2945, 48.8584, 2.2945) == 0.0

    def test_known_distance_paris_lyon(self):
        # Paris to Lyon is ~393 km straight line
        dist = mc.haversine_m(48.8566, 2.3522, 45.7640, 4.8357)
        assert 390_000 < dist < 400_000

    def test_antipodal_points(self):
        # North pole to south pole ~20,000 km
        dist = mc.haversine_m(90, 0, -90, 0)
        assert 20_000_000 < dist < 20_100_000

    def test_equator_quarter(self):
        # 0,0 to 0,90 is ~10,000 km
        dist = mc.haversine_m(0, 0, 0, 90)
        assert 10_000_000 < dist < 10_100_000

    def test_symmetry(self):
        d1 = mc.haversine_m(40.7128, -74.0060, 51.5074, -0.1278)
        d2 = mc.haversine_m(51.5074, -0.1278, 40.7128, -74.0060)
        assert d1 == pytest.approx(d2)


# ── Overpass query builder ───────────────────────────────────────────────


class TestBuildOverpassQuery:
    def test_basic_query_structure(self):
        q = mc.build_overpass_nearby("amenity", "restaurant", 48.85, 2.29, 500, 10)
        assert "[out:json]" in q
        assert '"amenity"="restaurant"' in q
        assert "around:500,48.85,2.29" in q
        assert "out center 10" in q

    def test_contains_node_and_way(self):
        q = mc.build_overpass_nearby("tourism", "hotel", 40.0, -74.0, 1000, 5)
        assert "node[" in q
        assert "way[" in q

    def test_bbox_query_structure(self):
        q = mc.build_overpass_bbox("amenity", "cafe", 40.75, -74.00, 40.77, -73.98, 20)
        assert "[out:json]" in q
        assert '"amenity"="cafe"' in q
        assert "40.75,-74.0,40.77,-73.98" in q


# ── Category validation ──────────────────────────────────────────────────


class TestCategories:
    def test_original_12_categories_exist(self):
        original = [
            "restaurant", "cafe", "bar", "hospital", "pharmacy", "hotel",
            "supermarket", "atm", "gas_station", "parking", "museum", "park",
        ]
        for cat in original:
            assert cat in mc.CATEGORY_TAGS, f"Missing original category: {cat}"

    def test_new_categories_exist(self):
        new_cats = [
            "school", "university", "bank", "police", "fire_station",
            "library", "airport", "train_station", "bus_stop", "dentist",
            "doctor", "cinema", "theatre", "gym", "post_office",
            "convenience_store", "bakery", "nightclub", "zoo", "playground",
        ]
        for cat in new_cats:
            assert cat in mc.CATEGORY_TAGS, f"Missing new category: {cat}"

    def test_all_categories_have_valid_tags(self):
        for cat, tag in mc.CATEGORY_TAGS.items():
            assert isinstance(tag, tuple), f"{cat}: tag should be tuple"
            assert len(tag) == 2, f"{cat}: tag should be (key, value)"
            assert isinstance(tag[0], str) and isinstance(tag[1], str)

    def test_at_least_40_categories(self):
        assert len(mc.CATEGORY_TAGS) >= 40


# ── OSRM profiles ────────────────────────────────────────────────────────


class TestOSRMProfiles:
    def test_driving_walking_cycling(self):
        assert "driving" in mc.OSRM_PROFILES
        assert "walking" in mc.OSRM_PROFILES
        assert "cycling" in mc.OSRM_PROFILES

    def test_profile_mappings(self):
        assert mc.OSRM_PROFILES["driving"] == "driving"
        assert mc.OSRM_PROFILES["walking"] == "foot"
        assert mc.OSRM_PROFILES["cycling"] == "bike"


# ── Argparse ─────────────────────────────────────────────────────────────


class TestArgparse:
    def test_distance_uses_to_flag(self):
        """The distance command should use --to, not two positional nargs='+'."""
        parser = mc.build_parser()
        args = parser.parse_args(["distance", "Paris", "--to", "Lyon"])
        assert args.command == "distance"
        assert args.origin == ["Paris"]
        assert args.to == ["Lyon"]

    def test_distance_multiword_origin(self):
        parser = mc.build_parser()
        args = parser.parse_args(["distance", "New", "York", "--to", "Boston"])
        assert args.origin == ["New", "York"]
        assert args.to == ["Boston"]

    def test_directions_uses_to_flag(self):
        parser = mc.build_parser()
        args = parser.parse_args(["directions", "Big Ben", "--to", "Tower Bridge"])
        assert args.command == "directions"

    def test_search_accepts_query(self):
        parser = mc.build_parser()
        args = parser.parse_args(["search", "Eiffel", "Tower"])
        assert args.command == "search"
        assert args.query == ["Eiffel", "Tower"]

    def test_nearby_accepts_category(self):
        parser = mc.build_parser()
        args = parser.parse_args(["nearby", "48.85", "2.29", "restaurant"])
        assert args.command == "nearby"
        assert args.category == "restaurant"

    def test_bbox_accepts_coordinates(self):
        parser = mc.build_parser()
        args = parser.parse_args(["bbox", "40.75", "-74.00", "40.77", "-73.98", "cafe"])
        assert args.command == "bbox"
        assert args.category == "cafe"

    def test_area_accepts_query(self):
        parser = mc.build_parser()
        args = parser.parse_args(["area", "Manhattan"])
        assert args.command == "area"


# ── Output helpers ───────────────────────────────────────────────────────


class TestOutputHelpers:
    def test_print_json_outputs_valid_json(self, capsys):
        mc.print_json({"key": "value", "num": 42})
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["key"] == "value"
        assert data["num"] == 42

    def test_error_exit_outputs_error_json(self):
        with pytest.raises(SystemExit) as exc_info:
            mc.error_exit("something went wrong")
        assert exc_info.value.code == 1
