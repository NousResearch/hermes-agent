"""Tests for the weather optional skill.

All network calls are mocked — no live requests are made.
"""

import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SKILL_DIR = TESTS_DIR / ".." / ".." / "optional-skills" / "productivity" / "weather"
SKILL_DIR = SKILL_DIR.resolve()
SCRIPTS_DIR = SKILL_DIR / "scripts"
SKILL_MD = SKILL_DIR / "SKILL.md"

sys.path.insert(0, str(SCRIPTS_DIR))

import weather  # noqa: E402


# ---------------------------------------------------------------------------
# Canned Open-Meteo responses
# ---------------------------------------------------------------------------

GEOCODE_HIT = {
    "results": [
        {
            "name": "Berlin",
            "latitude": 52.52437,
            "longitude": 13.41053,
            "country": "Germany",
            "admin1": "Berlin",
        }
    ]
}

GEOCODE_MISS = {"generationtime_ms": 0.5}

FORECAST_METRIC = {
    "latitude": 52.52,
    "longitude": 13.42,
    "current": {
        "time": "2026-07-21T12:00",
        "temperature_2m": 21.4,
        "apparent_temperature": 22.9,
        "relative_humidity_2m": 65,
        "weather_code": 3,
        "wind_speed_10m": 12.3,
        "wind_direction_10m": 180,
        "precipitation": 0.0,
    },
    "daily": {
        "time": ["2026-07-21", "2026-07-22"],
        "weather_code": [61, 95],
        "temperature_2m_max": [24.1, 19.8],
        "temperature_2m_min": [15.2, 13.0],
        "precipitation_sum": [2.4, 8.1],
        "precipitation_probability_max": [55, 90],
        "wind_speed_10m_max": [18.0, 32.5],
    },
}


def _fake_urlopen_factory(responses):
    """Return a fake urlopen yielding canned JSON bodies per URL substring."""

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        for key, payload in responses.items():
            if key in url:
                body = json.dumps(payload).encode("utf-8")
                cm = mock.MagicMock()
                cm.__enter__.return_value = io.BytesIO(body)
                cm.__exit__.return_value = False
                return cm
        raise AssertionError(f"Unexpected URL requested: {url}")

    return fake_urlopen


# ---------------------------------------------------------------------------
# WMO code mapping
# ---------------------------------------------------------------------------

class TestWmoMapping:
    def test_known_codes(self):
        assert weather.wmo_description(0) == "Clear sky"
        assert weather.wmo_description(3) == "Overcast"
        assert weather.wmo_description(61) == "Slight rain"
        assert weather.wmo_description(95) == "Thunderstorm"
        assert weather.wmo_description(99) == "Thunderstorm with heavy hail"

    def test_unknown_code(self):
        assert "Unknown" in weather.wmo_description(42)

    def test_non_numeric_code(self):
        assert weather.wmo_description(None) == "Unknown"


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

class TestGeocoding:
    def test_geocode_hit(self):
        fake = _fake_urlopen_factory({"geocoding-api": GEOCODE_HIT})
        with mock.patch.object(weather.urllib.request, "urlopen", fake):
            place = weather.geocode("Berlin")
        assert place["name"] == "Berlin"
        assert place["latitude"] == pytest.approx(52.52437)

    def test_geocode_miss_raises(self):
        fake = _fake_urlopen_factory({"geocoding-api": GEOCODE_MISS})
        with mock.patch.object(weather.urllib.request, "urlopen", fake):
            with pytest.raises(LookupError):
                weather.geocode("Nowhereville12345")

    def test_geocode_miss_exit_code_and_stderr(self, capsys):
        fake = _fake_urlopen_factory({"geocoding-api": GEOCODE_MISS})
        with mock.patch.object(weather.urllib.request, "urlopen", fake):
            rc = weather.main(["Nowhereville12345"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "City not found" in captured.err


# ---------------------------------------------------------------------------
# Forecast formatting + unit conversion
# ---------------------------------------------------------------------------

class TestForecast:
    def _run(self, argv):
        fake = _fake_urlopen_factory(
            {"geocoding-api": GEOCODE_HIT, "/v1/forecast": FORECAST_METRIC}
        )
        with mock.patch.object(weather.urllib.request, "urlopen", fake):
            return weather.main(argv)

    def test_text_output(self, capsys):
        rc = self._run(["Berlin", "--days", "2"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Weather for Berlin, Berlin, Germany" in out
        assert "Now: Overcast, 21 degC (feels 23 degC)" in out
        assert "wind 12 km/h S" in out
        assert "2026-07-21: Slight rain, 15/24 degC" in out
        assert "2026-07-22: Thunderstorm" in out
        assert "precip 8.1 mm (90%)" in out

    def test_multiword_city_joined(self):
        fake = _fake_urlopen_factory(
            {"geocoding-api": GEOCODE_HIT, "/v1/forecast": FORECAST_METRIC}
        )
        requested = []
        original = fake

        def spy(req, timeout=None):
            requested.append(req.full_url)
            return original(req, timeout=timeout)

        with mock.patch.object(weather.urllib.request, "urlopen", spy):
            rc = weather.main(["New", "York"])
        assert rc == 0
        geo_url = next(u for u in requested if "geocoding-api" in u)
        assert "New+York" in geo_url or "New%20York" in geo_url

    def test_imperial_units_requested_and_labeled(self, capsys):
        fake_responses = {
            "geocoding-api": GEOCODE_HIT,
            "/v1/forecast": FORECAST_METRIC,
        }
        requested = []
        base = _fake_urlopen_factory(fake_responses)

        def spy(req, timeout=None):
            requested.append(req.full_url)
            return base(req, timeout=timeout)

        with mock.patch.object(weather.urllib.request, "urlopen", spy):
            rc = weather.main(["Berlin", "--units", "imperial"])
        assert rc == 0
        forecast_url = next(u for u in requested if "/v1/forecast" in u)
        assert "temperature_unit=fahrenheit" in forecast_url
        assert "wind_speed_unit=mph" in forecast_url
        assert "precipitation_unit=inch" in forecast_url
        out = capsys.readouterr().out
        assert "degF" in out
        assert "mph" in out

    def test_metric_units_not_overridden(self):
        requested = []
        base = _fake_urlopen_factory(
            {"geocoding-api": GEOCODE_HIT, "/v1/forecast": FORECAST_METRIC}
        )

        def spy(req, timeout=None):
            requested.append(req.full_url)
            return base(req, timeout=timeout)

        with mock.patch.object(weather.urllib.request, "urlopen", spy):
            weather.main(["Berlin"])
        forecast_url = next(u for u in requested if "/v1/forecast" in u)
        assert "fahrenheit" not in forecast_url

    def test_json_format(self, capsys):
        rc = self._run(["Berlin", "--format", "json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["location"]["name"] == "Berlin"
        assert payload["forecast"]["daily"]["weather_code"] == [61, 95]

    def test_days_out_of_range_rejected(self):
        with pytest.raises(SystemExit):
            weather.main(["Berlin", "--days", "17"])

    def test_network_error_exit_code(self, capsys):
        def boom(req, timeout=None):
            raise weather.urllib.error.URLError("connection refused")

        with mock.patch.object(weather.urllib.request, "urlopen", boom):
            rc = weather.main(["Berlin"])
        assert rc == 1
        assert "network request failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Wind compass helper
# ---------------------------------------------------------------------------

class TestWindCompass:
    @pytest.mark.parametrize(
        "deg,label",
        [(0, "N"), (90, "E"), (180, "S"), (270, "W"), (359, "N"), (45, "NE")],
    )
    def test_compass_points(self, deg, label):
        assert weather.wind_compass(deg) == label

    def test_invalid_direction(self):
        assert weather.wind_compass(None) == "?"


# ---------------------------------------------------------------------------
# SKILL.md frontmatter
# ---------------------------------------------------------------------------

class TestFrontmatter:
    @pytest.fixture(scope="class")
    def frontmatter(self):
        yaml = pytest.importorskip("yaml")
        text = SKILL_MD.read_text(encoding="utf-8")
        assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
        block = text.split("---", 2)[1]
        return yaml.safe_load(block)

    def test_files_exist(self):
        assert SKILL_MD.is_file()
        assert (SCRIPTS_DIR / "weather.py").is_file()

    def test_required_fields(self, frontmatter):
        assert frontmatter["name"] == "weather"
        assert frontmatter["version"] == "0.1.0"
        assert frontmatter["author"] == "Hermes Agent"
        assert frontmatter["license"] == "MIT"

    def test_description_constraints(self, frontmatter):
        desc = frontmatter["description"]
        assert isinstance(desc, str)
        assert len(desc) <= 60
        assert desc.endswith(".")

    def test_platforms(self, frontmatter):
        assert frontmatter["platforms"] == ["linux", "macos", "windows"]

    def test_hermes_metadata(self, frontmatter):
        hermes = frontmatter["metadata"]["hermes"]
        assert hermes["tags"] == ["Weather", "Forecast", "Utilities"]
        assert hermes["related_skills"] == []
