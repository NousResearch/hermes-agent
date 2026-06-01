"""Unit tests for intent_fast_path (the pre-LLM weather fast-path).

Why: The fast-path bypasses the agent for weather questions; a regression that
makes the matcher too greedy (hijacking real conversation) or that returns a
partial/wrong string instead of falling through would silently degrade the
gateway.  These tests pin both the matching boundary and the strict
fall-through contract.

What: Covers the matcher (positives/negatives), location cleaning, handler
fall-through on timeout / HTTP 500 / empty geocoding, and a canned-success
render.  All network is monkeypatched — no real HTTP in the unit suite.

Test: ``pytest tests/test_intent_fast_path.py``.
"""

from __future__ import annotations

import pytest

import intent_fast_path as ifp


# ──────────────────────────────────────────────────────────────────────────
# Matcher
# ──────────────────────────────────────────────────────────────────────────
class TestWeatherMatcher:
    """intent_fast_path._weather_matcher — the gate."""

    @pytest.mark.parametrize(
        "text",
        [
            "weather",
            "weather woodstock il",
            "what's the weather in Denver",
            "whats the weather like in Denver?",
            "is it raining",
            "is it snowing in chicago",
            "forecast",
            "temperature in austin tx",
            "weather today",
            "weather for Woodstock, IL 60098",
        ],
    )
    def test_positive_matches(self, text):
        """Weather questions must match.

        Test: each parametrized weather phrase returns True.
        """
        assert ifp._weather_matcher(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "weather affects my mood",
            "/weather",
            "tell me a story about weather in fantasy worlds",
            "I love the weather here",
            "the weather was nice yesterday when we walked",
            "how does weather work",
            "",
        ],
    )
    def test_negative_matches(self, text):
        """Conversational / non-question text must NOT match.

        Test: each parametrized non-weather phrase returns False so the agent
        keeps ownership.
        """
        assert ifp._weather_matcher(text) is False


# ──────────────────────────────────────────────────────────────────────────
# Location extraction + cleaning
# ──────────────────────────────────────────────────────────────────────────
class TestLocation:
    """Location parsing and geocode-cleaning."""

    def test_default_location_is_none(self):
        """Bare 'weather' has no location (=> default Woodstock).

        Test: _extract_location('weather') is None.
        """
        assert ifp._extract_location("weather") is None

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("weather woodstock il", "woodstock il"),
            ("what's the weather in Denver", "Denver"),
            ("temperature in austin tx", "austin tx"),
        ],
    )
    def test_named_location_extracted(self, text, expected):
        """Named locations are pulled out verbatim (pre-clean).

        Test: parametrized phrase yields the expected raw location string.
        """
        assert ifp._extract_location(text) == expected

    def test_noise_location_rejected(self):
        """A too-short / letterless trailing token is rejected.

        Test: 'weather in 12' -> None (no letters), defers to agent.
        """
        assert ifp._extract_location("weather in 12") is None

    @pytest.mark.parametrize(
        "raw,clean",
        [
            ("Woodstock, IL 60098", "Woodstock"),
            ("Denver, CO", "Denver"),
            ("New York", "New York"),
            ("austin tx", "austin"),
        ],
    )
    def test_clean_for_geocode(self, raw, clean):
        """City, ST ZIP is reduced to bare city for Open-Meteo geocoding.

        Test: parametrized messy location cleans to the bare city token.
        """
        assert ifp._clean_location_for_geocode(raw) == clean


# ──────────────────────────────────────────────────────────────────────────
# httpx fakes
# ──────────────────────────────────────────────────────────────────────────
class _FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class _FakeClient:
    """Async context-manager stand-in for httpx.AsyncClient.

    ``responder`` maps a URL-substring to a _FakeResponse, raises on timeout, or
    is a callable(url) -> _FakeResponse.  Used to script geocode + forecast
    legs independently.
    """

    def __init__(self, responder, **_kwargs):
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        return self._responder(url)


def _install_httpx(monkeypatch, responder):
    """Patch intent_fast_path's httpx.AsyncClient with a scripted fake.

    Why: handler imports httpx locally; patch the attribute on the imported
    module so AsyncClient(timeout=...) returns our fake.
    Test: callers pass a responder closure; handler then routes through it.
    """
    import httpx

    def _factory(**kwargs):
        return _FakeClient(responder, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


_FORECAST_OK = {
    "current": {
        "temperature_2m": 62.3,
        "weathercode": 0,
        "windspeed_10m": 5.1,
    },
    "daily": {
        "weathercode": [3, 61, 0],
        "temperature_2m_max": [78.0, 70.0, 75.0],
        "temperature_2m_min": [51.0, 55.0, 49.0],
    },
}

_GEOCODE_OK = {
    "results": [
        {
            "latitude": 39.7392,
            "longitude": -104.9847,
            "name": "Denver",
            "admin1": "Colorado",
            "country": "United States",
        }
    ]
}


# ──────────────────────────────────────────────────────────────────────────
# Handler — strict fall-through
# ──────────────────────────────────────────────────────────────────────────
class TestHandlerFallThrough:
    """The handler must return None on ANY failure (never partial/wrong text)."""

    @pytest.mark.asyncio
    async def test_timeout_falls_through(self, monkeypatch):
        """A request timeout yields None (defer to agent).

        Test: responder raises httpx.TimeoutException -> handler returns None.
        """
        import httpx

        def _responder(url):
            raise httpx.TimeoutException("simulated timeout")

        _install_httpx(monkeypatch, _responder)
        assert await ifp._weather_handler("weather") is None

    @pytest.mark.asyncio
    async def test_http_500_falls_through(self, monkeypatch):
        """An HTTP 5xx on the forecast leg yields None.

        Test: default-location forecast returns status 500 -> None.
        """
        def _responder(url):
            return _FakeResponse(status_code=500, payload={})

        _install_httpx(monkeypatch, _responder)
        assert await ifp._weather_handler("weather") is None

    @pytest.mark.asyncio
    async def test_empty_geocoding_falls_through(self, monkeypatch):
        """Empty geocoding results yield None (unknown place -> agent).

        Test: named-location query, geocode returns {'results': []} -> None.
        """
        def _responder(url):
            if "geocoding-api" in url:
                return _FakeResponse(payload={"results": []})
            return _FakeResponse(payload=_FORECAST_OK)  # should never be reached

        _install_httpx(monkeypatch, _responder)
        assert await ifp._weather_handler("weather in Zxqwffville") is None

    @pytest.mark.asyncio
    async def test_missing_current_temp_falls_through(self, monkeypatch):
        """A forecast payload missing current.temperature_2m yields None.

        Test: forecast lacks current temp -> None (no partial answer).
        """
        broken = {"current": {"weathercode": 0}, "daily": {}}

        def _responder(url):
            return _FakeResponse(payload=broken)

        _install_httpx(monkeypatch, _responder)
        assert await ifp._weather_handler("weather") is None


# ──────────────────────────────────────────────────────────────────────────
# Handler — success render
# ──────────────────────────────────────────────────────────────────────────
class TestHandlerSuccess:
    """A clean payload renders a terse Fahrenheit reply with a 3-day forecast."""

    @pytest.mark.asyncio
    async def test_default_location_success(self, monkeypatch):
        """Default Woodstock query renders without any geocoding call.

        Test: only a forecast response is scripted; output contains the default
        location name, current temp °F, the condition, and three day labels.
        """
        geocode_called = {"hit": False}

        def _responder(url):
            if "geocoding-api" in url:
                geocode_called["hit"] = True
                return _FakeResponse(payload={"results": []})
            return _FakeResponse(payload=_FORECAST_OK)

        _install_httpx(monkeypatch, _responder)
        out = await ifp._weather_handler("weather")

        assert geocode_called["hit"] is False  # default skips geocoding
        assert out is not None
        assert "Woodstock, IL" in out
        assert "62°F" in out
        assert "Clear" in out
        assert "wind 5 mph" in out
        assert "Today:" in out and "Tomorrow:" in out and "Day 3:" in out
        assert "51–78°F" in out  # today low–high

    @pytest.mark.asyncio
    async def test_named_location_success(self, monkeypatch):
        """Named Denver query geocodes then renders with the geocoded name.

        Test: geocode + forecast scripted; output uses 'Denver, Colorado,
        United States' and the canned current temp.
        """
        def _responder(url):
            if "geocoding-api" in url:
                return _FakeResponse(payload=_GEOCODE_OK)
            return _FakeResponse(payload=_FORECAST_OK)

        _install_httpx(monkeypatch, _responder)
        out = await ifp._weather_handler("what's the weather in Denver")

        assert out is not None
        assert "Denver, Colorado, United States" in out
        assert "62°F" in out
        assert out.count("\n") >= 4  # header + now + blank + 3 forecast lines


# ──────────────────────────────────────────────────────────────────────────
# Dispatch / registry
# ──────────────────────────────────────────────────────────────────────────
class TestDispatch:
    """_intent_fast_path dispatch and exception safety."""

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, monkeypatch):
        """Non-weather text returns None from the top-level dispatch.

        Test: 'tell me a joke' -> None (no intent matched).
        """
        out = await ifp._intent_fast_path("tell me a joke")
        assert out is None

    @pytest.mark.asyncio
    async def test_handler_exception_is_swallowed(self):
        """A handler that raises must not propagate; dispatch returns None.

        Test: register a matcher that fires + a handler that raises, then assert
        _intent_fast_path returns None instead of raising.  Registry is restored.
        """
        saved = list(ifp._INTENT_HANDLERS)
        try:
            async def _boom(_text):
                raise RuntimeError("boom")

            ifp.register_intent(lambda _t: True, _boom)
            # 'weather' would normally match the real handler first; use a
            # string that only our raising handler matches by clearing others.
            ifp._INTENT_HANDLERS[:] = [(lambda _t: True, _boom)]
            out = await ifp._intent_fast_path("anything")
            assert out is None
        finally:
            ifp._INTENT_HANDLERS[:] = saved
