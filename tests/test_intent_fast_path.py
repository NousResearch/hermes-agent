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
            # True positives that MUST keep matching after the HIGH-1/HIGH-2 fix.
            "weather in New York City",
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

    # ── Adversarial-review regressions ─────────────────────────────────────
    # HIGH-1 (connector-path false positives) + HIGH-2 (filler+connector
    # over-matching).  These were CONFIRMED LIVE to leak through and get a bogus
    # weather answer (e.g. "forecast for the meeting" geocoded to Nenagh,
    # Ireland; "forecast in the lab" to Indiana).  A false answer is far worse
    # than a missed match, so each of these MUST return no-match / fall through.
    ADVERSARIAL_FALSE_POSITIVES = [
        "forecast for the meeting",
        "forecast in the lab",
        "weather report for the Q3 sales",
        "forecast budget for next quarter",
        "is it raining in the stock market",
        "weather in my code",
        "weather permitting can we meet",
        "how's the weather in the stock market",
    ]

    @pytest.mark.parametrize("text", ADVERSARIAL_FALSE_POSITIVES)
    def test_adversarial_false_positives_do_not_match(self, text):
        """Non-weather prose with a connector/filler must NOT match.

        Why: the connector path used to accept ANY trailing noun as a location;
        the filler path used to allow a noun-filler to bridge to a connector.
        What: assert the matcher returns False so the agent keeps ownership.
        Test: each confirmed-live false positive returns False.
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
        "text",
        TestWeatherMatcher.ADVERSARIAL_FALSE_POSITIVES,
    )
    def test_adversarial_false_positive_extracts_no_location(self, text):
        """Connector prose must not yield a geocodable location.

        Why: even if a future matcher tweak let one of these through, the handler
        must still refuse to geocode the prose.
        What: assert _extract_location returns None for every confirmed-live
        false positive, so the named-location branch is never entered.
        Test: each adversarial phrase -> None.
        """
        assert ifp._extract_location(text) is None

    def test_connector_location_ok_guard(self):
        """The connector-location guard accepts places, rejects prose.

        Why: this guard is the backstop behind the matcher restructure.
        What: real places (incl. ZIP-bearing) pass; "the X" / stopword nouns fail.
        Test: Denver/New York City/Woodstock, IL 60098 -> True; the meeting/
        my code/the stock market/budget for next quarter -> False.
        """
        assert ifp._connector_location_ok("Denver") is True
        assert ifp._connector_location_ok("New York City") is True
        assert ifp._connector_location_ok("Woodstock, IL 60098") is True
        assert ifp._connector_location_ok("the meeting") is False
        assert ifp._connector_location_ok("my code") is False
        assert ifp._connector_location_ok("the stock market") is False
        assert ifp._connector_location_ok("budget for next quarter") is False
        assert ifp._connector_location_ok(None) is False
        assert ifp._connector_location_ok("") is False

    def test_is_place_like_rejects_overlong_single_token(self):
        """A single absurdly long token can't be treated as a place.

        Why: LOW fix — guard `_is_place_like` against a >60-char single token.
        What: a 70-char alpha token returns False; a short one returns True.
        Test: 'a'*70 -> False; 'Denver' -> True.
        """
        assert ifp._is_place_like("a" * 70) is False
        assert ifp._is_place_like("Denver") is True

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
# Handler — hard timeout ceiling (named-location branch)
# ──────────────────────────────────────────────────────────────────────────
class _SlowClient:
    """AsyncClient stand-in whose first ``get`` sleeps past the hard ceiling.

    Why: exercise the ``asyncio.wait_for`` wall-clock ceiling around the named
    (geocode+forecast) branch without real network.
    """

    def __init__(self, sleep_s, **_kwargs):
        self._sleep_s = sleep_s

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        import asyncio as _asyncio

        await _asyncio.sleep(self._sleep_s)  # blows the ceiling
        return _FakeResponse(payload=_FORECAST_OK)


class TestHandlerHardTimeout:
    """The named-location branch must honour a hard wall-clock ceiling."""

    @pytest.mark.asyncio
    async def test_named_branch_respects_hard_ceiling(self, monkeypatch):
        """A slow geocode is cut off by the hard ceiling -> None, fast.

        Why: MEDIUM fix — two stacked per-call timeouts could exceed the
        sub-second promise; the branch is wrapped in asyncio.wait_for.
        What: shrink the ceiling, make the client sleep longer than it, assert
        the handler returns None and returns within a small multiple of the
        ceiling (so it really was cut off, not run to completion).
        Test: ceiling=0.1s, client sleeps 5s -> None in well under 5s.
        """
        import time

        import httpx

        monkeypatch.setattr(ifp, "_NAMED_BRANCH_CEILING_S", 0.1)
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: _SlowClient(5.0, **kw)
        )

        t0 = time.monotonic()
        out = await ifp._weather_handler("what's the weather in Denver")
        elapsed = time.monotonic() - t0

        assert out is None
        assert elapsed < 1.0, f"ceiling not enforced; took {elapsed:.2f}s"


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
