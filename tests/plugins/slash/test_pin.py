"""Unit tests for the 📌 pin → Atlas ingest handler — Plan 026-C.

Acceptance criteria covered:

AC1 — Reaction handler wired in gateway. Verified by ``test_handler_happy_path``:
      a ``reaction_added`` event with ``reaction='pushpin'`` from an
      allowlisted user produces a captured POST to ``/v1/ingest`` with
      the expected payload shape (raw_text per-message + provenance.source
      == ``slack_manual_pin``).

AC2 — Tests pass. This module covers:
        - emoji filter (only ``pushpin`` / ``📌`` trip the handler)
        - ``SLACK_ALLOWED_USERS`` gate (fail-closed + wildcard)
        - 3-message thread → 3 message lines in raw_text
        - Single-message pin (no ``thread_ts``)
        - /v1/ingest 500 → friendly error reply, no silent drop
        - /v1/ingest network error → friendly error reply
        - Slack thread-fetch failure → handler still ingests a placeholder

AC3 — Manual smoke instructions in plugin module (see ``MANUAL_SMOKE_TEST``
      constant). Verified by ``test_manual_smoke_test_present``.

AC4 — No regression on existing slash command tests. This file imports
      only ``plugins.slash.pin`` — sibling slash tests (``test_draft.py``,
      ``test_orchestrator.py``) are unaffected by additions.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from plugins.slash import pin as pin_mod
from plugins.slash.pin import (
    MANUAL_SMOKE_TEST,
    PIN_EMOJI_NAMES,
    PROVENANCE_SOURCE,
    PinContext,
    PinHandlerConfig,
    SlackMessage,
    build_ingest_payload,
    format_error_reply,
    format_success_reply,
    handle_pin_reaction,
    is_pin_reaction,
    is_user_allowed,
)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "emoji,expected",
    [
        ("pushpin", True),
        ("round_pushpin", True),
        ("📌", True),
        (":pushpin:", True),  # tolerate colon wrapping
        ("thumbsup", False),
        ("", False),
        ("pin", False),  # not a Slack canonical name
    ],
)
def test_is_pin_reaction(emoji, expected):
    assert is_pin_reaction(emoji) is expected


def test_pin_emoji_names_contains_pushpin():
    # Slack canonicalises 📌 → ``pushpin`` so this is the load-bearing
    # alias. If a future Slack change drops it the handler silently stops
    # working — guard with an explicit assert.
    assert "pushpin" in PIN_EMOJI_NAMES


def test_is_user_allowed_fail_closed_when_unset(monkeypatch):
    monkeypatch.delenv("SLACK_ALLOWED_USERS", raising=False)
    assert is_user_allowed("UBLAKE") is False


def test_is_user_allowed_explicit_id(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "UBLAKE,UOTHER")
    assert is_user_allowed("UBLAKE") is True
    assert is_user_allowed("UNOPE") is False
    assert is_user_allowed("") is False


def test_is_user_allowed_wildcard(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "*")
    assert is_user_allowed("U_ANYONE") is True


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


def _ctx(messages: List[SlackMessage]) -> PinContext:
    return PinContext(
        channel="C123",
        thread_ts="1700000000.000100",
        pinned_ts="1700000000.000300",
        pinned_by_user="UBLAKE",
        messages=tuple(messages),
    )


def test_build_ingest_payload_three_message_thread():
    msgs = [
        SlackMessage(ts="1700000000.000100", user="UBLAKE", text="kicking off the thread"),
        SlackMessage(ts="1700000000.000200", user="UGREG", text="sounds good"),
        SlackMessage(ts="1700000000.000300", user="UBLAKE", text="closing thought"),
    ]
    payload = build_ingest_payload(_ctx(msgs))

    # Required top-level keys.
    assert set(payload.keys()) == {"source", "raw_text", "provenance"}

    # Provenance carries the 026-C source tag + actor + channel.
    assert payload["source"]["connector"] == PROVENANCE_SOURCE
    assert payload["source"]["kind"] == "manual"
    assert payload["source"]["resource_id"] == "1700000000.000300"
    assert payload["source"]["run_id"] == "1700000000.000100"
    assert payload["provenance"]["actor"] == "UBLAKE"
    assert payload["provenance"]["ref"] == "C123"

    # raw_text contains exactly one chunk-line per Slack message so
    # Atlas's per-message chunker preserves attribution.
    body = payload["raw_text"]
    assert "kicking off the thread" in body
    assert "sounds good" in body
    assert "closing thought" in body
    # Speaker tags appear (downstream extractor uses them).
    assert "<@UBLAKE>" in body
    assert "<@UGREG>" in body


def test_build_ingest_payload_single_message():
    msgs = [SlackMessage(ts="1700000000.000300", user="UBLAKE", text="just this one")]
    payload = build_ingest_payload(_ctx(msgs))
    assert "just this one" in payload["raw_text"]


# ---------------------------------------------------------------------------
# Reply formatting
# ---------------------------------------------------------------------------


def test_format_success_reply_contains_urn():
    s = format_success_reply("job-abc-123")
    assert "urn:atlas:ingest:job-abc-123" in s
    assert s.startswith("✓")


def test_format_error_reply_is_visible():
    s = format_error_reply("boom")
    assert s.startswith("⚠ pin failed")
    assert "boom" in s


def test_format_error_reply_truncates_long_reasons():
    long = "x" * 500
    s = format_error_reply(long)
    assert len(s) < 300


# ---------------------------------------------------------------------------
# Fakes for the end-to-end handler test
# ---------------------------------------------------------------------------


@dataclass
class _CapturedReply:
    channel: str
    thread_ts: str
    text: str


class _FakeSlackClient:
    """Minimal async Slack WebClient stub.

    Returns canned thread-replies / history. Records all calls so tests
    can assert on them.
    """

    def __init__(
        self,
        *,
        replies: Optional[List[dict]] = None,
        history: Optional[List[dict]] = None,
        replies_raises: Optional[Exception] = None,
        history_raises: Optional[Exception] = None,
    ):
        self._replies = replies
        self._history = history
        self._replies_raises = replies_raises
        self._history_raises = history_raises
        self.calls: list[tuple[str, dict]] = []

    async def conversations_replies(self, **kwargs):
        self.calls.append(("conversations_replies", kwargs))
        if self._replies_raises is not None:
            raise self._replies_raises
        return {"messages": self._replies or []}

    async def conversations_history(self, **kwargs):
        self.calls.append(("conversations_history", kwargs))
        if self._history_raises is not None:
            raise self._history_raises
        return {"messages": self._history or []}


class _FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            # Mirror httpx's HTTPStatusError surface enough to satisfy
            # the handler's ``except Exception`` clause without dragging
            # in the full httpx exception class.
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse, *, raises: Optional[Exception] = None):
        self._response = response
        self._raises = raises
        self.posted: list[tuple[str, dict, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url: str, *, json: dict, headers: dict):
        self.posted.append((url, json, headers))
        if self._raises is not None:
            raise self._raises
        return self._response


class _FakeHttpxModule:
    def __init__(self, async_client: _FakeAsyncClient):
        self._client = async_client

    def AsyncClient(self, *a, **kw):  # noqa: N802 — mimics httpx surface
        return self._client


async def _capturing_reply(captured: list[_CapturedReply]):
    async def _reply(channel: str, thread_ts: str, text: str) -> None:
        captured.append(_CapturedReply(channel=channel, thread_ts=thread_ts, text=text))
    return _reply


# ---------------------------------------------------------------------------
# End-to-end handler tests
# ---------------------------------------------------------------------------


@pytest.fixture
def allowed_user(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "UBLAKE")
    yield "UBLAKE"


def _make_event(
    *,
    user: str = "UBLAKE",
    reaction: str = "pushpin",
    channel: str = "C123",
    item_ts: str = "1700000000.000300",
    thread_ts: Optional[str] = "1700000000.000100",
) -> dict:
    item: dict = {"channel": channel, "ts": item_ts, "type": "message"}
    if thread_ts is not None:
        item["thread_ts"] = thread_ts
    return {"reaction": reaction, "user": user, "item": item}


@pytest.mark.asyncio
async def test_handler_happy_path_three_message_thread(allowed_user):
    """📌 on a 3-message thread → /v1/ingest captured with all 3 messages."""
    slack = _FakeSlackClient(
        replies=[
            {"ts": "1700000000.000100", "user": "UBLAKE", "text": "msg one"},
            {"ts": "1700000000.000200", "user": "UGREG", "text": "msg two"},
            {"ts": "1700000000.000300", "user": "UBLAKE", "text": "msg three"},
        ],
    )
    fake_async = _FakeAsyncClient(_FakeResponse(200, {"job_id": "job-xyz"}))
    fake_httpx = _FakeHttpxModule(fake_async)
    captured: list[_CapturedReply] = []
    reply = await _capturing_reply(captured)

    job_id = await handle_pin_reaction(
        event=_make_event(),
        slack_client=slack,
        reply_in_thread=reply,
        config=PinHandlerConfig(
            atlas_base_url="http://atlas.test:8000",
            atlas_bearer="tok",
            httpx_module=fake_httpx,
        ),
    )

    assert job_id == "job-xyz"

    # One POST landed at /v1/ingest with the expected provenance.
    assert len(fake_async.posted) == 1
    url, body, headers = fake_async.posted[0]
    assert url == "http://atlas.test:8000/v1/ingest"
    assert headers["Authorization"] == "Bearer tok"
    assert body["source"]["connector"] == PROVENANCE_SOURCE
    assert body["provenance"]["actor"] == "UBLAKE"
    assert body["provenance"]["ref"] == "C123"
    # All three messages survived the wire format.
    assert "msg one" in body["raw_text"]
    assert "msg two" in body["raw_text"]
    assert "msg three" in body["raw_text"]

    # And one confirmation reply landed in-thread.
    assert len(captured) == 1
    assert captured[0].channel == "C123"
    assert captured[0].thread_ts == "1700000000.000100"
    assert "urn:atlas:ingest:job-xyz" in captured[0].text


@pytest.mark.asyncio
async def test_handler_drops_non_pin_emoji(allowed_user):
    """A 👍 reaction must NOT trigger any Slack / Atlas calls."""
    slack = _FakeSlackClient(replies=[])
    fake_async = _FakeAsyncClient(_FakeResponse(200, {"job_id": "should-not-fire"}))
    fake_httpx = _FakeHttpxModule(fake_async)
    captured: list[_CapturedReply] = []

    result = await handle_pin_reaction(
        event=_make_event(reaction="thumbsup"),
        slack_client=slack,
        reply_in_thread=await _capturing_reply(captured),
        config=PinHandlerConfig(
            atlas_base_url="http://atlas.test:8000",
            atlas_bearer="tok",
            httpx_module=fake_httpx,
        ),
    )

    assert result is None
    assert slack.calls == []
    assert fake_async.posted == []
    assert captured == []


@pytest.mark.asyncio
async def test_handler_drops_unauthorized_user(monkeypatch):
    """SLACK_ALLOWED_USERS gate — non-Blake 📌 reactions are ignored."""
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "UBLAKE")  # gate set; UEVE not on it
    slack = _FakeSlackClient(replies=[])
    fake_async = _FakeAsyncClient(_FakeResponse(200, {"job_id": "should-not-fire"}))
    fake_httpx = _FakeHttpxModule(fake_async)
    captured: list[_CapturedReply] = []

    result = await handle_pin_reaction(
        event=_make_event(user="UEVE"),
        slack_client=slack,
        reply_in_thread=await _capturing_reply(captured),
        config=PinHandlerConfig(
            atlas_base_url="http://atlas.test:8000",
            atlas_bearer="tok",
            httpx_module=fake_httpx,
        ),
    )

    assert result is None
    assert fake_async.posted == []
    assert captured == []  # silent drop is OK here — pinner had no permission


@pytest.mark.asyncio
async def test_handler_ingest_500_posts_error_reply(allowed_user):
    """Atlas /v1/ingest returning 500 → friendly in-thread error, no silent fail."""
    slack = _FakeSlackClient(
        replies=[
            {"ts": "1700000000.000300", "user": "UBLAKE", "text": "the message"},
        ],
    )
    fake_async = _FakeAsyncClient(_FakeResponse(500, {"detail": "atlas exploded"}))
    fake_httpx = _FakeHttpxModule(fake_async)
    captured: list[_CapturedReply] = []

    result = await handle_pin_reaction(
        event=_make_event(thread_ts=None),
        slack_client=slack,
        reply_in_thread=await _capturing_reply(captured),
        config=PinHandlerConfig(
            atlas_base_url="http://atlas.test:8000",
            atlas_bearer="tok",
            httpx_module=fake_httpx,
        ),
    )

    assert result is None
    assert len(captured) == 1
    assert captured[0].text.startswith("⚠ pin failed")


@pytest.mark.asyncio
async def test_handler_ingest_network_error_posts_error_reply(allowed_user):
    """Connection error → friendly error reply (still no silent fail)."""
    slack = _FakeSlackClient(
        replies=[
            {"ts": "1700000000.000300", "user": "UBLAKE", "text": "the message"},
        ],
    )
    fake_async = _FakeAsyncClient(
        _FakeResponse(200, {}),
        raises=ConnectionError("connection refused"),
    )
    fake_httpx = _FakeHttpxModule(fake_async)
    captured: list[_CapturedReply] = []

    result = await handle_pin_reaction(
        event=_make_event(thread_ts=None),
        slack_client=slack,
        reply_in_thread=await _capturing_reply(captured),
        config=PinHandlerConfig(
            atlas_base_url="http://atlas.test:8000",
            atlas_bearer="tok",
            httpx_module=fake_httpx,
        ),
    )

    assert result is None
    assert len(captured) == 1
    assert "pin failed" in captured[0].text
    assert "connection refused" in captured[0].text


@pytest.mark.asyncio
async def test_handler_slack_fetch_failure_falls_back_to_placeholder(allowed_user):
    """If conversations_replies AND conversations_history both fail, the
    handler still ingests a placeholder message (so Blake sees a
    confirmation rather than a Slack outage masquerading as a pin bug)."""
    slack = _FakeSlackClient(
        replies_raises=RuntimeError("slack down"),
        history_raises=RuntimeError("slack down"),
    )
    fake_async = _FakeAsyncClient(_FakeResponse(200, {"job_id": "job-fallback"}))
    fake_httpx = _FakeHttpxModule(fake_async)
    captured: list[_CapturedReply] = []

    result = await handle_pin_reaction(
        event=_make_event(),
        slack_client=slack,
        reply_in_thread=await _capturing_reply(captured),
        config=PinHandlerConfig(
            atlas_base_url="http://atlas.test:8000",
            atlas_bearer="tok",
            httpx_module=fake_httpx,
        ),
    )

    assert result == "job-fallback"
    assert len(fake_async.posted) == 1
    _url, body, _headers = fake_async.posted[0]
    assert "content unavailable" in body["raw_text"]
    assert "urn:atlas:ingest:job-fallback" in captured[0].text


@pytest.mark.asyncio
async def test_handler_no_thread_ts_uses_history(allowed_user):
    """A reaction on a non-threaded message → conversations_history path."""
    slack = _FakeSlackClient(
        # conversations_replies returns nothing (no thread)
        replies=[],
        history=[
            {"ts": "1700000000.000300", "user": "UBLAKE", "text": "lone message"},
        ],
    )
    fake_async = _FakeAsyncClient(_FakeResponse(200, {"job_id": "job-lone"}))
    fake_httpx = _FakeHttpxModule(fake_async)
    captured: list[_CapturedReply] = []

    result = await handle_pin_reaction(
        event=_make_event(thread_ts=None),
        slack_client=slack,
        reply_in_thread=await _capturing_reply(captured),
        config=PinHandlerConfig(
            atlas_base_url="http://atlas.test:8000",
            atlas_bearer="tok",
            httpx_module=fake_httpx,
        ),
    )

    assert result == "job-lone"
    # conversations_history must have been consulted as fallback.
    call_names = [name for name, _ in slack.calls]
    assert "conversations_history" in call_names
    _url, body, _headers = fake_async.posted[0]
    assert "lone message" in body["raw_text"]


@pytest.mark.asyncio
async def test_handler_drops_malformed_event(allowed_user):
    """Missing channel/ts → silent drop with a warning, no ingest call."""
    slack = _FakeSlackClient(replies=[])
    fake_async = _FakeAsyncClient(_FakeResponse(200, {"job_id": "x"}))
    fake_httpx = _FakeHttpxModule(fake_async)
    captured: list[_CapturedReply] = []

    result = await handle_pin_reaction(
        event={"reaction": "pushpin", "user": "UBLAKE", "item": {}},
        slack_client=slack,
        reply_in_thread=await _capturing_reply(captured),
        config=PinHandlerConfig(
            atlas_base_url="http://atlas.test:8000",
            atlas_bearer="tok",
            httpx_module=fake_httpx,
        ),
    )

    assert result is None
    assert fake_async.posted == []
    assert captured == []


@pytest.mark.asyncio
async def test_handler_missing_job_id_reports_error(allowed_user):
    """Atlas returns 200 but no job_id → friendly error reply."""
    slack = _FakeSlackClient(
        replies=[
            {"ts": "1700000000.000300", "user": "UBLAKE", "text": "the message"},
        ],
    )
    fake_async = _FakeAsyncClient(_FakeResponse(200, {}))
    fake_httpx = _FakeHttpxModule(fake_async)
    captured: list[_CapturedReply] = []

    result = await handle_pin_reaction(
        event=_make_event(thread_ts=None),
        slack_client=slack,
        reply_in_thread=await _capturing_reply(captured),
        config=PinHandlerConfig(
            atlas_base_url="http://atlas.test:8000",
            atlas_bearer="tok",
            httpx_module=fake_httpx,
        ),
    )

    assert result is None
    assert len(captured) == 1
    assert "no job_id" in captured[0].text


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_manual_smoke_test_present():
    """Plan 026-C AC: 'Manual smoke instructions in plugin README'.

    We embed the instructions in the module rather than a sibling README
    so they version with the code. This test guards against accidental
    deletion.
    """
    assert "Manual smoke test" in MANUAL_SMOKE_TEST
    assert "ATLAS_BASE_URL" in MANUAL_SMOKE_TEST
    assert "SLACK_ALLOWED_USERS" in MANUAL_SMOKE_TEST
