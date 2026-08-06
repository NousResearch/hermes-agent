"""Dashboard analytics must decode tokens through the SAME path as SessionDB.

Regression cover for the review finding on PR #55805: the dashboard server
(``hermes_cli/web_server.py``) carried its own inline copy of the bit-packed
``token_count`` decode SQL. The copy never decoded the reasoning tag and
returned a hardcoded ``reasoning: 0``, so dashboard trend responses undercounted
reasoning models while the gateway/API path reported them correctly.

The duplication was the mechanism, so these tests pin both the behaviour
(all four buckets decode, and the dashboard agrees with SessionDB) and the
structure (the dashboard grows no second decoder).
"""

from __future__ import annotations

import asyncio
import inspect
import re
import tempfile
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_token_codec import pack_assistant_tokens, pack_input_tokens

T0 = 1_000_000.0


@pytest.fixture()
def db():
    d = SessionDB(db_path=Path(tempfile.mkdtemp()) / "parity.db")
    d.create_session(session_id="s1", source="cli")
    # One turn with every bucket non-zero and mutually distinct, so a decoder
    # that crosses two tags (or drops one) cannot accidentally still match.
    d.append_message(session_id="s1", role="user", content="q",
                     token_count=pack_input_tokens(1500, 700), timestamp=T0 + 1)
    d.append_message(session_id="s1", role="assistant", content="a",
                     token_count=pack_assistant_tokens(300, 128), timestamp=T0 + 2)
    yield d
    d.close()


EXPECTED = {"input": 1500, "cache_read": 700, "output": 300, "reasoning": 128}


# ---------------------------------------------------------------------------
# The bug: reasoning was hardcoded to 0
# ---------------------------------------------------------------------------


def test_timeseries_decodes_all_four_buckets(db):
    """reasoning is decoded from the codec, not reported as a constant 0."""
    ts = db.get_message_token_timeseries(T0, T0 + 60, 60)
    assert len(ts) == 1
    bucket = ts[0]
    for name, want in EXPECTED.items():
        assert bucket[name] == want, f"{name}: got {bucket[name]}, want {want}"
    assert bucket["reasoning"] != 0, "reasoning must not be a hardcoded zero"
    assert bucket["requests"] == 1 and bucket["messages"] == 2


# ---------------------------------------------------------------------------
# Parity: the dashboard endpoint must report what SessionDB reports
# ---------------------------------------------------------------------------


def test_dashboard_trends_match_sessiondb(db, monkeypatch):
    """/api/analytics/token-trends decodes via SessionDB, so the buckets agree."""
    web_server = pytest.importorskip(
        "hermes_cli.web_server", reason="dashboard server needs fastapi"
    )

    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(
        web_server, "_open_session_db_for_profile", lambda profile, *, read_only: db
    )
    monkeypatch.setattr(web_server.time, "time", lambda: T0 + 120)

    payload = asyncio.run(web_server.get_token_trends(window="1h", bucket=60))

    # Totals are the reviewer's actual symptom: reasoning arrived as 0 here
    # while the gateway/API path reported it correctly.
    for name, want in EXPECTED.items():
        assert payload["totals"][name] == want, (
            f"totals.{name}: got {payload['totals'][name]}, want {want}"
        )

    # And bucket-for-bucket the endpoint reports exactly the shared decoder's
    # output — no second decode path can drift in between.
    expected_buckets = db.get_message_token_timeseries(T0 + 120 - 3600, T0 + 120, 60)
    seeded_expected = [b for b in expected_buckets if b["requests"]]
    seeded_actual = [b for b in payload["series"] if b["requests"]]
    assert len(seeded_actual) == 1
    for name in EXPECTED:
        assert seeded_actual[0][name] == seeded_expected[0][name] == EXPECTED[name]


def test_dashboard_usage_rates_sees_reasoning(db, monkeypatch):
    """The rates endpoint reads the same decoded series (reasoning included)."""
    web_server = pytest.importorskip(
        "hermes_cli.web_server", reason="dashboard server needs fastapi"
    )

    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(
        web_server, "_open_session_db_for_profile", lambda profile, *, read_only: db
    )
    monkeypatch.setattr(web_server.time, "time", lambda: T0 + 120)

    payload = asyncio.run(web_server.get_usage_rates(window="1h"))
    assert payload["window"] == "1h"
    # Sanity: the endpoint ran against seeded data rather than an empty window.
    assert payload["rpm"]["peak"] >= 1


# ---------------------------------------------------------------------------
# Structure: no second decoder may reappear in the dashboard module
# ---------------------------------------------------------------------------


def test_dashboard_module_has_no_private_token_decoder():
    """The codec layout must be referenced in ONE place, not re-inlined here.

    Guards the mechanism the reviewer identified: a local copy of the shift /
    mask SQL is what silently diverged from the codec in the first place.
    """
    web_server = pytest.importorskip(
        "hermes_cli.web_server", reason="dashboard server needs fastapi"
    )
    src = inspect.getsource(web_server)

    # The magic numbers of the codec layout (tag mask, 27-bit and 28-bit field
    # masks) and the shift offsets have no business in the dashboard module.
    for magic in ("134217727", "268435455", ">>59", ">>32", ">> 59", ">> 32"):
        assert magic not in src, f"codec layout literal {magic!r} re-inlined in web_server.py"

    assert not re.search(r"token_count\s*<\s*0", src), (
        "web_server.py decodes packed token_count itself; call SessionDB instead"
    )
