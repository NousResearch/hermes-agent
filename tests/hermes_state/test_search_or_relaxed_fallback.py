"""OR-relaxed zero-result retry for paraphrased session search recall.

FTS5's implicit AND between query terms means a multi-word query worded even slightly
differently from the stored sentence returns nothing — a fact saved as
"Sarah prefers the standup meeting scheduled early on Thursday mornings" is
invisible to "when does Sarah like her standup scheduled" purely because the
stored text has no "like". When the exact-match search (and the substring
fallbacks) return zero rows, ``search_messages`` retries the same FTS index
with the terms OR-joined; under the default rank sort rows covering more of
the terms surface first.

The retry is strictly additive: it only fires on a zero-result miss, never
reorders existing hits, and respects explicit boolean operators.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message(
        "s1",
        role="user",
        content=(
            "Sarah prefers the standup meeting scheduled early on "
            "Thursday mornings"
        ),
    )
    d.append_message(
        "s1", role="assistant", content="Noted, standup moved to Thursday."
    )
    d.append_message(
        "s1", role="user", content="graphiti daemon looks healthy today"
    )
    yield d
    try:
        d.close()
    except Exception:
        pass


@pytest.mark.parametrize("query, expected", [
    ("sarah standup scheduled", "sarah OR standup OR scheduled"),
    ("alpha AND beta", "alpha OR beta"),
    ('"docker networking" tls', '"docker networking" OR tls'),
    ("standup", None),
    ('"docker networking"', None),
    ("alpha OR beta", None),
    ("python NOT java", None),
])
def test_or_relaxed_query_rewrite(query, expected):
    """Implicit-AND terms and explicit AND become an any-term OR query; a quoted phrase is one
    unit; a single unit or explicit OR/NOT (exact semantics already expressed) does not relax."""
    assert SessionDB._or_relaxed_query(query) == expected


def test_paraphrased_query_recovers_via_or_retry(db):
    """Exact hits are untouched; a paraphrase whose extra word ("like") no stored row contains
    is recovered by the OR retry, every partially-matching row comes back, and the caller's
    role_filter still applies to the retried query."""
    exact = db.search_messages("standup Thursday")
    assert exact and "standup" in exact[0]["snippet"].lower()

    rows = db.search_messages("when does Sarah like her standup scheduled")
    assert rows and "standup" in " ".join(r["snippet"].lower() for r in rows)

    joined = " ".join(r["snippet"].lower() for r in db.search_messages("sarah standup thursday daemon"))
    assert "standup" in joined and "daemon" in joined

    assistant_only = db.search_messages("when does Sarah like her standup scheduled", role_filter=["assistant"])
    assert assistant_only and all(r["role"] == "assistant" for r in assistant_only)


def test_relaxation_does_not_fire_for_exact_semantics_or_absent_terms(db, monkeypatch):
    """Explicit NOT keeps its exclusion (relaxing would resurrect the Thursday rows), a genuine
    miss stays empty, and a CJK-routed miss never reaches the OR rewrite (the CJK index has its
    own substring semantics)."""
    not_rows = db.search_messages("standup NOT Thursday")
    assert all("thursday" not in r["snippet"].lower() for r in not_rows)
    assert db.search_messages("zebra xylophone quantum") == []

    calls = []
    monkeypatch.setattr(SessionDB, "_or_relaxed_query", staticmethod(lambda q: calls.append(q)))
    assert db.search_messages("站会 周五") == []
    assert calls == []
