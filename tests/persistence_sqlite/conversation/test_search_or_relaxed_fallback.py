"""OR-relaxed zero-result retry for paraphrased session search recall.

FTS5's implicit AND between query terms means a multi-word query worded even slightly
differently from the stored sentence returns nothing: a fact saved as
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

from persistence.sqlite.conversation._store import search
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.search import search_messages
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture
def holder(backend, scope_a):
    h = backend.open(scope_a).holder()
    create_session(h, session_id="s1", source="cli", model="m")
    append_message(
        h,
        "s1",
        role="user",
        content=(
            "Sarah prefers the standup meeting scheduled early on "
            "Thursday mornings"
        ),
    )
    append_message(
        h, "s1", role="assistant", content="Noted, standup moved to Thursday."
    )
    append_message(
        h, "s1", role="user", content="graphiti daemon looks healthy today"
    )
    return h


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
    assert search._or_relaxed_query(query) == expected


def test_paraphrased_query_recovers_via_or_retry(holder):
    """Exact hits are untouched; a paraphrase whose extra word ("like") no stored row contains
    is recovered by the OR retry, every partially-matching row comes back, and the caller's
    role_filter still applies to the retried query."""
    exact = search_messages(holder, "standup Thursday")
    assert exact and "standup" in exact[0]["snippet"].lower()

    rows = search_messages(holder, "when does Sarah like her standup scheduled")
    assert rows and "standup" in " ".join(r["snippet"].lower() for r in rows)

    joined = " ".join(r["snippet"].lower() for r in search_messages(holder, "sarah standup thursday daemon"))
    assert "standup" in joined and "daemon" in joined

    assistant_only = search_messages(holder, "when does Sarah like her standup scheduled", role_filter=["assistant"])
    assert assistant_only and all(r["role"] == "assistant" for r in assistant_only)


def test_relaxation_does_not_fire_for_exact_semantics_or_absent_terms(holder, monkeypatch):
    """Explicit NOT keeps its exclusion (relaxing would resurrect the Thursday rows), a genuine
    miss stays empty, and a CJK-routed miss never reaches the OR rewrite (the CJK index has its
    own substring semantics)."""
    not_rows = search_messages(holder, "standup NOT Thursday")
    assert all("thursday" not in r["snippet"].lower() for r in not_rows)
    assert search_messages(holder, "zebra xylophone quantum") == []

    calls = []
    monkeypatch.setattr(search, "_or_relaxed_query", lambda q: calls.append(q))
    assert search_messages(holder, "站会 周五") == []
    assert calls == []
