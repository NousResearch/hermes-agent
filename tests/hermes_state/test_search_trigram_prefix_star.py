"""A trailing prefix ``*`` must not disable the trigram substring fallback.

The ``session_search`` tool schema teaches the model the prefix syntax
(``deploy*``), so agents routinely append ``*`` to a word fragment. When the
fragment starts mid-token (``apability*`` for "capability") the unicode61 index
correctly returns nothing and the zero-result Latin retry falls through to the
trigram index -- but ``_quote_fts_tokens`` quoted the token together with its
star, and inside a phrase ``*`` is a literal. The retry then looked for the
text ``apability*`` and was guaranteed to miss, while the bare fragment found
every row. Substring indexes need no prefix operator, so the retry drops it.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    if not d._trigram_available:
        d.close()
        pytest.skip("SQLite build has no trigram tokenizer")
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content="Please run the capability check before the deployment")
    d.append_message("s1", role="assistant", content="The capability check passed; deployment is green")
    d.append_message("s1", role="user", content="unrelated note about lunch")
    yield d
    try:
        d.close()
    except Exception:
        pass


def test_bare_mid_token_fragment_reaches_trigram(db):
    # Control: the trigram fallback itself works for a mid-token fragment.
    assert len(db.search_messages("apability", limit=10)) == 2


def test_mid_token_fragment_with_trailing_star_reaches_trigram(db):
    assert len(db.search_messages("apability*", limit=10)) == 2


def test_every_starred_fragment_is_stripped(db):
    assert len(db.search_messages("apability* eployment*", limit=10)) == 2


def test_starred_fragment_keeps_boolean_operators(db):
    assert len(db.search_messages("apability* OR unch*", limit=10)) == 3


def test_starred_hyphenated_fragment_reaches_trigram(db):
    # The sanitizer quotes hyphenated terms first (``"y-che"*``); the star must go
    # before the outer quotes are stripped, or a literal quote reaches trigram.
    db.append_message("s1", role="user", content="the sky-check job is scheduled")
    assert len(db.search_messages("y-che", limit=10)) == 1
    assert len(db.search_messages("y-che*", limit=10)) == 1


def test_true_prefix_still_served_by_the_base_index(db):
    # A real prefix never reaches the fallback: the unicode61 index answers it.
    assert len(db.search_messages("capab*", limit=10)) == 2
