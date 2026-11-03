"""Boolean operators in session search: FTS5 honours only uppercase ``AND``/``OR``/``NOT``.

Lowercase ``and``/``or`` between two terms is promoted once, in the sanitizer; every other route
(the LIKE fallback, the CJK LIKE arm) then follows FTS5's rule, so lowercase ``not`` stays a word.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def test_lowercase_interior_operators_are_uppercased():
    """FTS5 only honours uppercase operators; ``python or rust`` must not
    search for the literal word "or"."""
    s = SessionDB._sanitize_fts5_query
    assert s('python or rust') == 'python OR rust'
    assert s('python and rust') == 'python AND rust'
    # Lowercase "not" is usually prose; it must not become an exclusion.
    assert s('tests do not pass') == 'tests do not pass'
    assert s('python NOT java') == 'python NOT java'
    # Quoted phrases keep their words verbatim.
    assert s('"cats or dogs" or fish') == '"cats or dogs" OR fish'
    # Stacked operators are not "between terms"; left untouched.
    assert s('python or or rust') == 'python or or rust'


def test_lowercase_or_matches_either_term(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", role="user", content="I write python daily")
    db.append_message("s1", role="assistant", content="rust is memory safe")
    db.append_message("s1", role="user", content="java is verbose")

    snippets = [r["snippet"] for r in db.search_messages("python or rust")]
    assert len(snippets) == 2
    assert any("python" in s for s in snippets)
    assert any("rust" in s for s in snippets)


def test_lowercase_not_is_a_word_on_the_like_route(db):
    """The LIKE route (role_filter with tool, stale FTS, fail-open) must agree with FTS5:
    lowercase ``not`` is searched, not turned into an exclusion."""
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", role="user", content="the tests do not pass")
    db.append_message("s1", role="tool", content="tool: the tests do not pass on CI")

    assert len(db.search_messages("tests do not pass")) == 2  # FTS5 route
    assert len(db.search_messages("tests do not pass", role_filter=["user", "assistant", "tool"])) == 2
    # Uppercase NOT still excludes on the LIKE route.
    excluded = db.search_messages("tests NOT CI", role_filter=["user", "tool"])
    assert [r["role"] for r in excluded] == ["user"]


def test_cjk_like_route_honours_explicit_operators(db):
    """Cron sessions are excluded from the substring indexes, so a CJK query filtered to
    them scans LIKE. Bare terms keep that arm's any-term recall, but a spelled operator
    must compile exactly: ``and`` needs both terms and ``NOT`` excludes."""
    db.create_session(session_id="c1", source="cron")
    db.append_message("c1", role="user", content="部署 监控 正常")
    db.append_message("c1", role="user", content="只有 测试 数据")
    db.append_message("c1", role="user", content="部署 测试 失败")

    def hits(q):
        return sorted(r["snippet"] for r in db.search_messages(q, source_filter=["cron"]))

    assert hits("部署 监控") == ["部署 测试 失败", "部署 监控 正常"]
    assert hits("部署 and 监控") == ["部署 监控 正常"]
    assert hits("部署 NOT 测试") == ["部署 监控 正常"]
    assert hits("监控 OR 数据") == ["只有 测试 数据", "部署 监控 正常"]
