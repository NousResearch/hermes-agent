"""Session title filters fold case in every script, not only ASCII: SQLite's LOWER()/LIKE fold ASCII
only, so a Cyrillic or accented title was unreachable by any query not typed in its exact case."""

import pytest

from persistence.conversation.sessions import NewSession, SessionOrder, SessionQuery, SessionWindow
from persistence.housekeeping.maintenance import PruneCriteria

TITLES = {"ru": "Пост-запуск релиза", "de": "Straße Ärger", "en": "Deploy notes"}


@pytest.fixture
def store(backend, scope_a):
    store = backend.open(scope_a)
    for sid, title in TITLES.items():
        store.sessions.create(NewSession(sid, "cli"))
        store.sessions.set_title(sid, title)
    return store


@pytest.mark.parametrize("needle, expected", [
    ("пост", ["ru"]), ("ПОСТ", ["ru"]), ("ПоСт-ЗАПУСК", ["ru"]), ("ärger", ["de"]), ("ÄRGER", ["de"]),
    ("DEPLOY", ["en"]), ("нет", []),
])
def test_title_search_and_prune_filter_fold_case_in_any_script(store, needle, expected):
    page = store.sessions.list_sessions(SessionQuery(order=SessionOrder.LAST_ACTIVE, search_text=needle),
                                        SessionWindow(limit=10))
    assert sorted(row.session.session_id for row in page) == expected
    matched = store.maintenance.count_open_prune_matches(criteria=PruneCriteria(title_like=needle))
    assert matched == len(expected)
