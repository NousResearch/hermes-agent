def test_slugify_subject_name():
    from hermes_cli.memory_wiki import slugify_subject

    assert slugify_subject("Memory Wiki") == "memory-wiki"
    assert slugify_subject("web/src/App.tsx") == "web-src-app-tsx"


def test_slugify_subject_empty_and_punctuation_fallback():
    from hermes_cli.memory_wiki import slugify_subject

    assert slugify_subject("") == "subject"
    assert slugify_subject("!!! --- ...") == "subject"


def test_extract_subject_candidates_prefers_titles_and_paths():
    from hermes_cli.memory_wiki import extract_subject_candidates

    session = {"title": "Build memory wiki", "preview": "I want to build a memory wiki"}
    messages = [
        {"role": "user", "content": "Add /memory to web/src/App.tsx"},
        {"role": "tool", "tool_name": "search_files", "content": ""},
    ]
    candidates = extract_subject_candidates(session, messages)
    names = [c.name for c in candidates]
    assert "memory wiki" in names
    assert "web/src/App.tsx" in names
    assert "search_files" in names


def test_extract_subject_candidates_merges_duplicate_slugs_and_accumulates_score():
    from hermes_cli.memory_wiki import extract_subject_candidates

    candidates = extract_subject_candidates(
        {"title": "Memory Wiki", "preview": "Let's discuss memory-wiki"},
        [{"role": "user", "content": "Please work on memory wiki"}],
    )

    matches = [candidate for candidate in candidates if candidate.slug == "memory-wiki"]
    assert len(matches) == 1
    assert matches[0].score > 6


def test_extract_subject_candidates_structured_text_mentions():
    from hermes_cli.memory_wiki import extract_subject_candidates

    candidates = extract_subject_candidates(
        {},
        [
            {
                "role": "user",
                "content": 'Use "Memory Wiki" from /memory and update HermesAgent next.',
            }
        ],
    )
    names = {candidate.name for candidate in candidates}

    assert "memory wiki" in names
    assert "/memory" in names
    assert "HermesAgent" in names


def test_extract_subject_candidates_nested_tool_calls_and_malformed_tool_calls():
    from hermes_cli.memory_wiki import extract_subject_candidates

    candidates = extract_subject_candidates(
        {},
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "search_files"}},
                    {"function": "not-a-dict"},
                    "not-a-dict",
                ],
            },
            {"role": "assistant", "tool_calls": 123},
        ],
    )

    names = {candidate.name for candidate in candidates}
    assert "search_files" in names


def test_extract_subject_candidates_first_user_keyword_phrase_and_package_mentions():
    from hermes_cli.memory_wiki import extract_subject_candidates

    candidates = extract_subject_candidates(
        {},
        [
            {
                "role": "user",
                "content": "Can you help debug the FastAPI pytest plugin failure?",
            },
            {"role": "assistant", "content": "Sure."},
        ],
    )
    by_slug = {candidate.slug: candidate for candidate in candidates}

    assert "fastapi-pytest-plugin-failure" in by_slug
    assert "fastapi" in by_slug
    assert "pytest" in by_slug
    assert "plugin" not in by_slug


def test_extract_subject_candidates_hyphenated_title_alias():
    from hermes_cli.memory_wiki import extract_subject_candidates

    candidates = extract_subject_candidates({"title": "memory-wiki"}, [])

    by_slug = {candidate.slug: candidate for candidate in candidates}

    assert by_slug["memory-wiki"].name == "memory wiki"


def test_build_memory_overview_aggregates_subjects_and_daily_logs(tmp_path):
    from hermes_cli.memory_wiki import build_memory_overview
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s-memory", "cli")
        db.append_message("s-memory", "user", "Build the Memory Wiki in web/src/App.tsx")
        db.append_message("s-memory", "assistant", "Added memory wiki subject cards.")
        db.append_message("s-memory", "tool", "", tool_name="search_files")

        db.create_session("s-fastapi", "discord")
        db.append_message("s-fastapi", "user", "Debug FastAPI pytest plugin failure")
        db.append_message("s-fastapi", "assistant", "Reviewed the pytest failure and planned a fix.")

        overview = build_memory_overview(db, subject_limit=20, day_limit=10)
    finally:
        db.close()

    assert set(overview) == {"subjects", "daily_logs", "recent_sessions"}

    subjects_by_slug = {subject["slug"]: subject for subject in overview["subjects"]}
    assert "memory-wiki" in subjects_by_slug
    assert "web-src-app-tsx" in subjects_by_slug
    assert subjects_by_slug["memory-wiki"]["session_count"] == 1
    assert subjects_by_slug["memory-wiki"]["message_count"] >= 2
    assert subjects_by_slug["memory-wiki"]["sessions"][0]["id"] == "s-memory"
    assert subjects_by_slug["memory-wiki"]["snippets"]

    assert len(overview["daily_logs"]) == 1
    day = overview["daily_logs"][0]
    assert day["session_count"] == 2
    assert day["message_count"] == 5
    assert {session["id"] for session in day["sessions"]} == {"s-memory", "s-fastapi"}
    assert {subject["slug"] for subject in day["subjects"]} >= {"memory-wiki", "fastapi"}
    assert any(item["kind"] == "tool" and "search_files" in item["text"] for item in day["work_items"])


def test_memory_subject_and_daily_log_detail_lookup_and_query_filter(tmp_path):
    from hermes_cli.memory_wiki import (
        build_memory_subjects,
        get_daily_log,
        get_memory_subject,
    )
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s-memory", "cli")
        db.append_message("s-memory", "user", "Build Memory Wiki dashboard")
        db.append_message("s-memory", "assistant", "Implemented Memory Wiki helpers")
        db.create_session("s-react", "cli")
        db.append_message("s-react", "user", "Investigate React dashboard layout")

        day = get_daily_log(db, "1970-01-02")
        assert day is None

        memory_subject = get_memory_subject(db, "memory-wiki")
        missing_subject = get_memory_subject(db, "does-not-exist")
        queried_subjects = build_memory_subjects(db, query="react")
    finally:
        db.close()

    assert memory_subject is not None
    assert memory_subject["slug"] == "memory-wiki"
    assert memory_subject["sessions"][0]["id"] == "s-memory"
    assert missing_subject is None
    queried_slugs = [subject["slug"] for subject in queried_subjects]
    assert "react" in queried_slugs
    assert all("react" in subject["slug"] or "react" in subject["name"].lower() for subject in queried_subjects)


class _MemoryWikiFakeDB:
    def __init__(self, sessions=None, messages=None):
        self.sessions = list(sessions or [])
        self.messages = dict(messages or {})
        self.list_limits = []
        self.message_session_ids = []

    def list_sessions_rich(self, *, limit, include_children, order_by_last_active):
        self.list_limits.append(limit)
        assert include_children is False
        assert order_by_last_active is True
        return self.sessions[:limit]

    def get_messages(self, session_id):
        self.message_session_ids.append(session_id)
        return list(self.messages.get(session_id, []))


class _FailingLoadDB:
    def list_sessions_rich(self, **kwargs):
        raise AssertionError("non-positive limits should not load sessions")


def test_memory_wiki_aggregators_return_empty_for_empty_db():
    from hermes_cli.memory_wiki import build_daily_logs, build_memory_subjects

    db = _MemoryWikiFakeDB()

    assert build_memory_subjects(db) == []
    assert build_daily_logs(db) == []
    assert db.list_limits == [500, 1500]
    assert db.message_session_ids == []


def test_memory_wiki_aggregators_handle_sessions_with_no_messages():
    from hermes_cli.memory_wiki import build_daily_logs, build_memory_subjects

    session = {
        "id": "s-empty",
        "title": "Memory Wiki",
        "source": "cli",
        "preview": "",
        "started_at": 10.0,
        "last_active": 10.0,
    }
    db = _MemoryWikiFakeDB([session], {"s-empty": []})

    subjects = build_memory_subjects(db, limit=10)
    logs = build_daily_logs(db, limit_days=10)

    assert [subject["slug"] for subject in subjects] == ["memory-wiki"]
    assert subjects[0]["message_count"] == 0
    assert subjects[0]["snippets"] == []
    assert len(logs) == 1
    assert logs[0]["session_count"] == 1
    assert logs[0]["message_count"] == 0
    assert logs[0]["sessions"][0]["id"] == "s-empty"


def test_memory_wiki_aggregators_do_not_load_for_zero_or_negative_limits():
    from hermes_cli.memory_wiki import build_daily_logs, build_memory_subjects

    db = _FailingLoadDB()

    assert build_memory_subjects(db, limit=0) == []
    assert build_memory_subjects(db, limit=-5) == []
    assert build_daily_logs(db, limit_days=0) == []
    assert build_daily_logs(db, limit_days=-5) == []


def test_memory_wiki_aggregators_bound_loaded_history():
    from hermes_cli.memory_wiki import build_daily_logs, build_memory_subjects

    sessions = [
        {
            "id": f"s-{index}",
            "title": f"Project Alpha{index} Topic",
            "source": "cli",
            "preview": "",
            "started_at": float(index * 86400),
            "last_active": float(index * 86400),
        }
        for index in range(120)
    ]
    db = _MemoryWikiFakeDB(sessions, {session["id"]: [] for session in sessions})

    subjects = build_memory_subjects(db, limit=2)
    logs = build_daily_logs(db, limit_days=2)

    assert db.list_limits == [100, 100]
    assert len(subjects) == 2
    assert len(logs) == 2
    assert len(db.message_session_ids) == 200
    assert set(db.message_session_ids) == {f"s-{index}" for index in range(100)}
