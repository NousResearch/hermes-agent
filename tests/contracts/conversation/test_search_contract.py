"""Contract suite for the ``SessionSearch`` capability.

Rows are planted through the reference fixture seam (``seed_session`` / ``seed_message``) or the public sessions and messages verbs, and observed through the four ``persistence.conversation.search.SessionSearch`` verbs. Every expectation is the store's answer for the same corpus and query. Relevance order is the backend's, so a query without ``NEWEST`` / ``OLDEST`` is compared as a set; ties of equal timestamps are compared as sets except on the ``tool`` role route, which orders them by id.
"""

from __future__ import annotations

import pytest

from persistence import RowVisibility
from persistence.conversation.search import SearchHit, SearchIndexRebuildStatus, SearchSort
from tests.contracts._scopes import profile_scope

T0 = 1_790_000_000.0
NEWEST, OLDEST = SearchSort.NEWEST, SearchSort.OLDEST

LEXICON = [
    ("user", "the quick brown fox"),
    ("assistant", "deploy chat-send now via my-app.config.ts"),
    ("user", "unrelated haystack needle"),
    ("tool", "toolbody quick result", "web_search"),
    ("user", "brown the fox quick"),
    ("user", "concatenate the corpus"),
    ("assistant", "TODO fix the quick parser"),
    ("user", "quick_brown underscore"),
    ("user", "Café crème brûlée"),
    ("tool", "independent result body", "hammer"),
    ("tool", "deployment logs attached", "hammer"),
    ("user", "修改youer服务端"),
    ("user", "Sarah prefers the standup meeting scheduled on Thursday"),
    ("user", "P2.2 release plan"),
    ("user", "QUICK uppercase Fox"),
]


def seed_rows(backend, scope, session_id, rows, *, source="cli", start=1.0, started_at=T0, model="m"):
    """One session holding ``rows`` ((role, content[, tool_name]) at T0 + start + i); returns their ids in order."""
    store = backend.open(scope)
    store.search.seed_session(session_id, source=source, model=model, started_at=started_at)
    return [store.search.seed_message(session_id, row[0], row[1], timestamp=T0 + start + i, tool_name=row[2] if len(row) > 2 else None)
            for i, row in enumerate(rows)]


def ids(backend, scope, query, **kwargs):
    store = backend.open(scope)
    return [hit.message_id for hit in store.search.search_messages(query, **kwargs)]


def named(id_list, wanted):
    return {id_list[i] for i in wanted}


# ── search_messages: the word route ───────────────────────────────────────────


def test_words_match_whole_words_of_content_tool_name_and_tool_calls(backend, scope_a):
    store_a = backend.open(scope_a)
    lex = seed_rows(backend, scope_a, "lex", LEXICON)
    calls = store_a.search.seed_message("lex", "assistant", "hello", timestamp=T0 + 50, tool_calls=(
        {"id": "c1", "type": "function", "function": {"name": "callneedle", "arguments": "{}"}},))
    # CONTRACT: a word matches whole words in any role, case-insensitively; ``_`` splits words.
    assert set(ids(backend, scope_a, "quick")) == named(lex, [0, 3, 4, 6, 7, 14])
    # CONTRACT: the tool name and the tool-call JSON are searchable.
    assert ids(backend, scope_a, "web_search") == [lex[3]]
    assert ids(backend, scope_a, "callneedle") == [calls]
    # CONTRACT: Latin diacritics fold both ways.
    for query in ("cafe", "crème", "creme"):
        assert ids(backend, scope_a, query) == [lex[8]]


def test_query_syntax_phrases_prefixes_and_operators(backend, scope_a):
    lex = seed_rows(backend, scope_a, "lex", LEXICON)
    cases = {
        '"quick brown fox"': [0], '"quick brown"': [0, 7], "dep*": [1, 10], "qui*": [0, 3, 4, 6, 7, 14], "hay*": [2],
        "quick AND fox": [0, 4, 14], "needle OR deploy": [1, 2], "quick NOT fox": [3, 6, 7],
        # CONTRACT: adjacent words bind tighter than NOT, NOT tighter than AND, AND tighter than OR.
        "quick NOT brown fox": [3, 6, 7, 14], "quick NOT parser AND fox": [0, 4, 14], "quick fox OR needle": [0, 2, 4, 14],
        "fox OR needle NOT haystack": [0, 4, 14],
        # CONTRACT: punctuated words are phrases, sanitized characters are spaces.
        "chat-send": [1], "my-app.config.ts": [1], "P2.2": [13], "quick_brown": [0, 7], "TODO: fix": [6], "fox+brown": [0, 4],
        '"quick bro"*': [0, 7], "quick*fox*": [0, 4, 14],
        # CONTRACT: one leading and one trailing operator word of any case is dropped; lowercase operators are words.
        "NOT quick": [0, 3, 4, 6, 7, 14], "quick OR": [0, 3, 4, 6, 7, 14], "quick%": [0, 3, 4, 6, 7, 14],
    }
    for query, expected in cases.items():
        assert set(ids(backend, scope_a, query)) == named(lex, expected), query


@pytest.mark.parametrize("query", ["", "   ", ":::", "OR", "AND NOT", '""', "(^~)", "((((", "a OR NOT b", "-quick", "quick.", "fox -brown",
                                   "\x00quick", "\x000\x00", "quick AND AND fox", "and"])
def test_text_with_nothing_searchable_or_rejected_syntax_matches_nothing(backend, scope_a, query):
    store_a = backend.open(scope_a)
    seed_rows(backend, scope_a, "lex", LEXICON + [("user", "or not and")])
    # CONTRACT: blank text, text that sanitizes to nothing and text the syntax still rejects return () without a retry.
    assert store_a.search.search_messages(query) == ()


def test_a_miss_retries_substrings_then_any_word(backend, scope_a):
    store_a = backend.open(scope_a)
    lex = seed_rows(backend, scope_a, "lex", LEXICON)
    skipped = seed_rows(backend, scope_a, "cron", [("user", "concatenated cron digest")], source="cron", start=40)
    tool = store_a.search.seed_message("lex", "tool", "toolconcatenation", timestamp=T0 + 60)
    # CONTRACT: a word that matches no whole word is retried as a substring of non-tool rows outside cron / subagent sessions.
    assert ids(backend, scope_a, "cat") == [lex[5]]
    assert skipped[0] not in ids(backend, scope_a, "cat") and tool not in ids(backend, scope_a, "cat")
    assert ids(backend, scope_a, "youer") == [lex[11]]
    # CONTRACT: the substring retry needs every word to have three or more characters.
    assert ids(backend, scope_a, "ca") == []
    # CONTRACT: a phrase whose words all appear, out of order, is found by the substring retry.
    store_a.search.seed_session("decoy", source="cli", started_at=T0)
    decoy = store_a.search.seed_message("decoy", "user", "brown the fox quick zebra", timestamp=T0 + 70)
    assert decoy in ids(backend, scope_a, '"quick zebra"')
    # CONTRACT: an implicit-AND miss is then retried with its words ORed; explicit OR / NOT is never relaxed.
    assert ids(backend, scope_a, "sarah standup weekly") == [lex[12]]
    assert ids(backend, scope_a, "sarah AND weekly") == [lex[12]]
    assert ids(backend, scope_a, "zebrafish OR weekly") == []
    assert ids(backend, scope_a, "sarah zebrafish NOT weekly") == []


# ── search_messages: the tool role and CJK routes ─────────────────────────────

TOOLS = [
    ("user", "the quick brown fox"),
    ("tool", "toolbody quick result", "web_search"),
    ("tool", "independent result body", "hammer"),
    ("tool", "deployment logs attached", "hammer"),
    ("user", "QUICK uppercase"),
    ("tool", "tool body " + "y" * 9000 + " deepneedle", "shell"),
    ("user", {"text": "hello"}),
    ("user", "hello plain"),
]


def test_tool_role_filter_reads_stored_text_by_substring(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = seed_rows(backend, scope_a, "tools", TOOLS)
    tool, both = ["tool"], ["tool", "user"]
    # CONTRACT: words match as ASCII-case-insensitive substrings of content, tool name or tool-call JSON, newest first.
    assert ids(backend, scope_a, "dep", role_filter=tool) == [rows[3], rows[2]]
    assert ids(backend, scope_a, "dep*", role_filter=tool) == [rows[3], rows[2]]
    assert ids(backend, scope_a, "hammer", role_filter=tool) == [rows[3], rows[2]]
    assert ids(backend, scope_a, "Quick", role_filter=both) == [rows[4], rows[1], rows[0]]
    assert ids(backend, scope_a, "Quick", role_filter=both, sort=OLDEST) == [rows[0], rows[1], rows[4]]
    # CONTRACT: NOT excludes the next word and OR separates groups.
    assert ids(backend, scope_a, "quick NOT toolbody", role_filter=both) == [rows[4], rows[0]]
    assert ids(backend, scope_a, "fox OR body", role_filter=both) == [rows[5], rows[2], rows[1], rows[0]]
    # CONTRACT: the whole tool body is read here, while the word route covers only its prefix.
    assert ids(backend, scope_a, "deepneedle") == []
    (hit,) = store_a.search.search_messages("deepneedle", role_filter=tool)
    content = TOOLS[5][1]
    start = content.index("deepneedle") - 40
    # CONTRACT: the snippet is 120 characters of content from 40 before the first word, unmarked.
    assert hit.message_id == rows[5] and hit.snippet == content[start : start + 120]
    assert store_a.search.search_messages("dep", role_filter=tool)[0].snippet == "deployment logs attached"
    # CONTRACT: structured content never matches here; the word route finds it.
    assert ids(backend, scope_a, "hello", role_filter=both) == [rows[7]]
    assert set(ids(backend, scope_a, "hello")) == {rows[6], rows[7]}
    tied = [store_a.search.seed_message("tools", "tool", f"tiedtool {i}", timestamp=T0 + 99) for i in range(3)]
    # CONTRACT: equal timestamps order by id in the sort direction.
    assert ids(backend, scope_a, "tiedtool", role_filter=tool) == tied[::-1]
    assert ids(backend, scope_a, "tiedtool", role_filter=tool, sort=OLDEST) == tied


CJK = [
    ("user", "广西漓江的风景"), ("user", "hello english only"), ("user", "桂林山水甲天下"), ("user", "漓字和江字和水字分开出现"),
    ("user", "今天漓江水很清"), ("tool", "工具输出也包含漓江水字样", "terminal"), ("user", "桂林ABC"), ("user", "桂林abc"),
    ("user", "前𠀀𠀁后"), ("user", "上海的夏天"),
]


def test_short_cjk_text_matches_any_word_as_a_substring_newest_first(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = seed_rows(backend, scope_a, "cjk", CJK)
    cases = {"西": [0], "桂林": [7, 6, 2], "广西 桂林": [7, 6, 2, 0], "广西 OR 桂林": [7, 6, 2, 0], "桂林*": [7, 6, 2],
             "漓江水 NOT 今天": [5, 4], "桂林abc": [7, 6], "𠀀": [8], "夏天 OR 风景": [9, 0]}
    for query, expected in cases.items():
        # CONTRACT: any one word, operators ignored, tool rows included; newest first whatever the sort.
        for sort in (SearchSort.RELEVANCE, OLDEST):
            assert ids(backend, scope_a, query, sort=sort) == [rows[i] for i in expected], query
    (hit,) = store_a.search.search_messages("西")
    # CONTRACT: the snippet is unmarked content.
    assert hit.snippet == "广西漓江的风景"


def test_long_cjk_text_matches_substrings_of_indexed_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = seed_rows(backend, scope_a, "cjk", CJK)
    # CONTRACT: three or more CJK characters per word match contiguous substrings of non-tool rows, marked.
    (hit,) = store_a.search.search_messages("漓江水")
    assert hit.message_id == rows[4] and hit.snippet == "今天>>>漓江水<<<很清"
    assert ids(backend, scope_a, '"漓江水"', sort=OLDEST) == [rows[4]]
    # CONTRACT: a tool role filter reaches the tool row by substring.
    assert ids(backend, scope_a, "漓江水", role_filter=["tool"]) == [rows[5]]
    assert ids(backend, scope_a, "漓", role_filter=["tool"]) == [rows[5]]
    seeded = {source: seed_rows(backend, scope_a, f"s-{source}", [("user", "sourced 桂林山水 row")], source=source, start=20)[0]
              for source in ("cron", "subagent", "telegram")}
    store_a.search.seed_session("s-child", source="cli", started_at=T0, delegate_child=True)
    child = store_a.search.seed_message("s-child", "user", "sourced 桂林山水 row", timestamp=T0 + 30)
    # CONTRACT: cron, subagent and delegate-child rows are outside the substring index; short text still reaches them.
    assert set(ids(backend, scope_a, "桂林山水")) == {rows[2], seeded["telegram"]}
    assert ids(backend, scope_a, "桂林山水", exclude_sources=["telegram"]) == [rows[2]]
    assert ids(backend, scope_a, "桂林山水", source_filter=["cron"]) == [seeded["cron"]]
    assert set(ids(backend, scope_a, "桂林")) == {rows[2], rows[6], rows[7], child, *seeded.values()}


# ── search_messages: filters, order and paging ────────────────────────────────


def test_visibility_hides_rewound_and_hidden_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    seed = backend.open(scope_a).search.seed_message
    store_a.search.seed_session("vis", source="cli", started_at=T0)
    live = seed("vis", "user", "visterm alpha live", timestamp=T0 + 1)
    rewound = seed("vis", "assistant", "visterm trailing rewound", timestamp=T0 + 2, visibility=RowVisibility.REWOUND)
    seed("vis", "user", "visterm hidden scaffold", timestamp=T0 + 3, display_kind="hidden")
    notice = seed("vis", "user", "visterm notice shown", timestamp=T0 + 4, display_kind="async_delegation_complete")
    archived = seed("vis", "user", "visterm echo original", timestamp=T0 + 5, visibility=RowVisibility.COMPACTION_ARCHIVED)
    # CONTRACT: active and compaction-archived rows match by default; rewound rows only with include_inactive.
    assert set(ids(backend, scope_a, "visterm")) == {live, notice, archived}
    assert set(ids(backend, scope_a, "visterm", include_inactive=True)) == {live, rewound, notice, archived}
    assert ids(backend, scope_a, "trailing") == [] and ids(backend, scope_a, "trailing", include_inactive=True) == [rewound]
    # CONTRACT: display_kind "hidden" rows never match.
    assert ids(backend, scope_a, "hidden", include_inactive=True) == []
    assert ids(backend, scope_a, "hidden", include_inactive=True, role_filter=["tool", "user"]) == []


def test_source_role_and_session_start_filters(backend, scope_a):
    found = {source: seed_rows(backend, scope_a, f"s-{source}", [("user", "sourced needle row")], source=source, start=i,
                               started_at=T0 + 100 * (i + 2))[0] for i, source in enumerate(("cron", "subagent", "telegram"))}
    early = seed_rows(backend, scope_a, "s-early", [("user", "sourced needle row")], start=9)[0]
    every = {*found.values(), early}
    assert set(ids(backend, scope_a, "sourced")) == every
    # CONTRACT: source_filter keeps those sources (an empty one keeps none); exclude_sources drops them (an empty one drops none).
    assert ids(backend, scope_a, "sourced", source_filter=["cron"]) == [found["cron"]]
    assert ids(backend, scope_a, "sourced", source_filter=[]) == []
    assert set(ids(backend, scope_a, "sourced", exclude_sources=["cron"])) == every - {found["cron"]}
    assert set(ids(backend, scope_a, "sourced", exclude_sources=[])) == every
    # CONTRACT: an empty role_filter is no filter.
    assert set(ids(backend, scope_a, "sourced", role_filter=[])) == every
    assert ids(backend, scope_a, "sourced", role_filter=["assistant"]) == []
    # CONTRACT: after_ts is inclusive and before_ts exclusive on the owning session's start.
    start = int(T0)
    assert set(ids(backend, scope_a, "sourced", after_ts=start + 200)) == set(found.values())
    assert set(ids(backend, scope_a, "sourced", after_ts=start + 201)) == {found["subagent"], found["telegram"]}
    assert set(ids(backend, scope_a, "sourced", before_ts=start + 300)) == {early, found["cron"]}
    assert set(ids(backend, scope_a, "sourced", after_ts=start + 200, before_ts=start + 400)) == {found["cron"], found["subagent"]}
    assert ids(backend, scope_a, "sourced", after_ts=start + 300, role_filter=["tool", "user"]) == [found["telegram"], found["subagent"]]


def test_sort_limit_and_offset(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = seed_rows(backend, scope_a, "order", [("user", f"pagetoken item {i}") for i in range(7)])
    # CONTRACT: NEWEST and OLDEST order by timestamp.
    assert ids(backend, scope_a, "pagetoken", sort=NEWEST) == rows[::-1]
    assert ids(backend, scope_a, "pagetoken", sort=OLDEST) == rows
    # CONTRACT: limit and offset slice the ordered hits; limit 0 is empty, a negative limit unbounded, a negative offset zero.
    assert ids(backend, scope_a, "pagetoken", sort=OLDEST, limit=3, offset=2) == rows[2:5]
    assert ids(backend, scope_a, "pagetoken", sort=OLDEST, limit=0) == []
    assert ids(backend, scope_a, "pagetoken", sort=OLDEST, limit=-1) == rows
    assert ids(backend, scope_a, "pagetoken", sort=OLDEST, limit=2, offset=-3) == rows[:2]
    assert ids(backend, scope_a, "pagetoken", sort=OLDEST, offset=7) == []
    assert len(ids(backend, scope_a, "pagetoken")) == 7 and len(ids(backend, scope_a, "pagetoken", limit=5)) == 5
    tied = [store_a.search.seed_message("order", "user", f"tiedterm {i}", timestamp=T0 + 50) for i in range(4)]
    # CONTRACT: equal timestamps keep every hit exactly once across offsets.
    pages = [ids(backend, scope_a, "tiedterm", sort=OLDEST, limit=3, offset=offset) for offset in (0, 3)]
    assert sorted(pages[0] + pages[1]) == sorted(tied) and [len(page) for page in pages] == [3, 1]


def test_an_offset_past_a_smaller_result_reaches_the_next_retry(backend, scope_a):
    rows = seed_rows(backend, scope_a, "edge", [("user", t) for t in [
        "the quick brown fox", "quick bridge", "brown quick", "quick needle", "QUICK", "unrelated words"]])
    # CONTRACT: each retry runs with the caller's offset, so paging past the two exact hits returns any-word hits.
    assert ids(backend, scope_a, "quick brown", sort=OLDEST) == [rows[0], rows[2]]
    assert ids(backend, scope_a, "quick brown", sort=OLDEST, offset=2) == [rows[2], rows[3], rows[4]]


# ── search_messages: the hit ──────────────────────────────────────────────────


def test_hit_carries_row_and_session_fields_and_a_marked_snippet(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = seed_rows(backend, scope_a, "lex", LEXICON, model="lex-model", started_at=T0 - 5)
    (hit,) = store_a.search.search_messages("toolbody")
    # CONTRACT: id, session, role, timestamp and tool name of the row; source, model and start of its session (epoch floats).
    assert (hit.message_id, hit.session_id, hit.role, hit.timestamp, hit.tool_name) == (rows[3], "lex", "tool", T0 + 4, "web_search")
    assert (hit.source, hit.model, hit.session_started) == ("cli", "lex-model", T0 - 5)
    # CONTRACT: a word route snippet marks each match.
    assert hit.snippet == ">>>toolbody<<< quick result"
    assert store_a.search.search_messages("web_search")[0].snippet == ">>>web_search<<<"
    assert {h.snippet for h in store_a.search.search_messages("quick fox")} == {"the >>>quick<<< brown >>>fox<<<", "brown the >>>fox<<< >>>quick<<<", ">>>QUICK<<< uppercase >>>Fox<<<"}
    long_body = "prefix " * 300 + "zanzibar " + "suffix " * 300
    store_a.search.seed_message("lex", "assistant", long_body, timestamp=T0 + 60)
    (long_hit,) = store_a.search.search_messages("zanzibar")
    # CONTRACT: long text is cut around the match and the cuts are marked with "...".
    assert long_hit.snippet.startswith("...prefix") and long_hit.snippet.endswith("suffix...") and ">>>zanzibar<<<" in long_hit.snippet
    assert len(long_hit.snippet) < 400


def test_hit_context_is_the_neighbouring_rows_whatever_their_visibility(backend, scope_a):
    store_a = backend.open(scope_a)
    seed = backend.open(scope_a).search.seed_message
    store_a.search.seed_session("ctx", source="cli", started_at=T0)
    seed("ctx", "user", "ctxterm alpha lead", timestamp=T0 + 1)
    seed("ctx", "assistant", "rewound reply " + "x" * 300, timestamp=T0 + 2, visibility=RowVisibility.REWOUND)
    seed("ctx", "user", [{"type": "text", "text": "ctxterm multimodal part"}, {"type": "image_url", "image_url": {"url": "u"}}], timestamp=T0 + 3)
    seed("ctx", "user", [{"type": "image_url", "image_url": {"url": "u"}}], timestamp=T0 + 4)
    seed("ctx", "user", {"text": "dictvalue"}, timestamp=T0 + 5)
    seed("ctx", "user", "short ctxterm tail", timestamp=T0 + 6, display_kind="hidden")
    seed("ctx", "user", "closing ctxterm", timestamp=T0 + 7)
    hits = {h.snippet: h.context for h in store_a.search.search_messages("ctxterm", sort=OLDEST)}
    rewound = "rewound reply " + "x" * 186
    # CONTRACT: previous row, the row, next row by (timestamp, id); text cut to 200; parts flattened, a parts list without text is a placeholder, a mapping is empty.
    assert [(c.role, c.content) for c in hits[">>>ctxterm<<< alpha lead"]] == [("user", "ctxterm alpha lead"), ("assistant", rewound)]
    assert [c.content for c in hits[">>>ctxterm<<< multimodal part\"}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"u\"}}]"]] == [rewound, "ctxterm multimodal part", "[multimodal content]"]
    assert [c.content for c in hits["closing >>>ctxterm<<<"]] == ["short ctxterm tail", "closing ctxterm"]
    assert [c.content for c in store_a.search.search_messages("dictvalue")[0].context] == ["[multimodal content]", "", "short ctxterm tail"]


def test_fields_projects_the_hit(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = seed_rows(backend, scope_a, "lex", LEXICON)
    # CONTRACT: unnamed fields are None (context ()); the projection keeps the hits.
    hits = store_a.search.search_messages("toolbody", fields=["id", "role"])
    assert hits == (SearchHit(message_id=rows[3], role="tool"),)
    (bare,) = store_a.search.search_messages("toolbody", fields=[])
    assert bare.message_id is None and bare.snippet is None and bare.context == ()
    (full,) = store_a.search.search_messages("toolbody", fields=("context", "snippet"))
    assert full.context and full.snippet and full.message_id is None
    # CONTRACT: a string or an unknown name is refused before the query is read.
    with pytest.raises(TypeError):
        store_a.search.search_messages("toolbody", fields="id")
    for query in ("toolbody", ""):
        with pytest.raises(ValueError):
            store_a.search.search_messages(query, fields=["id", "bogus"])


def test_public_transcript_writes_are_searchable(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence.conversation.messages import NewMessage
    from persistence.conversation.sessions import NewSession

    store_a.sessions.create(NewSession("public", "cli", model="pm"))
    row = store_a.messages.append("public", NewMessage("user", "publicneedle body", T0))
    (hit,) = store_a.search.search_messages("publicneedle")
    # CONTRACT: rows written through the messages verbs are found with their session's fields.
    assert (hit.message_id, hit.session_id, hit.source, hit.model) == (row, "public", "cli", "pm")
    assert isinstance(hit.session_started, float)


def test_two_scopes_are_search_isolated_through_one_provider(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    for scope, sentinel in ((scope_a, "sentinelalpha"), (scope_b, "sentinelbeta")):
        seed_rows(backend, scope, "shared", [("user", f"isoterm {sentinel}")])
    # CONTRACT: each scope finds only its own rows, even under include_inactive.
    assert [h.snippet for h in store_a.search.search_messages("isoterm")] == [">>>isoterm<<< sentinelalpha"]
    assert [h.snippet for h in store_b.search.search_messages("isoterm")] == [">>>isoterm<<< sentinelbeta"]
    assert store_b.search.search_messages("sentinelalpha", include_inactive=True) == ()
    assert backend.open(profile_scope("profile-c")).search.search_messages("isoterm") == ()


# ── index recovery: rebuild_search_index, search_index_rebuild_status, retry_deferred_search_recovery ──


def test_rebuild_search_index_keeps_every_answer(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = seed_rows(backend, scope_a, "lex", LEXICON)
    before = {query: set(ids(backend, scope_a, query)) for query in ("quick", "cat", "dep", "漓江水")}
    rebuilt = store_a.search.rebuild_search_index()
    # CONTRACT: the count of rebuilt indexes is a non-negative int, and search answers from the stored rows as before.
    assert isinstance(rebuilt, int) and rebuilt >= 0
    assert {query: set(ids(backend, scope_a, query)) for query in before} == before
    assert before["cat"] == {rows[5]}


def test_rebuild_search_index_leaves_a_pending_deferred_rebuild_pending(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.search.seed_deferred_rebuild(total=10, indexed=4)
    store_a.search.rebuild_search_index()
    # CONTRACT: the synchronous rebuild neither starts nor clears the deferred one.
    assert store_a.search.search_index_rebuild_status() == SearchIndexRebuildStatus(total=10, indexed=4)
    assert store_a.search.rebuild_search_index() >= 0
    assert store_a.search.search_index_rebuild_status() == SearchIndexRebuildStatus(total=10, indexed=4)


def test_search_index_rebuild_status_reads_the_stored_progress(backend, scope_a, scope_b):
    # CONTRACT: None while nothing is pending.
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    assert store_a.search.search_index_rebuild_status() is None
    for total, indexed, percent in ((10, 4, 40), (10, 25, 100), (10, -5, -50), (3, 1, 33)):
        store_a.search.seed_deferred_rebuild(total=total, indexed=indexed)
        status = store_a.search.search_index_rebuild_status()
        # CONTRACT: total and indexed as stored (indexed not clamped); percent is min(100, int(100 * indexed / total)).
        assert (status.total, status.indexed, status.percent) == (total, indexed, percent)
    # CONTRACT: a non-positive total means nothing is pending.
    store_a.search.seed_deferred_rebuild(total=0, indexed=0)
    assert store_a.search.search_index_rebuild_status() is None
    assert store_b.search.search_index_rebuild_status() is None
    with pytest.raises(ValueError):
        SearchIndexRebuildStatus(total=0, indexed=0)


def test_retry_deferred_search_recovery_is_false_with_nothing_deferred(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = seed_rows(backend, scope_a, "lex", LEXICON)
    # CONTRACT: False when this handle deferred no recovery; it never raises and search keeps answering.
    assert store_a.search.retry_deferred_search_recovery() is False
    assert store_a.search.retry_deferred_search_recovery() is False
    assert ids(backend, scope_a, "web_search") == [rows[3]]
