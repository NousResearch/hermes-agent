"""Message behaviours shared by every backend."""

from datetime import UTC, datetime

import pytest

from persistence.conversation.messages import NewMessage, ReplaceMode, TranscriptRead, TranscriptView
from persistence.conversation.search import SearchSort
from persistence.conversation.sessions import NewSession
from tests.contracts._concurrency import run_concurrently
from tests.contracts.conversation.test_messages import totals

SEEDED = datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
STAMP = SEEDED.timestamp()


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_archive_dropped_compares_loaded_content(backend, scope_a, role):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession("kept", "test"))
    repo = backend.open(scope_a).messages
    original = repo.append("kept", NewMessage(role, "  keep  ", STAMP))
    result = repo.replace_transcript("kept", [NewMessage(role, "keep", STAMP)], ReplaceMode.ARCHIVE_DROPPED)
    rows = repo.read_transcript("kept", TranscriptRead(view=TranscriptView.ALL))
    if role == "tool":
        assert result.survivor_ids != (original,)
        assert [row.content for row in rows] == ["  keep  ", "keep"]
    else:
        assert result.survivor_ids == (original,)
        assert [(row.id, row.content) for row in rows] == [(original, "  keep  ")]
    # CONTRACT: a kept row's stamp carries no snapshot (only its id and uid are stamped); an inserted row's carries one.
    assert (result.rows[0].db_row_snapshot is None) is (role != "tool")
    assert (totals(backend, scope_a, "kept")).message_count == 1


def test_unseen_reactions_include_archived_ancestors(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence.conversation.sessions import SessionLineageKind

    repo = backend.open(scope_a).messages
    store_a.sessions.create(NewSession("parent", "test"))
    original = repo.append("parent", NewMessage("user", "hello", STAMP))
    repo.compact("parent", [NewMessage("user", "summary", STAMP)])
    repo.set_reaction("parent", original, "A", author="user")
    store_a.sessions.end("parent", reason="compression")
    store_a.sessions.create(NewSession("child", "test", parent_session_id="parent"))
    store_a.sessions.create(NewSession("branch", "test", parent_session_id="parent", lineage_kind=SessionLineageKind.BRANCH))
    assert repo.take_unseen_reactions("branch") == ()
    outcomes = run_concurrently(lambda: repo.take_unseen_reactions("child"), lambda: repo.take_unseen_reactions("child"))
    taken = [o for o in outcomes if o]
    assert len(taken) == 1 and outcomes.count(()) == 1
    assert [(r.row_id, r.text, r.emoji) for r in taken[0]] == [(original, "hello", "A")]
    live = repo.append("child", NewMessage("user", "rewound", STAMP))
    repo.set_reaction("child", live, "B", author="user")
    repo.deactivate_message("child", live)
    assert repo.take_unseen_reactions("child") == ()


def test_empty_session_is_present_in_cooldown_snapshot(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence import Conflict
    from persistence.conversation.messages import CooldownSnapshot

    repo = backend.open(scope_a).messages
    absent = CooldownSnapshot(False, None, None)
    assert repo.read_compression_cooldown_snapshot("empty") == absent
    repo.restore_compression_cooldown("empty", absent)
    store_a.sessions.create(NewSession("empty", "test"))
    assert repo.read_compression_cooldown_snapshot("empty") == CooldownSnapshot(True, None, None)
    with pytest.raises(Conflict):
        repo.restore_compression_cooldown("empty", absent)
    repo.record_compression_cooldown("empty", STAMP, "recorded")
    repo.clear_compression_cooldown("empty")
    assert repo.read_compression_cooldown_snapshot("empty") == CooldownSnapshot(True, None, None)


@pytest.mark.parametrize("flag", [None, False, True])
@pytest.mark.parametrize("ancestor", [False, True])
def test_hidden_delivery_replay_does_not_require_suppression(backend, scope_a, flag, ancestor):
    store_a = backend.open(scope_a)
    from persistence.conversation.messages import DelegationDelivery

    store_a.sessions.create(NewSession("hidden-delivery", "cli"))
    meta = {"delegation_id": "job"}
    if flag is not None:
        meta["presentation_suppressed"] = flag
    original = store_a.messages.append("hidden-delivery", NewMessage("user", "original", STAMP, display_kind="hidden", display_metadata=meta))
    sid = "hidden-delivery"
    if ancestor:
        store_a.sessions.end(sid, reason="compression")
        sid = "child-delivery"
        store_a.sessions.create(NewSession(sid, "cli", parent_session_id="hidden-delivery"))
    delivery = DelegationDelivery("replay", {"delegation_id": "job"})
    assert store_a.messages.append_delegation_delivery(sid, delivery) == original
    different = store_a.messages.append_delegation_delivery(sid, DelegationDelivery("new", {"delegation_id": "other"}))
    assert different != original
    rows = store_a.messages.read_transcript(sid)
    assert len(rows) == (1 if ancestor else 2)


@pytest.mark.parametrize("suppressed", [False, True])
def test_delivery_visibility_and_replay_survive_compression(backend, scope_a, suppressed):
    store_a = backend.open(scope_a)
    from persistence.conversation.messages import DelegationDelivery

    store_a.sessions.create(NewSession("delivery", "test"))
    repo = backend.open(scope_a).messages
    delivery = DelegationDelivery("result", {"delegation_id": "job", "presentation_suppressed": suppressed})
    original = repo.append_delegation_delivery("delivery", delivery)
    assert repo.append_delegation_delivery("delivery", delivery) == original
    (row,) = repo.read_transcript("delivery")
    assert row.display_kind == ("hidden" if suppressed else "async_delegation_complete")
    store_a.sessions.end("delivery", reason="compression")
    store_a.sessions.create(NewSession("delivery-child", "test", parent_session_id="delivery"))
    assert repo.append_delegation_delivery("delivery-child", delivery) == original
    assert repo.read_transcript("delivery-child") == ()


def test_delivery_turn_fence_checks_conversation_and_keeps_replays(backend, scope_a):
    store_a = backend.open(scope_a)
    from persistence import Conflict
    from persistence.conversation.messages import DelegationDelivery

    repo = backend.open(scope_a).messages
    coord = backend.open(scope_a).coordination
    store_a.sessions.create(NewSession("lease-parent", "test"))
    notice = DelegationDelivery("first", {"delegation_id": "first"})
    original = repo.append_delegation_delivery("lease-parent", notice)
    store_a.sessions.end("lease-parent", reason="compression")
    store_a.sessions.create(NewSession("lease-child", "test", parent_session_id="lease-parent"))
    assert coord.try_acquire_turn_lease(session_id="lease-parent", holder="owner", ttl_seconds=300)
    assert repo.append_delegation_delivery("lease-child", notice) == original
    with pytest.raises(Conflict):
        repo.append_delegation_delivery("lease-child", DelegationDelivery("new", {"delegation_id": "new"}))
    assert repo.read_transcript("lease-child") == ()
    coord.release_turn_lease(session_id="lease-parent", holder="owner")
    coord.try_acquire_compression_lock(session_id="lease-child", holder="compressor", ttl_seconds=300)
    repo.append_delegation_delivery("lease-child", DelegationDelivery("new", {"delegation_id": "new"}))
    assert len(repo.read_transcript("lease-child")) == 1


@pytest.mark.parametrize("separated", [False, True])
def test_ordinal_rewind_uses_repaired_user_turns(backend, scope_a, separated):
    store_a = backend.open(scope_a)
    from tests.contracts._reference.conversation._turn_policy import composite_carrier
    from persistence import RewindRefused
    from persistence.conversation.messages import RewindTargetUnavailable

    repo = backend.open(scope_a).messages
    store_a.sessions.create(NewSession("ordinal", "test"))
    carrier = composite_carrier("old task", "REAL ASK")
    repo.append("ordinal", NewMessage("user", "prefix", STAMP))
    if separated:
        repo.append("ordinal", NewMessage("assistant", "previous answer", STAMP))
    repo.append("ordinal", NewMessage("user", carrier, STAMP))
    repo.append("ordinal", NewMessage("assistant", "answer", STAMP))
    before = repo.read_transcript("ordinal")
    with pytest.raises(RewindRefused):
        repo.rewind_user_turn("ordinal", -1, warm_history=[])
    assert repo.read_transcript("ordinal") == before
    if separated:
        result = repo.rewind_user_turn("ordinal", -1, require_composite=True)
        assert result.live_text == "REAL ASK"
    else:
        with pytest.raises(RewindTargetUnavailable, match="not a compaction carrier"):
            repo.rewind_user_turn("ordinal", -1, require_composite=True)
        assert repo.read_transcript("ordinal") == before


def test_search_observes_public_transcript_writes(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    repo = backend.open(scope_a).messages
    search = backend.open(scope_a).search
    store_a.sessions.create(NewSession("searchable", "test"))
    original = repo.append("searchable", NewMessage("user", "uniqueapple brown fox", STAMP))
    assert [hit.message_id for hit in search.search_messages("uniqueapple")] == [original]
    assert store_b.search.search_messages("uniqueapple") == ()
    search.rebuild_search_index()
    assert [hit.message_id for hit in search.search_messages("uniqueapple")] == [original]
    repo.set_user_message_content("searchable", original, "uniquebanana")
    assert search.search_messages("uniqueapple") == ()
    assert [hit.message_id for hit in search.search_messages("uniquebanana")] == [original]
    repo.deactivate_message("searchable", original)
    assert search.search_messages("uniquebanana") == ()
    assert [hit.message_id for hit in search.search_messages("uniquebanana", include_inactive=True)] == [original]


@pytest.mark.parametrize("query, rows, expected", [
    ('"quick br"*', ["quick brown", "quick bridge", "quick x brown", "brown quick"], [0, 1]),
    ('"chat-se"*', ["chat sender", "chat-send", "chat x sender", "sender chat"], [0, 1]),
    ("chat-se*", ["chat sender", "chat-send", "chat x sender", "sender chat"], [0, 1]),
    (" dep* ", ["deployment", "independent"], [0]),
])
def test_search_prefix_matches_contiguous_token_runs(backend, scope_a, query, rows, expected):
    search = backend.open(scope_a).search
    search.seed_session("prefix", source="cli")
    ids = [search.seed_message("prefix", "user", text, timestamp=STAMP + i) for i, text in enumerate(rows)]
    hits = search.search_messages(query, sort=SearchSort.OLDEST)
    assert [hit.message_id for hit in hits] == [ids[index] for index in expected]
    assert all(hit.snippet == f">>>{rows[index]}<<<" for hit, index in zip(hits, expected))


def test_search_absent_punctuation_phrase_matches_nothing(backend, scope_a):
    search = backend.open(scope_a).search
    search.seed_session("punctuation", source="cli")
    search.seed_message("punctuation", "user", "plain text", timestamp=STAMP)
    assert search.search_messages('"!!!"') == ()


@pytest.mark.parametrize("query, expected", [("!!!", [1, 7]), ("???", [2]), ("...", [6]), ("!!", []), ("!", []), (";;;", [])])
def test_search_punctuation_phrase_preserves_literal_hits(backend, scope_a, query, expected):
    search = backend.open(scope_a).search
    search.seed_session("literal", source="cli")
    corpus = ["plain text", "!!!", "???", "!!", "!", ".", "...", "prefix !!! suffix"]
    ids = [search.seed_message("literal", "user", text, timestamp=STAMP + i) for i, text in enumerate(corpus)]
    hits = search.search_messages(f'"{query}"', sort=SearchSort.OLDEST)
    assert [hit.message_id for hit in hits] == [ids[index] for index in expected]
    assert search.search_messages(query) == ()


def test_search_extension_b_uses_cjk_substrings(backend, scope_a):
    search = backend.open(scope_a).search
    search.seed_session("extension", source="cli")
    expected = search.seed_message("extension", "user", "前𠀀𠀁后", timestamp=STAMP)
    search.seed_message("extension", "user", "不相关", timestamp=STAMP)
    assert [hit.message_id for hit in search.search_messages("𠀀")] == [expected]


@pytest.mark.parametrize("query", ["桂林abc", "桂林ab*"])
def test_search_cjk_substrings_ignore_ascii_case(backend, scope_a, query):
    search = backend.open(scope_a).search
    search.seed_session("mixed-cjk", source="cli")
    expected = [search.seed_message("mixed-cjk", "user", text, timestamp=STAMP + i) for i, text in enumerate(("桂林ABC", "桂林abc", "前桂林ABC后"))]
    search.seed_message("mixed-cjk", "user", "桂林xyz", timestamp=STAMP + 5)
    hits = search.search_messages(query, sort=SearchSort.OLDEST)
    assert [hit.message_id for hit in hits] == expected[::-1]
    assert [hit.snippet for hit in hits] == ["前桂林ABC后", "桂林abc", "桂林ABC"]


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
@pytest.mark.parametrize("content, stored", [(0, "0"), (False, "0"), (True, "1"), (1.25, "1.25"), ("", ""), (None, None), ({"text": "hello"}, {"text": "hello"}), ([{"type": "text", "text": "hello"}], [{"type": "text", "text": "hello"}])])
def test_scalar_content_uses_stored_text_representation(backend, scope_a, role, content, stored):
    store_a = backend.open(scope_a)
    from persistence.conversation.messages import ConversationRequest

    store_a.sessions.create(NewSession("scalar", "test"))
    store_a.messages.append("scalar", NewMessage(role, content, STAMP))
    (row,) = store_a.messages.read_transcript("scalar")
    (replayed,) = store_a.messages.read_conversation("scalar", ConversationRequest(repair_alternation=False))
    assert row.content == stored
    assert replayed.as_openai_message()["content"] == stored


@pytest.mark.parametrize("operation", ["append", "batch", "compact", "replace"])
def test_invalid_message_epochs_use_the_insertion_clock(backend, scope_a, operation, monkeypatch):
    store_a = backend.open(scope_a)
    from persistence.conversation.messages import TranscriptRead

    clock = STAMP
    monkeypatch.setattr("time.time", lambda: clock)
    store_a.sessions.create(NewSession("epochs", "test"))
    epochs = (1.5, 0, -0.5, -2)
    messages = [NewMessage("user", str(epoch), epoch) for epoch in epochs]
    if operation == "append":
        for message in messages:
            store_a.messages.append("epochs", message)
    elif operation == "batch":
        store_a.messages.append_batch("epochs", messages)
    elif operation == "compact":
        store_a.messages.compact("epochs", messages)
    else:
        store_a.messages.replace_transcript("epochs", messages, ReplaceMode.DESTRUCTIVE)
    rows = store_a.messages.read_transcript("epochs", TranscriptRead())
    fallback = clock
    expected = []
    for epoch in epochs:
        stored = epoch if epoch >= 0 else (clock if operation == "append" else fallback)
        expected.append(stored)
        fallback = max(fallback, stored) + 1e-6
    assert [row.timestamp for row in rows] == expected


@pytest.mark.parametrize("seen", [False, True])
def test_reactions_from_metadata_preserve_other_authors(backend, scope_a, seen):
    store_a = backend.open(scope_a)
    from persistence.conversation.messages import ConversationRequest

    store_a.sessions.create(NewSession("metadata-reactions", "test"))
    original = {"author": "user", "emoji": "R", "at": STAMP, "seen": seen, "extra": "retained"}
    metadata = {"other": {"keep": True}, "reactions": [original]}
    message_id = store_a.messages.append("metadata-reactions", NewMessage("user", "preserved", STAMP, display_metadata=metadata))
    (record,) = store_a.messages.read_transcript("metadata-reactions")
    assert record.display_metadata == metadata
    request = ConversationRequest(repair_alternation=False)
    (replayed,) = store_a.messages.read_conversation("metadata-reactions", request)
    assert replayed.as_openai_message()["display_metadata"] == metadata
    taken = store_a.messages.take_unseen_reactions("metadata-reactions")
    assert [reaction.emoji for reaction in taken] == ([] if seen else ["R"])
    assert store_a.messages.take_unseen_reactions("metadata-reactions") == ()
    result = store_a.messages.set_reaction("metadata-reactions", message_id, "S", author="agent")
    assert [(reaction.author, reaction.emoji) for reaction in result] == [("user", "R"), ("agent", "S")]
    (replayed,) = store_a.messages.read_conversation("metadata-reactions", request)
    after = replayed.as_openai_message()["display_metadata"]
    assert after["other"] == {"keep": True}
    assert after["reactions"][0] == {**original, "seen": True}
    store_a.messages.set_reaction("metadata-reactions", message_id, "S", author="agent")
    (replayed,) = store_a.messages.read_conversation("metadata-reactions", request)
    assert replayed.as_openai_message()["display_metadata"]["reactions"] == [{**original, "seen": True}]


@pytest.mark.parametrize("before, after, keep", [
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}, False),
    ([{"type": "text", "text": "one"}], [{"text": "one", "type": "text"}], False),
    (False, 0, False),
    (False, "0", True),
    ({"a": 1, "b": 2}, {"a": 1, "b": 2}, True),
])
def test_replacement_compares_serialized_loaded_identity(backend, scope_a, before, after, keep):
    store_a = backend.open(scope_a)
    from persistence import RowVisibility

    store_a.sessions.create(NewSession("identity", "test"))
    original = store_a.messages.append("identity", NewMessage("user", before, STAMP))
    result = store_a.messages.replace_transcript("identity", [NewMessage("user", after, STAMP)], ReplaceMode.ARCHIVE_DROPPED)
    rows = store_a.messages.read_transcript("identity", TranscriptRead(view=TranscriptView.ALL))
    assert (result.survivor_ids == (original,)) is keep
    assert [row.visibility for row in rows] == ([RowVisibility.ACTIVE] if keep else [RowVisibility.REWOUND, RowVisibility.ACTIVE])
    assert rows[0].id == original
    assert rows[-1].id == result.survivor_ids[0]


@pytest.mark.parametrize("reordered", [False, True])
def test_replacement_tool_calls_preserve_key_order(backend, scope_a, reordered):
    store_a = backend.open(scope_a)
    from persistence import RowVisibility

    store_a.sessions.create(NewSession("tool-identity", "test"))
    first = {"id": "call", "function": {"name": "work", "arguments": "{}"}}
    second = {"function": {"arguments": "{}", "name": "work"}, "id": "call"} if reordered else first
    original = store_a.messages.append("tool-identity", NewMessage("assistant", "call", STAMP, tool_calls=(first,)))
    result = store_a.messages.replace_transcript("tool-identity", [NewMessage("assistant", "call", STAMP, tool_calls=(second,))], ReplaceMode.ARCHIVE_DROPPED)
    rows = store_a.messages.read_transcript("tool-identity", TranscriptRead(view=TranscriptView.ALL))
    assert (result.survivor_ids == (original,)) is not reordered
    assert rows[0].visibility is (RowVisibility.REWOUND if reordered else RowVisibility.ACTIVE)


@pytest.mark.parametrize("content, options, query", [
    ({"message": "mappingneedle"}, {}, "mappingneedle"),
    ([{"type": "image_url", "image_url": {"url": "https://invalid.example/imageneedle.png"}}], {}, "imageneedle"),
    (1234567, {}, "1234567"),
    (False, {}, "0"),
    (True, {}, "1"),
    (1.25, {}, "1.25"),
    ([{"type": "text", "text": "hello", "metadata": "metaneedle"}], {}, "metaneedle"),
    ("hello", {"tool_calls": ({"id": "x", "type": "function", "function": {"name": "callneedle", "arguments": "{}"}},)}, "callneedle"),
    ([{"type": "text", "text": "textneedle"}], {}, "textneedle"),
])
def test_search_uses_stored_payload_fields(backend, scope_a, content, options, query):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession("payload", "cli"))
    original = store_a.messages.append("payload", NewMessage("user", content, STAMP, **options))
    store_a.messages.append("payload", NewMessage("user", "unrelated", STAMP))
    hits = store_a.search.search_messages(query)
    assert [hit.message_id for hit in hits] == [original]
    assert f">>>{query}<<<" in hits[0].snippet
    assert store_a.search.search_messages("multimodal") == ()
    assert store_a.search.search_messages("absentneedle") == ()


@pytest.mark.parametrize("seed_first", [False, True])
@pytest.mark.parametrize("same_session", [False, True])
def test_seed_and_public_messages_share_ids_and_corpus(backend, scope_a, seed_first, same_session):
    store_a = backend.open(scope_a)
    search = backend.open(scope_a).search
    public_sid = "seed" if same_session else "public"
    ids = []
    if seed_first:
        search.seed_session("seed", source="cli")
        ids.append(search.seed_message("seed", "user", "seed matchable", timestamp=STAMP))
    store_a.sessions.create(NewSession(public_sid, "cli"))
    ids.append(store_a.messages.append(public_sid, NewMessage("user", "public matchable", STAMP)))
    if not seed_first:
        search.seed_session("seed", source="cli")
        ids.append(search.seed_message("seed", "user", "seed matchable", timestamp=STAMP))
    assert len(set(ids)) == 2
    found = [search.search_messages("matchable", sort=SearchSort.OLDEST, limit=1, offset=offset)[0].message_id for offset in range(2)]
    assert sorted(found) == sorted(ids)
    assert search.search_messages("matchable", sort=SearchSort.OLDEST, limit=1, offset=2) == ()
    assert search.search_messages("absent") == ()
    seeded = ids[0] if seed_first else ids[1]
    assert store_a.messages.set_user_message_content("seed", seeded, "changed")
    assert [hit.message_id for hit in search.search_messages("matchable")] == [item for item in ids if item != seeded]


@pytest.mark.parametrize("include_inactive", [False, True])
def test_search_excludes_hidden_candidates_but_keeps_context(backend, scope_a, include_inactive):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession("hidden", "cli"))
    store_a.messages.append("hidden", NewMessage("user", "needle hidden", STAMP, display_kind="hidden"))
    visible = store_a.messages.append("hidden", NewMessage("user", "needle notice", STAMP, display_kind="async_delegation_complete"))
    for role_filter in (None, ["tool", "user"]):
        hits = store_a.search.search_messages("needle", include_inactive=include_inactive, role_filter=role_filter)
        assert [hit.message_id for hit in hits] == [visible]
        assert any(row.content == "needle hidden" for row in hits[0].context)


@pytest.mark.parametrize("character, interior", [("〇", True), ("\u8c48", True), ("\uf900", False), ("\U0002a6df", True)])
@pytest.mark.parametrize("kind", ["term", "phrase"])
def test_search_cjk_classifier_boundaries(backend, scope_a, character, interior, kind):
    search = backend.open(scope_a).search
    search.seed_session("range", source="cli")
    ids = [search.seed_message("range", "user", value, timestamp=STAMP + i) for i, value in enumerate((f"前{character}后", character, "无"))]
    query = character if kind == "term" else f'"{character}"'
    assert {hit.message_id for hit in search.search_messages(query)} == set(ids[:2] if interior else ids[1:2])


@pytest.mark.parametrize("query, corpus", [("林", ["桂林山水", "林山水", "桂山水"]), ("桂林ab", ["前桂林ABC后", "桂林abc", "桂林xyz"])])
def test_search_cjk_prefixes_match_inside_runs(backend, scope_a, query, corpus):
    search = backend.open(scope_a).search
    search.seed_session("cjk-prefix", source="cli")
    ids = [search.seed_message("cjk-prefix", "user", content, timestamp=STAMP + i) for i, content in enumerate(corpus)]
    hits = search.search_messages(query + "*")
    assert [hit.message_id for hit in hits] == ids[:2][::-1]
    assert all(query.lower() in hit.snippet.lower() for hit in hits)


@pytest.mark.parametrize("query, expected", [("桂林é", [1]), ("山桂林é", [0, 1])])
@pytest.mark.parametrize("kind", ["term", "phrase", "prefix"])
def test_search_cjk_case_follows_short_and_long_routes(backend, scope_a, query, expected, kind):
    search = backend.open(scope_a).search
    search.seed_session("cjk-case", source="cli")
    ids = [search.seed_message("cjk-case", "user", text, timestamp=STAMP + i) for i, text in enumerate((query[:-1] + "É", query, query[:-1] + "e"))]
    text = {"term": query, "phrase": f'"{query}"', "prefix": query + "*"}[kind]
    assert {hit.message_id for hit in search.search_messages(text)} == {ids[index] for index in expected}


@pytest.mark.parametrize("query, corpus, expected", [
    ("山水湖i", ["山水湖I", "山水湖i", "山水湖ı", "山水湖İ", "山水湖x"], [0, 1]),
    ("山水湖İ", ["山水湖I", "山水湖i", "山水湖ı", "山水湖İ"], [3]),
    ("山水湖k", ["山水湖K", "山水湖k", "山水湖K", "山水湖x"], [0, 1, 2]),
    ("山水湖s", ["山水湖S", "山水湖s", "山水湖ſ", "山水湖x"], [0, 1, 2]),
    ("山水湖ß", ["山水湖ß", "山水湖ẞ", "山水湖ss"], [0, 1]),
    ("山水湖σ", ["山水湖Σ", "山水湖σ", "山水湖ς", "山水湖x"], [0, 1, 2]),
])
@pytest.mark.parametrize("kind", ["term", "phrase", "prefix"])
def test_long_cjk_search_uses_single_character_case_folding(backend, scope_a, query, corpus, expected, kind):
    search = backend.open(scope_a).search
    search.seed_session("cjk-fold", source="cli")
    ids = [search.seed_message("cjk-fold", "user", text, timestamp=STAMP + i) for i, text in enumerate(corpus)]
    text = f'"{query}"' if kind == "phrase" else query + ("*" if kind == "prefix" else "")
    hits = search.search_messages(text, sort=SearchSort.OLDEST)
    assert [hit.message_id for hit in hits] == [ids[index] for index in expected]
    assert [hit.snippet for hit in hits] == [f">>>{corpus[index]}<<<" for index in expected]
    assert search.search_messages("山水湖z") == ()


def test_long_cjk_highlight_offsets_keep_unfolded_characters(backend, scope_a):
    search = backend.open(scope_a).search
    search.seed_session("cjk-offset", source="cli")
    original = search.seed_message("cjk-offset", "user", "İ前 山水湖I 后山水湖ı", timestamp=STAMP)
    (hit,) = search.search_messages("山水湖i")
    assert hit.message_id == original
    assert hit.snippet == "İ前 >>>山水湖I<<< 后山水湖ı"


@pytest.mark.parametrize("content, stored, replacement, keep, query, replacement_hits", [
    (1e20, "1.0e+20", "1e+20", False, "1.0e", False),
    (1e20, "1.0e+20", "1.0e+20", True, "1.0e", True),
    (1e-7, "1.0e-07", "1e-07", False, "1.0e", False),
    (-0.0, "0.0", "-0.0", False, "0.0", True),
    (-0.0, "0.0", "0.0", True, "0.0", True),
    (0.0, "0.0", "0.0", True, "0.0", True),
    (float("nan"), None, "nan", False, "nan", True),
    (float("nan"), None, None, True, "nan", False),
    (float("inf"), "Inf", "inf", False, "Inf", True),
    (float("inf"), "Inf", "Inf", True, "Inf", True),
    (float("-inf"), "-Inf", "-inf", False, "Inf", True),
    (float("-inf"), "-Inf", "-Inf", True, "Inf", True),
    (1.25, "1.25", "1.25", True, "1.25", True),
    (1.25, "1.25", 1.25, False, "1.25", True),
])
def test_float_content_preserves_storage_replacement_and_search(backend, scope_a, content, stored, replacement, keep, query, replacement_hits):
    store_a = backend.open(scope_a)
    from persistence import RowVisibility
    from persistence.conversation.messages import ConversationRequest, TranscriptRead

    store_a.sessions.create(NewSession("float-content", "cli"))
    repo = backend.open(scope_a).messages
    original = repo.append("float-content", NewMessage("user", content, STAMP))
    (row,) = repo.read_transcript("float-content")
    (wire,) = repo.read_transcript("float-content", TranscriptRead())
    (replayed,) = repo.read_conversation("float-content", ConversationRequest(repair_alternation=False))
    assert row.content == wire.content == replayed.as_openai_message()["content"] == stored
    before_ids = {original} if stored is not None else set()
    assert {hit.message_id for hit in store_a.search.search_messages(query)} == before_ids
    result = repo.replace_transcript("float-content", [NewMessage("user", replacement, STAMP)], ReplaceMode.ARCHIVE_DROPPED)
    rows = repo.read_transcript("float-content", TranscriptRead(view=TranscriptView.ALL))
    assert (result.survivor_ids == (original,)) is keep
    assert rows[0].id == original
    assert rows[0].content == stored
    assert [row.visibility for row in rows] == ([RowVisibility.ACTIVE] if keep else [RowVisibility.REWOUND, RowVisibility.ACTIVE])
    assert rows[-1].id == result.survivor_ids[0]
    assert rows[-1].content == (stored if keep or isinstance(replacement, float) else replacement)
    after_ids = set(result.survivor_ids) if replacement_hits else set()
    assert {hit.message_id for hit in store_a.search.search_messages(query)} == after_ids
    assert {hit.message_id for hit in store_a.search.search_messages(query, include_inactive=True)} == before_ids | after_ids


@pytest.mark.parametrize("epoch", ["1700000000.125", 1700000000.125, "0", ""])
def test_reaction_return_converts_ingested_epoch_without_rewriting_metadata(backend, scope_a, epoch):
    store_a = backend.open(scope_a)
    from persistence.conversation.messages import ConversationRequest, TranscriptRead

    store_a.sessions.create(NewSession("reaction-epoch", "cli"))
    repo = backend.open(scope_a).messages
    original = {"author": "user", "emoji": "R", "at": epoch, "extra": "retained"}
    metadata = {"other": {"keep": True}, "reactions": [original]}
    message_id = repo.append("reaction-epoch", NewMessage("user", "reaction carrier", STAMP, display_metadata=metadata))
    result = repo.set_reaction("reaction-epoch", message_id, "S", author="agent")
    assert [(reaction.author, reaction.emoji) for reaction in result] == [("user", "R"), ("agent", "S")]
    assert result[0].at == float(epoch or 0.0)
    (row,) = repo.read_transcript("reaction-epoch", TranscriptRead())
    (replayed,) = repo.read_conversation("reaction-epoch", ConversationRequest(repair_alternation=False))
    after = row.display_metadata
    assert replayed.as_openai_message()["display_metadata"] == after
    assert after["other"] == {"keep": True}
    assert after["reactions"][0] == original
    assert after["reactions"][1]["author"] == "agent"
    assert after["reactions"][1]["emoji"] == "S"
    assert result[1].at == after["reactions"][1]["at"]
    assert repo.set_reaction("reaction-epoch", 999999, "S", author="agent") is None
    # The author is stored as given, an empty one included.
    unnamed = repo.set_reaction("reaction-epoch", message_id, "S", author="")
    assert [(reaction.author, reaction.emoji) for reaction in unnamed] == [("user", "R"), ("agent", "S"), ("", "S")]
    (changed,) = repo.read_transcript("reaction-epoch", TranscriptRead())
    assert changed.display_metadata["reactions"][:2] == after["reactions"]
