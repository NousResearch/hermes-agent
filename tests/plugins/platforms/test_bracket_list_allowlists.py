"""Allowlists stored as a JSON-list *string* are honoured by every platform gate (issue #76457).

Configs written by ``hermes config set KEY '["-100","-200"]'`` before the writer learned
to emit YAML lists hold the literal ``'["-100","-200"]'`` as a string. The Telegram gate
already decodes that shape (``gateway/platforms/_shared.py::decode_json_list_literal``);
the Discord / WhatsApp / DingTalk / Matrix gates comma-split it into one bogus entry that
matches nothing, silently locking out every allowlisted chat, room or user.
"""

from gateway.config import Platform, PlatformConfig

BRACKET_LIST = '["-100", "-200"]'


def _discord(extra):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="x", extra=dict(extra))
    adapter._gate_env_snapshot = None
    return adapter


def _whatsapp(extra):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=dict(extra))
    return adapter


def _dingtalk(extra):
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    return DingTalkAdapter(PlatformConfig(enabled=True, extra=dict(extra)))


def _matrix(extra, monkeypatch):
    """A real MatrixAdapter. ``_extra_csv_set`` reads the scoped env var FIRST, and under a
    default-profile install that rung is ``os.environ`` — so a MATRIX_* value bridged by the host
    would decide the result instead of the ``extra`` under test."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    for env in ("MATRIX_ALLOWED_ROOMS", "MATRIX_ALLOWED_USERS", "MATRIX_FREE_RESPONSE_ROOMS",
                "MATRIX_IGNORE_USER_PATTERNS"):
        monkeypatch.delenv(env, raising=False)
    return MatrixAdapter(PlatformConfig(enabled=True, token="x", extra=dict(extra)))


MATRIX_FIELDS = (("allowed_rooms", "_allowed_rooms"), ("allowed_users", "_allowed_user_ids"),
                 ("free_response_rooms", "_free_rooms"))
ROOM_LIST = '["!aaa:srv", "!bbb:srv"]'
ROOMS = {"!aaa:srv", "!bbb:srv"}


def test_matrix_bracket_list_string_is_decoded_for_every_list(monkeypatch):
    """All three Matrix lists resolve through ``_extra_csv_set``, so one decode covers them —
    but each is asserted, because each is a separate gate with a separate failure."""
    for key, attr in MATRIX_FIELDS:
        got = getattr(_matrix({key: ROOM_LIST}, monkeypatch), attr)
        assert got == ROOMS, (key, got)
    # the room whitelist has a second, independently-read attribute
    assert _matrix({"allowed_rooms": ROOM_LIST}, monkeypatch)._allowed_room_ids == ROOMS
    users = _matrix({"allowed_users": '["@a:srv", "@b:srv"]'}, monkeypatch)._allowed_user_ids
    assert users == {"@a:srv", "@b:srv"}


def test_matrix_csv_and_yaml_list_keep_their_meaning(monkeypatch):
    """The two shapes that already worked must be untouched: a CSV string written by
    ``hermes config set`` (its native shape for a string-typed key) and a real YAML list."""
    for raw, want in (("!aaa:srv, !bbb:srv", ROOMS), (["!aaa:srv", "!bbb:srv"], ROOMS),
                      ("!aaa:srv", {"!aaa:srv"}), ("", set()), ([], set())):
        for key, attr in MATRIX_FIELDS:
            got = getattr(_matrix({key: raw}, monkeypatch), attr)
            assert got == want, (key, raw, got)


def test_matrix_empty_literal_keeps_its_current_meaning(monkeypatch):
    """``'[]'`` must NOT decode to an empty set.

    ``_is_allowed_matrix_room_event`` is permissive on an empty set, so an empty allowlist admits
    every joined room. Decoding ``'[]'`` would therefore convert a value that denies every group
    room into one that admits all of them — a semantic change this decode fix does not make.
    """
    for key, attr in MATRIX_FIELDS:
        got = getattr(_matrix({key: "[]"}, monkeypatch), attr)
        assert got == {"[]"}, (key, got)
    assert _matrix({"allowed_rooms": "[]"}, monkeypatch)._allowed_room_ids == {"[]"}
    # and the two shapes that DO mean "unset" still do
    for raw in ("", None):
        assert _matrix({"allowed_rooms": raw}, monkeypatch)._allowed_room_ids == set()
    assert _matrix({}, monkeypatch)._allowed_room_ids == set()


def test_matrix_malformed_authorization_list_fails_closed(monkeypatch):
    """A non-empty garbage list must never widen access.

    ``allowed_rooms`` is a whitelist, so a set matching no room rejects every group room
    (``_is_allowed_matrix_room_event``); ``allowed_users`` matching no sender authorizes nobody.
    Both stay closed, and the malformed value keeps the legacy comma-split path.
    """
    for raw in ("[not-json", '["a", "b"', '{"a":1}'):
        m = _matrix({"allowed_rooms": raw, "allowed_users": raw}, monkeypatch)
        assert m._allowed_room_ids, raw
        assert "!real:srv" not in m._allowed_room_ids, raw
        assert "@real:srv" not in m._allowed_user_ids, raw


def test_matrix_decode_does_not_touch_require_mention_or_ignore_patterns(monkeypatch):
    """``ignore_user_patterns`` is read by ``_csv_set`` directly, NOT by ``_extra_csv_set``, and
    must keep that path: a regex may legitimately begin with ``[`` or contain a comma
    (``{1,3}``), and every profile writes the field as a real YAML list."""
    m = _matrix({"require_mention": False, "thread_require_mention": True,
                 "ignore_user_patterns": ["[0-9]+", r"^@a{1,3}:srv$"]}, monkeypatch)
    assert m._require_mention is False
    assert m._thread_require_mention is True
    assert {p.pattern for p in m._ignored_user_patterns} == {"[0-9]+", r"^@a{1,3}:srv$"}
    assert _matrix({"require_mention": True}, monkeypatch)._require_mention is True


def test_bracket_list_string_is_decoded_by_every_platform_gate():
    assert _discord({"allowed_channels": BRACKET_LIST})._get_allowed_channels() == {"-100", "-200"}
    assert _discord({"free_response_channels": BRACKET_LIST})._discord_free_response_channels() == {"-100", "-200"}
    assert _whatsapp({"free_response_chats": BRACKET_LIST})._whatsapp_free_response_chats() == {"-100", "-200"}
    dingtalk = _dingtalk({"allowed_chats": BRACKET_LIST, "allowed_users": '["Alice"]'})
    assert dingtalk._dingtalk_allowed_chats() == {"-100", "-200"}
    assert dingtalk._allowed_users == {"alice"}


def test_plain_csv_and_yaml_lists_keep_their_meaning():
    assert _discord({"allowed_channels": "-100, -200"})._get_allowed_channels() == {"-100", "-200"}
    assert _discord({"allowed_channels": ["-100", "-200"]})._get_allowed_channels() == {"-100", "-200"}
    assert _whatsapp({"free_response_chats": "a,b"})._whatsapp_free_response_chats() == {"a", "b"}
    # Malformed JSON is not a list: it stays on the legacy comma-split path.
    assert _dingtalk({"allowed_chats": "[not-json"})._dingtalk_allowed_chats() == {"[not-json"}
