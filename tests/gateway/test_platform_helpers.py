"""Direct unit tests for shared per-chat gating helpers in gateway/platforms/helpers.py."""

from gateway.platforms.helpers import parse_chat_id_set


class TestParseChatIdSet:
    """Contract: list/tuple/set → str-coerced stripped members; string → CSV;
    None/blank → empty set. Every adapter's native_mention_only_* reader
    relies on these shapes."""

    def test_none_and_blank_yield_empty_set(self):
        assert parse_chat_id_set(None) == set()
        assert parse_chat_id_set("") == set()
        assert parse_chat_id_set("   ") == set()
        assert parse_chat_id_set([]) == set()

    def test_csv_string_strips_whitespace_and_blank_entries(self):
        assert parse_chat_id_set("a, b ,,c") == {"a", "b", "c"}

    def test_list_strips_and_drops_blank_entries(self):
        assert parse_chat_id_set(["a", " b ", ""]) == {"a", "b"}

    def test_tuple_and_set_inputs_are_accepted(self):
        assert parse_chat_id_set((1, 2)) == {"1", "2"}
        assert parse_chat_id_set({"x"}) == {"x"}

    def test_non_string_members_are_str_coerced(self):
        # Telegram chat ids are commonly written as bare YAML ints.
        assert parse_chat_id_set([-1001234, "C1"]) == {"-1001234", "C1"}

    def test_bare_scalar_string_is_single_member(self):
        assert parse_chat_id_set("C0B0F0AS084") == {"C0B0F0AS084"}
