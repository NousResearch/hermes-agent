"""SurrogateSafeFileHistory: undecodable stdin bytes are persisted losslessly."""

from hermes_cli.cli_history import SurrogateSafeFileHistory, encode_history_text


def test_escaped_stdin_byte_is_written_back_as_original_byte(tmp_path):
    path = tmp_path / "history"
    raw = b"GE\x94FT \xc3\xbc"  # 0x94 is not valid UTF-8; \xc3\xbc is "ü"
    text = raw.decode("utf-8", errors="surrogateescape")  # as prompt_toolkit reads stdin
    assert "\udc94" in text

    SurrogateSafeFileHistory(str(path)).store_string(text)  # stock FileHistory raises here

    assert b"+" + raw + b"\n" in path.read_bytes()


def test_history_round_trips_text_exactly(tmp_path):
    path = tmp_path / "history"
    first = "GE\udc94FT"
    second = "line one\nzweite Zeile \udc80 ok"
    history = SurrogateSafeFileHistory(str(path))
    history.store_string(first)
    history.store_string(second)

    loaded = list(SurrogateSafeFileHistory(str(path)).load_history_strings())

    assert loaded == [second, first]


def test_non_escape_lone_surrogate_is_kept_not_dropped():
    encoded = encode_history_text("a\ud83db")  # half of an emoji pair
    assert encoded == b"a" + "\ud83d".encode("utf-8", "surrogatepass") + b"b"


def test_plain_text_encoding_unchanged():
    assert encode_history_text("Grüße ✓") == "Grüße ✓".encode()


def test_missing_file_loads_empty(tmp_path):
    assert list(SurrogateSafeFileHistory(str(tmp_path / "none")).load_history_strings()) == []
