"""Profile-local phrases that suppress memory prefetch for whole-message acks."""

from agent.memory_provider import is_trivial_prompt, record_trivial_acknowledgement


def _use_home(monkeypatch, home):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)


def test_phrase_file_is_read_and_reloaded_without_matching_sentences(tmp_path, monkeypatch):
    _use_home(monkeypatch, tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    phrases = state / "trivial_ack_phrases.txt"

    assert not is_trivial_prompt("de una")
    phrases.write_text("\ufeff# one whole message per line\nde una\n", encoding="utf-8")
    assert is_trivial_prompt("de una")
    assert not is_trivial_prompt("de una, revisa la causa")

    phrases.write_text("# edited in place\nde una\nsin problema\n", encoding="utf-8")
    assert is_trivial_prompt("sin problema")

    phrases.unlink()
    assert not is_trivial_prompt("sin problema")


def test_phrase_file_is_profile_scoped_and_rejects_non_phrases(tmp_path, monkeypatch):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    for home, phrase in ((profile_a, "confirmado"), (profile_b, "recibido")):
        state = home / "state"
        state.mkdir(parents=True)
        (state / "trivial_ack_phrases.txt").write_text(
            f"{phrase}\nignore this phrase 12\n^.*$\n", encoding="utf-8"
        )

    _use_home(monkeypatch, profile_a)
    assert is_trivial_prompt("confirmado")
    assert not is_trivial_prompt("recibido")
    assert not is_trivial_prompt("ignore this phrase 12")
    assert not is_trivial_prompt("anything")

    _use_home(monkeypatch, profile_b)
    assert is_trivial_prompt("recibido")
    assert not is_trivial_prompt("confirmado")


def test_recording_a_phrase_grows_the_list(tmp_path, monkeypatch):
    _use_home(monkeypatch, tmp_path)

    assert record_trivial_acknowledgement("Entendido")
    assert is_trivial_prompt("entendido")
    assert record_trivial_acknowledgement("entendido")  # idempotent
    assert (tmp_path / "state" / "trivial_ack_phrases.txt").read_text(
        encoding="utf-8"
    ) == "entendido\n"


def test_recording_refuses_regexes_digits_and_long_text(tmp_path, monkeypatch):
    _use_home(monkeypatch, tmp_path)
    phrases = tmp_path / "state" / "trivial_ack_phrases.txt"

    for text in ("", "   ", "^.*$", "ignore this phrase 12", "one two three four five"):
        assert not record_trivial_acknowledgement(text)
    assert not phrases.exists()
    assert not is_trivial_prompt("ignore this phrase 12")


def test_a_recorded_phrase_only_matches_the_whole_message(tmp_path, monkeypatch):
    _use_home(monkeypatch, tmp_path)

    assert record_trivial_acknowledgement("de una")
    assert is_trivial_prompt("de una")
    assert not is_trivial_prompt("de una, revisa la causa")


def test_sender_prefix_alone_is_not_treated_as_an_ack():
    assert not is_trivial_prompt("[Sender]")
