import pytest

from agent.stream_repetition_guard import (
    REPETITION_CUT_MARKER,
    StreamRepetitionLoopError,
    StreamingRepetitionGuard,
    find_cross_turn_repeated_line_from_env,
    find_repeated_line_from_env,
    truncate_at_repetition,
)
INCIDENT_LINE = (
    "- **Texture** : `TEXTURE_LOCAL.md` (trace creative, emergence). "
    "Pas dans un Hermes stock.\n"
)


def test_stream_repetition_guard_catches_repeated_status_lines():
    guard = StreamingRepetitionGuard(
        min_total_chars=200,
        min_line_chars=40,
        repeat_threshold=4,
    )
    repeated = (
        "- **Texture** : `TEXTURE_LOCAL.md` (trace creative, emergence). "
        "Pas dans un Hermes stock.\n"
    )

    with pytest.raises(StreamRepetitionLoopError) as exc:
        for _ in range(4):
            guard.feed(repeated)

    assert exc.value.repeat_count == 4
    assert "TEXTURE_LOCAL.md" in exc.value.repeated_line


def test_stream_repetition_guard_ignores_short_repeated_lines():
    guard = StreamingRepetitionGuard(
        min_total_chars=20,
        min_line_chars=40,
        repeat_threshold=4,
    )

    for _ in range(20):
        guard.feed("OK\n")


def test_stream_repetition_guard_handles_split_chunks():
    guard = StreamingRepetitionGuard(
        min_total_chars=200,
        min_line_chars=40,
        repeat_threshold=3,
    )
    line = (
        "**Sanity check : generation loop detected in the current "
        "assistant response.**\n"
    )

    with pytest.raises(StreamRepetitionLoopError):
        for _ in range(3):
            guard.feed(line[:25])
            guard.feed(line[25:])


def test_stream_repetition_guard_catches_no_newline_periodic_tail():
    guard = StreamingRepetitionGuard(
        min_total_chars=120,
        min_line_chars=40,
        repeat_threshold=8,
        tail_check_min_chars=80,
        tail_max_period=40,
        tail_min_repeats=5,
        heavy_window_chars=1000,
    )
    with pytest.raises(StreamRepetitionLoopError) as exc:
        guard.feed("Je vais agir maintenant. " * 12)
    assert "periodic tail" in exc.value.repeated_line


def test_stream_repetition_guard_spares_long_unique_single_line():
    guard = StreamingRepetitionGuard(
        min_total_chars=120,
        min_line_chars=40,
        repeat_threshold=4,
        tail_check_min_chars=80,
        tail_max_period=40,
        tail_min_repeats=5,
        heavy_window_chars=1000,
        heavy_check_every_chars=128,
    )
    guard.feed(
        " ".join(
            f"segment-{i}-avec-un-contenu-specifique-et-progressif"
            for i in range(80)
        )
    )


def test_stream_repetition_guard_catches_low_entropy_window():
    guard = StreamingRepetitionGuard(
        min_total_chars=300,
        min_line_chars=40,
        repeat_threshold=8,
        tail_check_min_chars=10_000,
        tail_min_repeats=999,
        heavy_window_chars=600,
        heavy_check_every_chars=128,
        ngram_size=12,
        distinct_ngram_ratio_threshold=0.25,
        entropy_threshold=2.5,
    )
    with pytest.raises(StreamRepetitionLoopError) as exc:
        guard.feed(("aaaaabaaaaac" * 80)[:900])
    assert "low-entropy" in exc.value.repeated_line


# ── Windowed-density discrimination (incident vs. legitimate spaced repeats) ──
# The guard must fire on a degenerate tail (a few lines cycling to the cap) but
# NOT on legitimate text that merely contains a recurring line interleaved with
# unique content.  Confirmed false positives before windowing: unattended
# night-cycle digests and the companion's quoted-log analysis turns.

def test_windowed_guard_fires_on_block_loop_no_line_ever_adjacent():
    """Incident (b): an A-B-C block loop — no line is ever adjacent to itself,
    so run-length adjacency would miss it, but the block dominates the tail."""
    block = (
        "- **Texture** : `TEXTURE_LOCAL.md` (trace creative). Pas dans stock.\n"
        "- **Blocs** : `state/blocs.md` (verrous, blocages). Pas dans stock.\n"
        "- **Active** : `state/active-items.md` (suivi). Pas dans stock.\n"
    )
    exc = find_repeated_line_from_env("Preambule sain et unique.\n" + block * 15)
    assert exc is not None
    assert exc.repeat_count >= 8


def test_windowed_guard_spares_digest_with_spaced_status_line():
    """A per-section status line repeated across a digest is legitimate: it is
    interleaved with unique headers/hashes so it never dominates the window."""
    digest = "".join(
        f"## Source {i} : rapport du collecteur numero {i}, details varies\n"
        "- RAS pour cette source, aucun changement detecte depuis le passage.\n"
        f"  (hash observe abc{i}def, timestamp variable {i}h30 AST du jour)\n"
        for i in range(12)
    )
    assert find_repeated_line_from_env(digest) is None


def test_windowed_guard_spares_quoted_log_with_recurring_error_line():
    """The companion analysing a log: an identical error line recurs but is
    spaced by unique timestamped lines — must not be flagged as a loop."""
    log = "Voici le log a analyser :\n```\n" + "".join(
        f"2026-07-03 0{i % 10}:15:0{i % 10} INFO demarrage tentative numero {i}\n"
        "ERROR connection refused to 127.0.0.1:10240 - retry in 5s configure\n"
        for i in range(14)
    ) + "```\nConclusion : le port etait sature pendant la fenetre.\n"
    assert find_repeated_line_from_env(log) is None


def test_windowed_guard_max_distinct_separates_loop_from_diverse_repeat():
    """Same recurring line, two shapes: dominated window fires, diverse does not."""
    dominated = StreamingRepetitionGuard(
        min_total_chars=200, min_line_chars=10, repeat_threshold=4, max_distinct=3
    )
    with pytest.raises(StreamRepetitionLoopError):
        for _ in range(4):
            dominated.feed("cette ligne se repete sans rien entre elle du tout\n")

    diverse = StreamingRepetitionGuard(
        min_total_chars=200, min_line_chars=10, repeat_threshold=4, max_distinct=3
    )
    for i in range(20):
        diverse.feed(f"contexte unique et different pour le tour numero {i} ici\n")
        diverse.feed("cette ligne se repete mais espacee par du contenu neuf\n")
    # No raise: the window always holds many distinct lines.


def test_guard_flush_counts_trailing_unterminated_line():
    guard = StreamingRepetitionGuard(
        min_total_chars=200,
        min_line_chars=40,
        repeat_threshold=4,
    )
    for _ in range(3):
        guard.feed(INCIDENT_LINE)
    guard.feed(INCIDENT_LINE.rstrip("\n"))  # no trailing newline

    with pytest.raises(StreamRepetitionLoopError):
        guard.flush()


def test_find_repeated_line_from_env_detects_completed_loop():
    text = "Intro sane paragraph before the degeneration.\n" + INCIDENT_LINE * 20
    exc = find_repeated_line_from_env(text)
    assert exc is not None
    assert "TEXTURE_LOCAL.md" in exc.repeated_line
    assert exc.repeat_count >= 8


def test_find_repeated_line_from_env_clean_text_returns_none():
    text = "\n".join(
        f"Ligne unique numero {i} avec suffisamment de caracteres pour compter."
        for i in range(60)
    )
    assert find_repeated_line_from_env(text) is None


def test_find_repeated_line_from_env_respects_kill_switch(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_REPETITION_GUARD", "0")
    assert find_repeated_line_from_env(INCIDENT_LINE * 30) is None


def test_cross_turn_guard_detects_consecutive_assistant_repetition(monkeypatch):
    monkeypatch.setenv("HERMES_CROSS_TURN_REPETITION_THRESHOLD", "4")
    line = (
        "Je dois verifier l'etat du dashboard avant de conclure que la quete "
        "est terminee.\n"
    )
    messages = [
        {"role": "assistant", "content": line},
        {"role": "tool", "content": "ok", "tool_call_id": "t1"},
        {"role": "assistant", "content": line},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": line},
    ]

    exc = find_cross_turn_repeated_line_from_env(messages, line)

    assert exc is not None
    assert exc.repeat_count == 4
    assert "dashboard" in exc.repeated_line


def test_cross_turn_guard_spares_nonconsecutive_repetition(monkeypatch):
    monkeypatch.setenv("HERMES_CROSS_TURN_REPETITION_THRESHOLD", "4")
    line = (
        "Je dois verifier l'etat du dashboard avant de conclure que la quete "
        "est terminee.\n"
    )
    messages = [
        {"role": "assistant", "content": line},
        {"role": "assistant", "content": "Contenu neuf assez long pour casser la serie."},
        {"role": "assistant", "content": line},
    ]

    assert find_cross_turn_repeated_line_from_env(messages, line) is None


def test_cross_turn_guard_respects_kill_switch(monkeypatch):
    monkeypatch.setenv("HERMES_CROSS_TURN_REPETITION_THRESHOLD", "2")
    monkeypatch.setenv("HERMES_CROSS_TURN_REPETITION_GUARD", "0")
    line = "Une ligne non triviale repetee sur plusieurs tours assistant.\n"

    assert (
        find_cross_turn_repeated_line_from_env(
            [{"role": "assistant", "content": line}],
            line,
        )
        is None
    )


def test_reasoning_from_env_kill_switch(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_REPETITION_GUARD_REASONING", "off")
    assert StreamingRepetitionGuard.reasoning_from_env() is None
    monkeypatch.delenv("HERMES_STREAM_REPETITION_GUARD_REASONING")
    assert StreamingRepetitionGuard.reasoning_from_env() is not None
    # Main switch also disables the reasoning guard.
    monkeypatch.setenv("HERMES_STREAM_REPETITION_GUARD", "0")
    assert StreamingRepetitionGuard.reasoning_from_env() is None


def test_truncate_at_repetition_cuts_at_second_occurrence():
    exc = find_repeated_line_from_env(INCIDENT_LINE * 20)
    assert exc is not None
    text = "Preambule sain qui doit survivre a la coupe.\n" + INCIDENT_LINE * 20
    cut = truncate_at_repetition(text, exc.repeated_line)

    assert cut.count("TEXTURE_LOCAL.md") == 1  # first occurrence kept
    assert "Preambule sain" in cut
    assert cut.endswith(REPETITION_CUT_MARKER)
    assert len(cut) < len(text)


def test_truncate_at_repetition_single_occurrence_untouched():
    # Loop was in another channel (e.g. reasoning): content has the line once.
    text = "Debut.\n" + INCIDENT_LINE + "Fin propre.\n"
    exc = find_repeated_line_from_env(INCIDENT_LINE * 20)
    assert truncate_at_repetition(text, exc.repeated_line) == text
