"""Give-up notifications clip on a word boundary; the ``gave_up`` event names
how the worker stopped in ``exit_kind``, the key the ``crashed`` event uses.

The give-up message renders the last error, which usually ends with the
worker's own last output. A hard slice stopped mid-word ("...report it via
kanban_co"), which reads as a corrupted message rather than a truncated one.
"""

import json
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers_notifier import _clip_words, _fmt_gave_up
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _ev(**payload):
    return SimpleNamespace(payload=payload)


def _clip(text, limit):
    return _clip_words(_ev(error=text), "error", "{}", limit)


# --- word-boundary clip -----------------------------------------------------


def test_clip_ends_on_a_word_boundary_with_an_ellipsis():
    text = "the quick brown fox jumps over the lazy dog and keeps on running"
    out = _clip(text, 30)
    assert out == "the quick brown fox jumps over…"
    assert text.startswith(out[:-1])


def test_clip_leaves_short_text_alone():
    assert _clip("already short", 160) == "already short"


def test_clip_collapses_whitespace():
    assert _clip("a\n\n  b", 160) == "a b"


def test_clip_strips_trailing_punctuation_before_the_ellipsis():
    assert _clip("first clause, second clause goes on", 16) == "first clause…"


def test_clip_falls_back_to_a_hard_cut_for_one_long_token():
    token = "/very/long/path/" + "x" * 80
    out = _clip("see " + token, 40)
    assert out == ("see " + token)[:40] + "…"


def test_clip_absent_value_renders_nothing():
    assert _clip_words(_ev(), "error", " (last: {})", 160) == ""


def test_gave_up_message_does_not_cut_mid_word():
    error = (
        "worker exited cleanly (rc=0) without kanban_complete, kanban_block "
        "or kanban_request_review — protocol violation. If the prior run "
        "already did the work, verify it and report it via kanban_complete"
    )
    n = SimpleNamespace(head="Kanban t_1", task_id="t_1")
    msg, _, _ = _fmt_gave_up(_ev(failures=3, error=error), n)
    last = msg.split(" (last: ", 1)[1].split("). Fix the cause", 1)[0]
    assert last.endswith("…")
    body = last[:-1]
    # The clipped text is a whole-word prefix of the original error.
    assert error.startswith(body)
    assert error[len(body)] == " "


# --- exit_kind on the gave_up event -----------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kb.init_db()
    return home


def _gave_up_payload(exit_code: int) -> dict:
    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="t", assignee="a", max_retries=1)
        kb.claim_task(conn, tid, claimer=f"{host}:w0")
        pid = 72000 + exit_code
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
        conn.commit()
        kbd._record_worker_exit(pid, exit_code << 8)
        kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, tid).status == "blocked"
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'", (tid,),
        ).fetchone()
        return json.loads(row["payload"])


@pytest.mark.parametrize(
    "exit_code, expected",
    [
        (0, "clean_exit"),
        (1, "nonzero_exit"),
        (kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE, "terminal_provider"),
    ],
)
def test_gave_up_event_carries_exit_kind(kanban_home, exit_code, expected):
    assert _gave_up_payload(exit_code)["exit_kind"] == expected
