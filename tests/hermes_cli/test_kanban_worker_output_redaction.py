"""A dead worker's board diagnostic keeps the reason, not internal handles or echo."""
import run_agent  # Activate bootstrap before the per-test home-I/O guard.
from hermes_cli import kanban_db_dispatch as dispatch

SID_A = 'local-' + 'a' * 64
SID_B = 'local-' + 'b' * 64
REFUSAL = 'Kanban owner refused: Kanban execution unknown; retained by owner'


def _output(monkeypatch, raw):
    monkeypatch.setattr(dispatch._kb, 'read_worker_log', lambda *a, **k: raw)
    return dispatch._worker_final_output('t_synthetic', board='default')


def test_session_handles_never_reach_the_board_diagnostic(monkeypatch):
    raw = f'Session: {SID_A} {REFUSAL}\nSession: {SID_B} {REFUSAL}\n'
    out = _output(monkeypatch, raw)
    assert SID_A not in out and SID_B not in out
    assert 'local-<redacted>' in out
    assert REFUSAL in out


def test_repeated_refusals_collapse_to_one_reason_with_a_count(monkeypatch):
    raw = ''.join(f'Session: {SID_A} {REFUSAL}\n' for _ in range(3))
    out = _output(monkeypatch, raw)
    assert out.count(REFUSAL) == 1
    assert f'{REFUSAL} (x3)' in out


def test_distinct_lines_are_all_kept_in_order(monkeypatch):
    raw = f'Session: {SID_A} first reason\nSession: {SID_B} second reason\n'
    out = _output(monkeypatch, raw)
    assert 'first reason' in out and 'second reason' in out
    assert out.index('first reason') < out.index('second reason')
    assert '(x' not in out


def test_a_short_hash_is_not_treated_as_a_session_handle(monkeypatch):
    # Only the 32-64 hex handle the CLI prints is redacted; ordinary hex text survives.
    raw = 'commit local-deadbeef failed\n'
    assert _output(monkeypatch, raw) == 'commit local-deadbeef failed'


def test_the_exit_trailer_is_trimmed_but_still_read_for_the_exit_code(monkeypatch):
    # One log, two readers: the trailer the worker writes must leave the board diagnostic and
    # still decide the exit classification. Both live behind the same late-bound log reader.
    raw = (
        'session_id: 0000\r\n'
        f'Session: {SID_A} {REFUSAL}\r\n'
        '\u2502  the model explanation  \u2502\r\n'
        f'[kanban-worker-exit] rc={2}\r\n'
    )
    monkeypatch.setattr(dispatch._kb, 'read_worker_log', lambda *a, **k: raw)
    out = dispatch._worker_final_output('t_synthetic', board='default')
    assert 'kanban-worker-exit' not in out and 'session_id:' not in out
    assert '\u2502' not in out and 'the model explanation' in out and REFUSAL in out
    assert dispatch._worker_log_exit_code('t_synthetic', board='default') == 2
