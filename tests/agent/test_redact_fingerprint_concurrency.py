"""Concurrent first-use of the fingerprint key must not split one install's output.

The winner of the O_EXCL create sits between creating ``secret_fingerprint.key`` and
writing its 64-hex payload for a short window; a loser reading inside that window used to
degrade to the generic ``***`` while the winner printed ``***#xxxxxx`` — two shapes for the
same secret on one install (the PR #145787 gap). These tests hold that window open
deterministically (spawn+Event gate, mirroring tests/hermes_cli/test_install_identity.py)
and require every concurrent first-use party to land on the SAME fingerprint-shaped
output, while the fail-closed faces (readonly HOME, malformed key) stay exactly ``***``.
"""

import multiprocessing
import os
import re
import stat
import threading
import time

import pytest

from agent import redact as _redact_mod
from tools.spill_safety import open_exclusive

_FP_FORM = re.compile(r"\*\*\*#[0-9a-f]{6}")
_SECRET = "PasswordPassword"
_PAYLOAD = "0123456789abcdef" * 4  # a fixed, valid 32-byte hex key the winner publishes


@pytest.fixture
def fp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    getattr(_redact_mod, "_fingerprint_keys", {}).clear()
    yield tmp_path
    getattr(_redact_mod, "_fingerprint_keys", {}).clear()


def _key_path():
    return _redact_mod._fingerprint_key_path()


def _assert_fingerprint_shape(out):
    assert _FP_FORM.fullmatch(out), out  # fingerprint form, never the bare generic mask
    assert _SECRET not in out  # and never a character of the secret


# ── thread race: the loser must wait out the winner's create→payload window ──

def test_threads_concurrent_first_use_agree_on_one_fingerprint(fp_home):
    writer_entered = threading.Event()
    release_writer = threading.Event()
    outputs = {}

    def winner():
        fh = open_exclusive(_key_path())  # key file exists now; payload still missing
        try:
            writer_entered.set()
            assert release_writer.wait(timeout=30)
            fh.write(_PAYLOAD + "\n")
        finally:
            fh.close()
        outputs["winner"] = _redact_mod.mask_secret(_SECRET)

    def loser():
        assert writer_entered.wait(timeout=30)
        outputs["loser"] = _redact_mod.mask_secret(_SECRET)

    wt = threading.Thread(target=winner)
    lt = threading.Thread(target=loser)
    wt.start()
    assert writer_entered.wait(timeout=10)
    lt.start()
    time.sleep(0.05)  # health: allow HX013 -- this held window IS the race under test: the loser must already sit inside its bounded re-read of the payload-less file when the winner publishes
    release_writer.set()
    wt.join(timeout=30)
    lt.join(timeout=30)
    assert not wt.is_alive() and not lt.is_alive()  # bounded: no party hung
    _assert_fingerprint_shape(outputs["winner"])
    assert outputs["loser"] == outputs["winner"]  # one install, one shape

    # the raced key file is sound and a cold caller repeats the same fingerprint
    key = fp_home / "secret_fingerprint.key"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert re.fullmatch(r"[0-9a-f]{64}", key.read_text().strip())
    _redact_mod._fingerprint_keys.clear()
    assert _redact_mod.mask_secret(_SECRET) == outputs["winner"]


# ── independent-process race: the same contract across process boundaries ──

def _process_winner(home, writer_entered, release_writer, results):
    os.environ["HERMES_HOME"] = home
    fh = open_exclusive(_key_path())
    try:
        writer_entered.set()
        assert release_writer.wait(timeout=60)
        fh.write(_PAYLOAD + "\n")
    finally:
        fh.close()
    results.put(_redact_mod.mask_secret(_SECRET))


def _process_loser(home, loser_entered, results):
    os.environ["HERMES_HOME"] = home
    loser_entered.set()
    results.put(_redact_mod.mask_secret(_SECRET))


def test_processes_concurrent_first_use_agree_on_one_fingerprint(fp_home):
    ctx = multiprocessing.get_context("spawn")
    writer_entered = ctx.Event()
    release_writer = ctx.Event()
    loser_entered = ctx.Event()
    winner_out = ctx.Queue()
    loser_out = ctx.Queue()

    winner = ctx.Process(
        target=_process_winner,
        args=(str(fp_home), writer_entered, release_writer, winner_out),
    )
    loser = ctx.Process(
        target=_process_loser,
        args=(str(fp_home), loser_entered, loser_out),
    )
    winner.start()
    assert writer_entered.wait(timeout=60)
    loser.start()
    assert loser_entered.wait(timeout=60)
    time.sleep(0.05)  # health: allow HX013 -- this held window IS the race under test: the loser process must already sit inside its bounded re-read of the empty file when the winner publishes
    release_writer.set()
    winner.join(timeout=60)
    loser.join(timeout=60)
    assert winner.exitcode == 0 and loser.exitcode == 0  # bounded: no party hung or crashed
    out_w = winner_out.get(timeout=10)
    out_l = loser_out.get(timeout=10)
    _assert_fingerprint_shape(out_w)
    assert out_l == out_w  # two processes, one install, one shape

    # the raced key file is sound on disk and a cold caller repeats the same fingerprint
    key = fp_home / "secret_fingerprint.key"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert re.fullmatch(r"[0-9a-f]{64}", key.read_text().strip())
    _redact_mod._fingerprint_keys.clear()
    assert _redact_mod.mask_secret(_SECRET) == out_w


# ── fail-closed faces are untouched by the concurrency wait ──

def test_readonly_home_fails_closed_without_creating_key(tmp_path, monkeypatch):
    home = tmp_path / "readonly-home"
    home.mkdir()
    home.chmod(0o500)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _redact_mod._fingerprint_keys.clear()
    try:
        out = _redact_mod.mask_secret(_SECRET)  # must not raise
        assert out == "***"
        assert _SECRET not in out
        assert not (home / "secret_fingerprint.key").exists()
    finally:
        _redact_mod._fingerprint_keys.clear()
        home.chmod(0o700)  # let pytest clean tmp_path up


@pytest.mark.parametrize("content", ["", "not-hex-at-all", "zz" * 32, "ab" * 31])
def test_preexisting_malformed_key_fails_closed(fp_home, content):
    key = fp_home / "secret_fingerprint.key"
    key.write_text(content)
    key.chmod(0o600)
    out = _redact_mod.mask_secret(_SECRET)  # must not raise, must not leak, stays generic
    assert out == "***"
    assert _SECRET not in out
