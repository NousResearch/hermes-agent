"""The neutral helper modules: key parsing, error classification and user copy, transcript limits, holder identity, transcript rescue and the withheld-prompt helpers of ``session_rules.turns``.

The text rules and values are pinned, and so are two shapes: a contract error answers by its type before any text rule, the rescue file goes under the directory the caller passes.
"""

from __future__ import annotations

import errno
import json
import os
import sqlite3
import subprocess
import sys

import pytest

import process_identity
import storage_limits
import storage_user_copy
from persistence.core.errors import (
    CorruptStoreError,
    DeletedWalGeneration,
    DiskFull,
    NotFound,
    SearchIndexCorrupt,
    SessionRowMissing,
    StoreContention,
    StoreReplaced,
)
from persistence.core.keys import session_key_profile
from session_keys import profile_from_session_key, profile_from_session_key_namespace
from session_rescue import divert_session_transcript_jsonl
from storage_errors import classify_persistence_error, is_disk_full_error, is_lock_contention, is_malformed_db_error


@pytest.mark.parametrize(("key", "keyed", "namespaced"), [
    ("agent:main:telegram:dm:1", "default", "default"),
    ("agent:main~:telegram:dm:1", "main", "main"),
    ("agent:work:telegram:dm:1", "work", "work"),
    ("agent:work", None, "work"),
    ("agent::x", None, "default"),
    ("hosted:room:1", None, None),
    ("", None, None),
    (None, None, None),
])
def test_a_session_key_names_its_profile(key, keyed, namespaced):
    assert session_key_profile(key) == keyed
    assert profile_from_session_key(key) == namespaced


def test_the_key_namespace_inverse():
    assert [profile_from_session_key_namespace(ns) for ns in ("main", "main~", "work")] == ["default", "main", "work"]


@pytest.mark.parametrize(("error", "cause"), [
    (StoreContention("held"), "locked"),
    (DiskFull("no room"), "disk"),
    (CorruptStoreError("damaged"), "corrupt"),
    (SearchIndexCorrupt("index"), "fts_index"),
    (StoreReplaced("gone"), "replaced"),
    (DeletedWalGeneration("old wal"), "deleted_wal"),
    (SessionRowMissing("s1"), "session_row_missing"),
    # A contract error declaring no cause is classified by its text.
    (NotFound("session", "database is locked"), "locked"),
])
def test_a_contract_error_answers_by_its_type_first(error, cause):
    assert classify_persistence_error(error) == cause


def test_a_declared_cause_outranks_the_text():
    # The text alone would read as lock contention.
    assert classify_persistence_error(CorruptStoreError("database is busy")) == "corrupt"
    assert is_lock_contention(CorruptStoreError("database is busy")) is False
    assert is_lock_contention(StoreContention("anything")) is True
    assert is_disk_full_error(DiskFull("anything")) is True


@pytest.mark.parametrize(("error", "cause"), [
    (None, "unknown"),
    (sqlite3.OperationalError("database is locked"), "locked"),
    ("database is busy", "locked"),
    (sqlite3.DatabaseError("database disk image is malformed"), "corrupt"),
    ("fts5: corruption found reading messages_fts_data", "fts_index"),
    ("FOREIGN KEY constraint failed", "session_row_missing"),
    ("session is being compressed", "compression"),
    ("Session 's1' is closed by compression", "compression_closed"),
    ("turn lease lost", "turn_lease"),
    ("holds a deleted state.db-wal inode", "deleted_wal"),
    ("state.db was replaced underneath the gateway", "replaced"),
    (OSError(errno.ENOSPC, "No space left on device"), "disk"),
    ("attempt to write a readonly database", "disk"),
    ("something else", "unknown"),
])
def test_other_errors_keep_the_text_rules(error, cause):
    assert classify_persistence_error(error) == cause


def test_lock_and_disk_full_predicates_on_raw_errors():
    assert is_lock_contention(sqlite3.OperationalError("database table is locked")) is True
    assert is_lock_contention("no such table: sessions") is False
    assert is_disk_full_error(OSError(errno.ENOSPC, "x")) is True
    assert is_disk_full_error("database or disk is full") is True
    assert is_disk_full_error(None) is False


def test_the_malformed_store_predicate():
    assert is_malformed_db_error(sqlite3.DatabaseError("database disk image is malformed")) is True
    assert is_malformed_db_error(sqlite3.OperationalError("malformed database schema (x)")) is True
    assert is_malformed_db_error(sqlite3.OperationalError("database is locked")) is False
    assert is_malformed_db_error(ValueError("database disk image is malformed")) is False


def test_the_withheld_prompt_texts_are_the_applications():
    from agent import message_metadata
    from session_rules import turns

    # The store and the application read and restore a vetoed prompt by the same text.
    assert (turns.WITHHELD_PROMPT, turns.PRE_LLM_BLOCK_WITHHELD_PROMPT) == (
        message_metadata.WITHHELD_PROMPT, message_metadata.PRE_LLM_BLOCK_WITHHELD_PROMPT)


def test_storage_failure_copy_splits_a_full_disk_from_a_read_only_file():
    assert storage_user_copy.describe_storage_failure(DiskFull("no room")).code == "disk_full"
    assert storage_user_copy.describe_storage_failure("attempt to write a readonly database").code == "storage_readonly"
    assert storage_user_copy.describe_storage_failure(StoreContention("held")).code == "storage_locked"
    assert storage_user_copy.storage_failure_details("a  b\n" + "c" * 300, limit=10) == "a b ccc..."


_UNSAVED = "Sessions will not be saved until this is fixed."
_FIX = "Run `hermes doctor --fix` to diagnose and repair."
_LONG_CAUSE = "RuntimeError: something else\nsecond line " + "y" * 400


# Each text is the unavailable notice for "P" after the last open failure is set to ``cause``.
@pytest.mark.parametrize(("cause", "details", "text"), [
    (None, False, f"P. {_UNSAVED} Run `hermes doctor` to check the storage location."),
    (None, True, f"P. {_UNSAVED} Run `hermes doctor` to check the storage location."),
    ("OperationalError: database is locked", False,
     f"P: the session database is locked by another Hermes process. {_UNSAVED} Wait a moment and try again; if it persists, "
     "stop the other Hermes process (`hermes gateway stop`)."),
    ("OperationalError: database is locked", True,
     f"P: the session database is locked by another Hermes process. {_UNSAVED} Wait a moment and try again; if it persists, "
     "stop the other Hermes process (`hermes gateway stop`).\nDetails: OperationalError: database is locked"),
    ("sqlite3.OperationalError: locking protocol", True,
     f"P: the session database could not be opened; it may be on a network or unsupported drive. {_UNSAVED} Move it to a "
     "local disk (`hermes doctor` shows where it is), then start Hermes again.\nDetails: sqlite3.OperationalError: locking protocol"),
    ("OperationalError: disk I/O error", False,
     f"P: the session database file is read-only or not writable. {_UNSAVED} {_FIX} If the database lives on a network drive, "
     "move it to a local disk."),
    ("OperationalError: attempt to write a readonly database", False,
     f"P: the session database file is read-only or not writable. {_UNSAVED} {_FIX}"),
    ("DatabaseError: database disk image is malformed", False,
     f"P: the session database file is damaged. {_UNSAVED} {_FIX} Recovery: `hermes sessions recover --source <state.db> --inspect-only`."),
    ("OperationalError: database or disk is full", False, f"P: the disk is full. {_UNSAVED} Free some disk space, then try again."),
    ("OSError: [Errno 28] No space left on device", True,
     f"P: the disk is full. {_UNSAVED} Free some disk space, then try again.\nDetails: OSError: [Errno 28] No space left on device"),
    ("PermissionError: [Errno 13] Permission denied: '/x/state.db'", True,
     f"P: the session database could not be opened. {_UNSAVED} {_FIX}\nDetails: PermissionError: [Errno 13] Permission denied: '/x/state.db'"),
    (_LONG_CAUSE, True, f"P: the session database could not be opened. {_UNSAVED} {_FIX}\nDetails: RuntimeError: something else second line "
                        + "y" * 156 + "..."),
])
def test_the_unavailable_notice_names_the_last_open_failure(monkeypatch, cause, details, text):
    monkeypatch.setattr(storage_user_copy, "_last_init_error", None)
    storage_user_copy.set_last_init_error(cause)
    assert storage_user_copy.get_last_init_error() == cause
    assert storage_user_copy.format_session_db_unavailable("P", details=details) == text


def test_the_last_open_failure_is_kept_until_the_next_one(monkeypatch):
    monkeypatch.setattr(storage_user_copy, "_last_init_error", None)
    assert storage_user_copy.get_last_init_error() is None
    storage_user_copy.set_last_init_error("OperationalError: database is locked")
    storage_user_copy.set_last_init_error("OperationalError: disk I/O error")
    assert storage_user_copy.get_last_init_error() == "OperationalError: disk I/O error"
    storage_user_copy.set_last_init_error(None)
    assert storage_user_copy.get_last_init_error() is None


@pytest.mark.parametrize(("configured", "limit"), [(None, 20_000), (5, 5), (0, 0), (-1, 20_000), ("x", 20_000)])
def test_transcript_limits_come_from_the_sessions_config(monkeypatch, configured, limit):
    import hermes_cli.config

    sessions = {} if configured is None else {"max_export_messages": configured, "max_resume_messages": configured}
    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: {"sessions": sessions})
    assert storage_limits.resolved_max_export_messages() == limit
    assert storage_limits.resolved_max_resume_messages() == limit


def test_a_holder_is_dead_only_on_kernel_proof():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    stamped = f"pid={child.pid}{process_identity.holder_namespace_token()}:turn=t"
    assert process_identity.holder_process_is_dead(stamped) is True
    assert process_identity.holder_process_is_dead(f"pid={os.getpid()}{process_identity.holder_namespace_token()}") is False
    assert process_identity.holder_process_is_dead("") is False
    assert process_identity.holder_process_is_dead("pid=0") is False
    if process_identity._local_pid_namespace().supported:
        # Where PID namespaces exist, an unstamped or foreign holder is never probed.
        assert process_identity.holder_process_is_dead(f"pid={child.pid}:turn=t") is False
        assert process_identity.holder_process_is_dead(f"pid={child.pid}:pidns=1:turn=t") is False


# Each interpreter option that takes an operand, as the next token or attached to the short option. An attached
# operand holds a ``c`` and a separate one is a bare word, so a reader that does not know the option reads the operand
# as more options (and finds ``-c``) or as the program.
OPERAND_TAKING_OPTIONS = [
    ["-Q", "new"],
    ["-W", "ignore::DeprecationWarning"],
    ["-Werror::DeprecationWarning"],
    ["-X", "utf8"],
    ["-Xtracemalloc=5"],
    ["-Xc"],
    ["--check-hash-based-pycs", "always"],
    ["--jit", "on"],
]


@pytest.mark.parametrize("options", OPERAND_TAKING_OPTIONS)
def test_an_operand_taking_option_is_read_past_to_the_program(options):
    from gateway.status_inline_source import inline_source_flag_index

    inline = ["python", *options, "-c", "import os", "14980", "hermes", "gateway", "run"]
    assert inline_source_flag_index(inline) == len(options) + 1
    assert inline_source_flag_index(["python", *options, "-m", "hermes_cli.main", "gateway", "run"]) is None


def test_a_rescued_transcript_appends_under_the_callers_directory(tmp_path):
    sessions_dir = tmp_path / "rescue"
    assert divert_session_transcript_jsonl(" ", [{"role": "user"}], sessions_dir=sessions_dir) is None
    assert divert_session_transcript_jsonl("s1", [], sessions_dir=sessions_dir) is None
    assert not sessions_dir.exists()
    path = divert_session_transcript_jsonl(" s1 ", [{"role": "user", "content": "é"}, None, "plain"], sessions_dir=sessions_dir)
    assert path == sessions_dir / "s1.jsonl"
    divert_session_transcript_jsonl("s1", [{"role": "assistant"}], sessions_dir=sessions_dir)
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        {"role": "user", "content": "é"}, {"content": "plain"}, {"role": "assistant"}]
