"""``hermes -r <session> <message>``: the words after a real session target become the first turn.

Driven through ``_resolve_chat_session_args`` against a real ``SessionDB`` in the sandboxed
HERMES_HOME, with argv joined by the real ``_coalesce_session_name_args`` + parser.
"""

from __future__ import annotations

import pytest

from hermes_cli._parser import build_top_level_parser
from hermes_cli.main import _coalesce_session_name_args, _resolve_chat_session_args
from hermes_state import SessionDB

SID = "20261007_191147_38b910"


@pytest.fixture
def sessions():
    db = SessionDB()
    db.create_session(SID, source="cli")
    db.create_session("20261007_191200_aaaaaa", source="cli")
    db.set_session_title("20261007_191200_aaaaaa", "Pokemon Agent Dev")
    db.close()


def _resolve(argv):
    args = build_top_level_parser()[0].parse_args(_coalesce_session_name_args(argv))
    args.query = getattr(args, "query", None)
    args.no_restore_cwd = True
    _resolve_chat_session_args(args, use_tui=False)
    return args.resume, args.query


@pytest.mark.parametrize("argv, expected", [
    (["-r", SID, "what", "changed?"], (SID, "what changed?")),
    (["chat", "-r", SID, "fix", "the", "bug"], (SID, "fix the bug")),
    (["-c", "Pokemon", "Agent", "Dev", "add", "tests"], ("20261007_191200_aaaaaa", "add tests")),
    (["-r", "Pokemon", "Agent", "Dev", "go"], ("20261007_191200_aaaaaa", "go")),
])
def test_trailing_words_after_a_real_session_become_the_query(sessions, argv, expected):
    assert _resolve(argv) == expected


@pytest.mark.parametrize("argv, expected", [
    # A name that resolves whole keeps resuming with no message (multi-word titles).
    (["-c", "Pokemon", "Agent", "Dev"], ("20261007_191200_aaaaaa", None)),
    (["-r", SID], (SID, None)),
    # No prefix names a session: the joined value is kept so the CLI reports "Session not found".
    (["-r", "20991231_000000_ffffff", "hello", "there"], ("20991231_000000_ffffff hello there", None)),
    # An explicit -q wins; the trailing words stay part of the target as before.
    (["chat", "-q", "hi", "-r", SID, "extra"], (f"{SID} extra", "hi")),
])
def test_names_that_resolved_before_are_unchanged(sessions, argv, expected):
    assert _resolve(argv) == expected
