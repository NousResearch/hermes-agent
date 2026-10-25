"""The `hermes sessions recover --source <db>` remedy must survive being PASTED.

`hermes_state_repair` prints a salvage command at three sites on the
DB-corruption recovery path — the exhausted-repair-budget diagnostic
(`_persistent_repair_exhausted_error`) and both `_backup_free_space_error`
forensic-backup refusals, which `_backup_db_file` returns to the operator.

Each interpolated the DB path bare into a backticked span the operator is told
to paste, so a `HERMES_HOME` holding a space (Google Drive's ``My Drive``, or
the Windows ``C:/Users/<First Last>`` default) printed a remedy that split into
two words in the SHELL before argparse ever saw it::

    printed  : hermes sessions recover --source /Users/x/My Drive/hermes/state.db --inspect-only
    bash argv: ['hermes','sessions','recover','--source','/Users/x/My',
                'Drive/hermes/state.db','--inspect-only']
    argparse : --source binds '/Users/x/My'

An unreachable remedy on the one path where the operator has least slack: the
database is already beyond automatic repair.

THE PASTE SIMULATOR IS A REAL SHELL AND THE PARSER IS THE REAL PARSER. The
strings are not reconstructed here: the real message builders are called, the
backticked span is extracted from what they actually returned, the words come
from ``/bin/bash``, and those words are fed to the real ``hermes sessions
recover`` argparse subparser. Asserting on the string's shape — or simulating
the paste with ``shlex`` while the implementation decides safety with ``shlex``
— would be a tautology, not a test.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_state_repair
from hermes_cli.cli_hint import hint_arg
from hermes_cli.subcommands.sessions import build_sessions_parser

BASH = shutil.which("bash")

requires_bash = pytest.mark.skipif(
    BASH is None or sys.platform.startswith("win"),
    reason="the paste oracle needs a real POSIX shell",
)

# Path shapes an operator really has. "My Drive" is how Google Drive mounts on
# macOS; the Windows default lives under C:/Users/<First Last>.
HOSTILE_DIRS = ["My Drive", "Program Files", "First Last"]
ORDINARY_DIRS = ["hermes", "dot-hermes"]


def _bash_words(printed: str, cwd: str):
    """The argv a REAL bash produces for `printed`, or None if bash refuses.

    NUL-delimited so a word containing whitespace survives the round trip.
    """
    proc = subprocess.run(
        [str(BASH), "-c", 'printf "%s\\0" ' + printed],
        capture_output=True,
        cwd=cwd,
    )
    if proc.returncode != 0:
        return None
    out = proc.stdout.decode("utf-8", errors="replace")
    return out.split("\0")[:-1] if out else []


def _recover_spans(message: str):
    """Every backticked `hermes ... sessions recover ...` span in `message`."""
    return [s for s in re.findall(r"`([^`]*)`", message) if "sessions recover" in s]


def _recover_parser() -> argparse.ArgumentParser:
    """The REAL `hermes sessions recover` parser from the shipped registration."""
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_sessions_parser(subparsers, cmd_sessions=lambda _args: 0)
    return parser


def _damaged_db(home: Path) -> Path:
    db = home / "state.db"
    db.write_bytes(b"not a database")
    return db


def _messages(db: Path):
    """The three operator-facing strings, from the REAL builders.

    ``_backup_free_space_error`` is driven through its two real refusal
    branches (no headroom, and a volume it could not stat) by making
    ``_disk_budget`` report each — the refusal text is what ``_backup_db_file``
    hands back to the operator.
    """
    out = {"exhausted": hermes_state_repair._persistent_repair_exhausted_error(db)}
    with patch.object(hermes_state_repair, "_disk_budget",
                      return_value=(None, 10 ** 12, 1, 10 ** 9)):
        out["low_disk"] = hermes_state_repair._backup_free_space_error(db)
    with patch.object(hermes_state_repair, "_disk_budget",
                      return_value=("could not determine free space", 0, 0, 0)):
        out["unstattable"] = hermes_state_repair._backup_free_space_error(db)
    return out


@requires_bash
@pytest.mark.parametrize("hostile_dir", HOSTILE_DIRS)
@pytest.mark.parametrize("site", ["exhausted", "low_disk", "unstattable"])
def test_printed_remedy_survives_a_real_paste(tmp_path, monkeypatch, site, hostile_dir):
    """Pasted into bash, the printed span must still carry the path as ONE argv
    word, and the real parser must bind that exact path to ``--source``."""
    home = tmp_path / hostile_dir / "hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = _damaged_db(home)

    message = _messages(db)[site]
    spans = _recover_spans(message)
    assert spans, f"{site} no longer prints a `sessions recover` remedy: {message}"

    parser = _recover_parser()
    for span in spans:
        words = _bash_words(span, str(home))
        assert words is not None, f"bash refused the printed remedy outright: {span}"
        assert words[:3] == ["hermes", "sessions", "recover"], words

        try:
            args = parser.parse_args(words[1:])
        except SystemExit as exc:  # argparse rejected the pasted command outright
            pytest.fail(
                f"the real parser REFUSED the pasted remedy (exit {exc.code})\n"
                f"  printed: {span}\n  argv: {words}"
            )
        assert args.source == Path(db), (
            f"pasted remedy bound --source to {args.source!r}, not the real "
            f"database {str(db)!r}\n  printed: {span}\n  argv: {words}"
        )


@requires_bash
@pytest.mark.parametrize("ordinary_dir", ORDINARY_DIRS)
def test_ordinary_paths_keep_the_plain_unquoted_spelling(tmp_path, monkeypatch, ordinary_dir):
    """The over-fix guard: a path needing no escaping must NOT gain quotes.

    Blanket-quoting every path would also 'pass' the test above while making
    every ordinary message harder to read, so pin the readable spelling.
    """
    home = tmp_path / ordinary_dir
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = _damaged_db(home)
    assert " " not in str(db), "this control needs a path that needs no escaping"

    for site, message in _messages(db).items():
        for span in _recover_spans(message):
            assert f"--source {db}" in span, f"{site} quoted an ordinary path: {span}"
            words = _bash_words(span, str(home))
            assert words is not None and str(db) in words, (site, span, words)


def test_source_arg_uses_the_attached_form_for_a_leading_dash():
    """`--source` takes a VALUE, so a value starting with '-' needs the
    attached ``--source=<value>`` spelling: argparse binds a bare following
    token as an option and refuses with 'expected one argument'."""
    rendered = hermes_state_repair._source_arg(Path("-weird/state.db"))
    assert "--source=-weird/state.db" in rendered, rendered

    parser = _recover_parser()
    words = _bash_words("hermes sessions recover " + rendered + " --inspect-only", os.getcwd())
    assert words is not None
    args = parser.parse_args(words[1:])
    assert args.source == Path("-weird/state.db")
    assert args.inspect_only is True


def test_source_arg_falls_back_without_hermes_cli(monkeypatch):
    """Scaffold/embed installs without ``hermes_cli`` must still print a
    pasteable span — the fallback is the always-quoted attached form."""
    real_import = __import__

    def _no_cli_hint(name, *a, **kw):
        if name == "hermes_cli.cli_hint":
            raise ImportError("no hermes_cli in this install")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _no_cli_hint)
    rendered = hermes_state_repair._source_arg(Path("/tmp/My Drive/state.db"))
    assert rendered == "'--source=/tmp/My Drive/state.db'"


def test_hint_arg_is_the_choke_point():
    """`_source_arg` must delegate, not re-derive the escaping rules."""
    for raw in ["/plain/state.db", "/My Drive/state.db", "-dash/state.db",
                "/glob*/state.db", "/x$HOME/state.db"]:
        assert hermes_state_repair._source_arg(Path(raw)) == hint_arg("--source", raw)
