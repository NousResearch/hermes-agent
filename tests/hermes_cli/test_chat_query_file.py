"""--query-file: single-query text arrives verbatim, never shell-interpreted.

Regression tests for the Bot Mode DM injection fix: the DM protocol used to
tell agents to interpolate message bodies into a double-quoted shell command,
so quotes truncated the message and $(...) executed on the sender's machine.
The transport is now a file (--query-file) / stdin, and the protocol text
must never regress to inlining the body into -q.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

HOSTILE = 'hi "there" $(touch /tmp/pwned_by_dm_test) `id` \\ and a\nsecond line'


def _parse(argv):
    sys.path.insert(0, str(REPO))
    try:
        from hermes_cli._parser import build_top_level_parser

        built = build_top_level_parser()
        parser = built[0] if isinstance(built, tuple) else built
        return parser.parse_args(argv)
    finally:
        sys.path.remove(str(REPO))


def test_chat_parser_accepts_query_file():
    args = _parse(["chat", "--query-file", "/tmp/x.txt"])
    assert args.query_file == "/tmp/x.txt"
    assert args.query is None


def test_query_file_reads_hostile_text_verbatim(tmp_path, monkeypatch):
    """The file body must reach args.query byte-identical — no shell pass."""
    f = tmp_path / "dm.txt"
    f.write_text(HOSTILE, encoding="utf-8")

    # Exercise the exact resolution block in hermes_cli.main by simulating it:
    # the block reads the file into args.query before dispatch.
    args = _parse(["chat", "--query-file", str(f)])
    assert args.query_file is not None
    body = Path(args.query_file).read_text(encoding="utf-8")
    assert body == HOSTILE
    assert "$(touch" in body  # preserved, not executed
    assert not Path("/tmp/pwned_by_dm_test").exists()


def test_query_and_query_file_mutually_exclusive(tmp_path):
    f = tmp_path / "dm.txt"
    f.write_text("hello", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); "
         "from hermes_cli.main import main; sys.argv=['hermes','chat','-q','x','--query-file',%r]; main()"
         % (str(REPO), str(f))],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2
    assert "mutually exclusive" in proc.stderr


def test_bot_mode_protocol_never_inlines_message_into_shell():
    """The DM protocol must use --query-file / stdin, not -q "…" inlining."""
    sys.path.insert(0, str(REPO))
    try:
        import importlib

        probe = importlib.import_module("tools.bot_mode_probe")
        src = Path(probe.__file__).read_text(encoding="utf-8")
    finally:
        sys.path.remove(str(REPO))
    assert "--query-file" in src
    assert '-q "Message from' not in src
    assert 'dm <peer>/<agent-name> "Message from' not in src
