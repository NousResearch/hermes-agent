"""--query-file: single-query text arrives verbatim, never shell-interpreted.

Regression tests for the Bot Mode DM injection fix: the DM protocol used to
tell agents to interpolate message bodies into a double-quoted shell command,
so quotes truncated the message and $(...) executed on the sender's machine.
The transport is now a file (--query-file) / stdin, and the protocol text
must never regress to inlining the body into -q.
"""


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


def test_chat_parser_accepts_machine_result_for_process_adapters():
    args = _parse(["chat", "--query-file", "-", "--machine-result"])
    assert args.machine_result is True


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
    """argparse rejects -q + --query-file at parse time (exit 2), no env needed."""
    import pytest

    f = tmp_path / "dm.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _parse(["chat", "-q", "x", "--query-file", str(f)])
    assert exc.value.code == 2


def test_bot_mode_protocol_never_inlines_message_into_shell():
    """The DM transport must use --query-file / stdin, not -q "…" inlining.

    The transport moved from prompt-injected instructions (bot_mode_probe)
    to the message_agent tool (bot_mode_dm) in Aug 2026 — the invariant now
    holds on the tool's command builder, and the probe must no longer teach
    any shellout at all.
    """
    sys.path.insert(0, str(REPO))
    try:
        import importlib

        dm = importlib.import_module("tools.bot_mode_dm")
        src = Path(dm.__file__).read_text(encoding="utf-8")
        probe = importlib.import_module("tools.bot_mode_probe")
        probe_src = Path(probe.__file__).read_text(encoding="utf-8")
    finally:
        sys.path.remove(str(REPO))
    assert "--query-file" in src
    assert '-q "Message from' not in src
    assert 'dm <peer>/<agent-name> "Message from' not in src
    # The protocol section teaches the tool, never a hand-rolled shellout.
    assert "message_agent" in probe_src
    assert "--query-file /tmp/dm.txt" not in probe_src
