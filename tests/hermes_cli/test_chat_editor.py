"""`hermes chat --editor`: compose a multi-line prompt in $EDITOR, send as ONE query.

Issue #121123: pasting multi-line text (logs, snippets, traces) into the CLI
fired one message per line. ``--editor`` opens $VISUAL/$EDITOR on a temp file
and the saved buffer becomes ``args.query`` verbatim -- a single turn.
"""

import shlex
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO))
try:
    from hermes_cli import main as cli_main
    from hermes_cli._parser import build_top_level_parser
finally:
    sys.path.remove(str(REPO))

HOSTILE = 'log line 1 "quoted" $(evil) `id` back\\slash\nsecond line\n\nfourth after blank'


def _parse(argv):
    built = build_top_level_parser()
    parser = built[0] if isinstance(built, tuple) else built
    return parser.parse_args(argv)


def _resolve():
    fn = getattr(cli_main, "_read_editor_query", None)
    assert callable(fn), "--editor resolver not implemented (RED)"
    return fn


class _Tty:
    def __init__(self, is_tty=True):
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


def _fake_editor(tmp_path, body):
    """EDITOR value whose 'edit session' writes ``body`` to the temp file."""
    script = tmp_path / "fake_editor.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"Path(sys.argv[1]).write_text({body!r}, encoding='utf-8')\n",
        encoding="utf-8",
    )
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def _args_with_editor(monkeypatch, tmp_path, body, *, is_tty=True):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, body))
    monkeypatch.setattr(sys, "stdin", _Tty(is_tty))
    return _parse(["chat", "--editor"])


def test_chat_parser_accepts_editor():
    args = _parse(["chat", "--editor"])
    assert args.editor is True
    assert args.query is None


def test_editor_mutually_exclusive_with_query(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _parse(["chat", "-q", "x", "--editor"])
    assert exc.value.code == 2


def test_editor_mutually_exclusive_with_query_file(tmp_path):
    f = tmp_path / "q.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _parse(["chat", "--query-file", str(f), "--editor"])
    assert exc.value.code == 2


def test_editor_buffer_becomes_single_verbatim_query(tmp_path, monkeypatch):
    """Multi-line buffer arrives as ONE args.query, byte-identical."""
    args = _args_with_editor(monkeypatch, tmp_path, HOSTILE)
    _resolve()(args)
    assert args.query == HOSTILE
    assert args.query.count("\n") == 3  # one message, not four


def test_editor_strips_comment_lines(tmp_path, monkeypatch):
    args = _args_with_editor(monkeypatch, tmp_path, "#! ignore me\nreal prompt\n")
    _resolve()(args)
    assert args.query == "real prompt"


def test_editor_empty_buffer_aborts(tmp_path, monkeypatch):
    args = _args_with_editor(monkeypatch, tmp_path, "#! only comments\n  \n")
    with pytest.raises(SystemExit) as exc:
        _resolve()(args)
    assert exc.value.code != 0
    assert args.query is None


def test_editor_conflicts_with_programmatic_query(tmp_path, monkeypatch):
    """Direct-namespace callers (bypassing argparse) still get exclusivity."""
    args = _args_with_editor(monkeypatch, tmp_path, "buffer")
    args.query = "preset"
    with pytest.raises(SystemExit) as exc:
        _resolve()(args)
    assert exc.value.code == 2


def test_editor_needs_interactive_terminal(tmp_path, monkeypatch):
    args = _args_with_editor(monkeypatch, tmp_path, "buffer", is_tty=False)
    with pytest.raises(SystemExit) as exc:
        _resolve()(args)
    assert exc.value.code == 2
