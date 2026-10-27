"""The live --query-file read keeps main's contract (B2-09): undecodable bytes are replaced; an
unreadable or empty file exits 2 naming it, with a failed ``result`` line under stream-json."""
import argparse
import json

import pytest

from hermes_cli import gateway_chat
import hermes_cli.gateway_chat_startup as startup


def _args(path, fmt="text"):
    return argparse.Namespace(query_file=str(path), query=None, output_format=fmt, model=None)


@pytest.fixture
def launched(monkeypatch):
    seen = []
    monkeypatch.setattr(startup, "ensure_launch_provider", lambda args: True)
    monkeypatch.setattr(gateway_chat, "_register_terminal_process", lambda: None)

    async def run(args, emitter=None):
        seen.append(args.query)
        return 0
    monkeypatch.setattr(gateway_chat, "run_gateway_chat", run)
    return seen


@pytest.mark.parametrize("case, expected", [("missing", "cannot read --query-file"),
                                            ("directory", "cannot read --query-file"),
                                            ("empty", "--query-file is empty"), ("latin1", None)])
@pytest.mark.parametrize("fmt", ["text", "stream-json"])
def test_query_file_read_matches_main(tmp_path, capsys, launched, case, expected, fmt):
    path = {"missing": tmp_path / "gone.txt", "directory": tmp_path}.get(case, tmp_path / "dm.txt")
    if case == "empty":
        path.write_text("  \n", encoding="utf-8")
    if case == "latin1":
        path.write_bytes(b"caf\xe9 $(id) \"quoted\"")
    code = gateway_chat.launch_from_args(_args(path, fmt))
    out, err = capsys.readouterr()
    if expected is None:
        assert code == 0 and launched == ["caf\ufffd $(id) \"quoted\""]
        return
    assert code == 2 and launched == []
    assert f"Error: {expected}" in err.replace(f"--query-file {path}", "--query-file")
    if fmt == "stream-json":
        result = [json.loads(line) for line in out.splitlines() if line.strip()][-1]
        assert result["type"] == "result" and result.get("is_error", result.get("subtype") != "success")
