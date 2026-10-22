"""``hermes dump`` must count configured MCP servers (#139829).

Servers live under the top-level ``mcp_servers`` config key; the dump read
``mcp.servers`` — a key nothing writes — so it reported 0 on every install,
sending operators hunting for a configuration fault that does not exist.
"""

from pathlib import Path
from types import SimpleNamespace


def _mcp_line(out: str) -> str:
    for line in out.splitlines():
        if "mcp_servers:" in line:
            return line
    raise AssertionError(f"no 'mcp_servers:' line in dump output:\n{out}")


def _seed(home: Path, config_yaml: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config_yaml)
    (home / ".env").write_text("")


def _run_dump(monkeypatch, capsys, tmp_path, config_yaml: str) -> str:
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    _seed(get_hermes_home(), config_yaml)
    dump.run_dump(SimpleNamespace(show_keys=False))
    return capsys.readouterr().out


def test_dump_counts_top_level_mcp_servers(monkeypatch, capsys, tmp_path):
    out = _run_dump(monkeypatch, capsys, tmp_path, (
        "mcp_servers:\n"
        "  example:\n"
        "    command: npx\n"
        "  other:\n"
        "    command: uvx\n"
    ))
    assert _mcp_line(out).strip().endswith("2"), (
        f"dump must count top-level mcp_servers; got: {_mcp_line(out)!r}"
    )


def test_dump_mcp_count_zero_without_servers(monkeypatch, capsys, tmp_path):
    out = _run_dump(monkeypatch, capsys, tmp_path, "display:\n  streaming: true\n")
    assert _mcp_line(out).strip().endswith("0")
