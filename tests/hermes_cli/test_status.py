from types import SimpleNamespace

from hermes_cli.status import show_status


def test_show_status_includes_tavily_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-1234567890abcdef")

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Tavily" in output
    assert "tvly...cdef" in output
