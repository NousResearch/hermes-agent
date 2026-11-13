"""CLI diagnostics must not emit contributor-controlled terminal codes."""
from scripts.ci import check_thread_hops as guard


def test_untrusted_path_and_source_controls_are_escaped(tmp_path, monkeypatch, capsys):
    path = tmp_path / "tools" / "bad\x1b[31m.py"
    path.parent.mkdir()
    path.write_text("x.submit(f)  # \x1b[31mCONTROL\n", encoding="utf-8")
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    assert guard.main([str(path), "--all"]) == 1
    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert output.count("\\x1b") == 2
    assert "CONTROL" in output
