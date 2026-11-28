"""Zero-line log tails skip history, including when following or filtering."""

import pytest

from hermes_cli.logs import tail_log


@pytest.mark.parametrize("large", [False, True], ids=["small", "large"])
@pytest.mark.parametrize("level", [None, "WARNING"], ids=["unfiltered", "filtered"])
@pytest.mark.parametrize("follow", [False, True], ids=["tail", "follow"])
def test_zero_lines_does_not_print_history(tmp_path, monkeypatch, capsys, large, level, follow):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    history = "2026-01-01 00:00:00 WARNING demo: historical entry\n"
    (log_dir / "agent.log").write_text(history * (30_000 if large else 3), encoding="utf-8")
    followed = []
    monkeypatch.setattr("hermes_cli.logs._follow_log", lambda path, **filters: followed.append(path))

    tail_log(num_lines=0, level=level, follow=follow)

    output = capsys.readouterr().out
    assert "historical entry" not in output
    assert followed == ([log_dir / "agent.log"] if follow else [])
