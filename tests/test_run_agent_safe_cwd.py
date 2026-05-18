from unittest.mock import patch

import run_agent


def test_safe_process_cwd_falls_back_when_process_cwd_deleted(tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()

    with patch("run_agent.os.getcwd", side_effect=FileNotFoundError):
        assert run_agent._safe_process_cwd(str(fallback)) == str(fallback)


def test_safe_process_cwd_uses_home_before_tmp_when_cwd_deleted(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    with patch("run_agent.os.getcwd", side_effect=FileNotFoundError):
        assert run_agent._safe_process_cwd("/does/not/exist") == str(home)


def test_run_agent_does_not_use_eager_getcwd_defaults():
    source = open(run_agent.__file__, encoding="utf-8").read()

    assert "os.getenv(\"TERMINAL_CWD\", os.getcwd())" not in source
    assert "os.getenv('TERMINAL_CWD', os.getcwd())" not in source
