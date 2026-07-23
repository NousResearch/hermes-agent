from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main


@pytest.mark.parametrize(
    "url",
    [
        "https://token@github.com/NousResearch/hermes-agent.git",
        "https://user:pass@github.com/NousResearch/hermes-agent.git",
        "https://user:p@ss@github.com/NousResearch/hermes-agent.git",
        "https://token@github.com/NousResearch/hermes-agent",
        "https://token@github.com/NousResearch/hermes-agent.git/",
    ],
)
def test_credentialed_official_urls_are_not_forks(url):
    assert hermes_main._is_fork(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://token@github.com/someuser/hermes-agent.git",
        "https://token@github.com/someuser@github.com/NousResearch/hermes-agent.git",
        "https://token@github.com/someuser/hermes-agent.git?mirror=user@host",
        "https://token@github.com/someuser/hermes-agent.git#owner@host",
    ],
)
def test_at_sign_outside_authority_does_not_hide_fork(url):
    assert hermes_main._is_fork(url) is True


def test_url_sanitizer_only_removes_authority_userinfo():
    url = "https://alice:secret@github.com/user@org/repo.git?q=a@b#c@d"
    assert hermes_main._strip_url_credentials(url) == (
        "https://github.com/user@org/repo.git?q=a@b#c@d"
    )


def test_cmd_update_redacts_credentials_in_fork_warning(monkeypatch, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    origin_url = "https://alice:moonlit-pass@github.com/someuser/hermes-agent.git"

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(
        hermes_main, "_pause_windows_gateways_for_update", lambda: None
    )
    monkeypatch.setattr(hermes_main, "_discard_lockfile_churn", lambda *args: None)
    monkeypatch.setattr(hermes_main, "_get_origin_url", lambda *args: origin_url)
    monkeypatch.setattr(hermes_main, "_resolve_update_branch", lambda args: "main")
    monkeypatch.setattr(
        hermes_main.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="fetch stopped for test"
        ),
    )

    with pytest.raises(SystemExit, match="1"):
        hermes_main._cmd_update_impl(SimpleNamespace(yes=True), gateway_mode=False)

    output = capsys.readouterr().out
    assert "https://github.com/someuser/hermes-agent.git" in output
    assert "alice" not in output
    assert "moonlit-pass" not in output
