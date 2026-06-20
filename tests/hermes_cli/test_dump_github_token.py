"""Tests for the GitHub-credential probe in ``hermes dump`` (hermes_cli.dump.run_dump).

``hermes dump`` lists each known credential as "set" / "not set" so support
triage can see what auth is configured.  GitHub credentials are honored by both
``GITHUB_TOKEN`` and ``GH_TOKEN`` (the latter is what ``gh`` and GitHub Actions
export), but the probe only looked at ``GITHUB_TOKEN`` — so a perfectly valid
``GH_TOKEN``-only environment reported GitHub as "not set".

These tests drive ``run_dump`` end-to-end and assert on the ``github`` line.
``load_hermes_dotenv`` is patched to a no-op so a real ``.env`` on the test
machine can't reintroduce a token and mask the env we set.
"""

from types import SimpleNamespace
from unittest.mock import patch


def _github_line(captured_out: str) -> str:
    for line in captured_out.splitlines():
        if line.strip().startswith("github "):
            return line.strip()
    raise AssertionError(f"no 'github' api_keys line in dump output:\n{captured_out}")


def test_dump_reads_github_from_gh_token_when_github_token_unset(monkeypatch, capsys):
    """GH_TOKEN-only env: the github row must read 'set', not 'not set'."""
    from hermes_cli import dump

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "ghp_exampletoken")

    with patch("hermes_cli.dump.load_hermes_dotenv"):
        dump.run_dump(SimpleNamespace(show_keys=False))

    line = _github_line(capsys.readouterr().out)
    assert line.endswith("set")
    assert "not set" not in line


def test_dump_reports_github_not_set_when_neither_var_present(monkeypatch, capsys):
    """Guard against over-broadening: with neither var set it stays 'not set'."""
    from hermes_cli import dump

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    with patch("hermes_cli.dump.load_hermes_dotenv"):
        dump.run_dump(SimpleNamespace(show_keys=False))

    assert _github_line(capsys.readouterr().out).endswith("not set")


def test_dump_still_reads_github_token_directly(monkeypatch, capsys):
    """The original GITHUB_TOKEN path is unchanged."""
    from hermes_cli import dump

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_directtoken")
    monkeypatch.delenv("GH_TOKEN", raising=False)

    with patch("hermes_cli.dump.load_hermes_dotenv"):
        dump.run_dump(SimpleNamespace(show_keys=False))

    line = _github_line(capsys.readouterr().out)
    assert line.endswith("set")
    assert "not set" not in line
