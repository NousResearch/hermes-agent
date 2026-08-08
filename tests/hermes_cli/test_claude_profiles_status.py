"""`hermes auth status claude-profiles` — what a person reads before a job.

The command reads local state and one usage endpoint. It starts no model and
it spends no tokens. It prints the nickname a person chose, never an address
and never a token.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent import claude_cli_profiles as ccp
from hermes_cli.auth_commands import auth_status_command

SOON = datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)


def write_config(section):
    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    path.write_text(yaml.dump({"claude_cli_profiles": section} if section else {}))


def configure(tmp_path, count=2):
    write_config({"profiles": [
        {"name": name, "config_dir": str(tmp_path / name)}
        for name in ("work", "spare")[:count]
    ]})


def reader_for(numbers):
    def reader(p, **_):
        return numbers[p.name]
    return reader


def usage(name, five_hour=None, weekly=None, problem=None, five_hour_reset=None):
    return ccp.ProfileUsage(
        name=name,
        five_hour_percent=five_hour,
        weekly_percent=weekly,
        five_hour_reset=five_hour_reset,
        problem=problem,
    )


class TestStatusLines:
    def test_no_profiles_configured_says_the_switcher_is_off(self):
        write_config(None)
        text = "\n".join(ccp.status_lines())
        assert "off" in text.lower()
        assert "two" in text.lower()

    def test_one_profile_configured_says_the_switcher_is_off(self, tmp_path):
        configure(tmp_path, count=1)
        text = "\n".join(ccp.status_lines(usage_reader=lambda p, **_: pytest.fail("no read")))
        assert "off" in text.lower()

    def test_every_profile_appears_with_its_numbers(self, tmp_path):
        configure(tmp_path)
        lines = ccp.status_lines(usage_reader=reader_for({
            "work": usage("work", 31.0, 44.0),
            "spare": usage("spare", 2.0, 3.0),
        }))
        text = "\n".join(lines)
        assert "work" in text and "spare" in text
        assert "31%" in text and "44%" in text

    def test_a_full_window_shows_its_reset_time(self, tmp_path):
        configure(tmp_path)
        lines = ccp.status_lines(usage_reader=reader_for({
            "work": usage("work", 100.0, 44.0, five_hour_reset=SOON),
            "spare": usage("spare", 2.0, 3.0),
        }))
        text = "\n".join(lines)
        assert "2026-08-07 18:00 UTC" in text
        assert "full" in text

    def test_the_profile_in_use_is_marked(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("spare")
        lines = ccp.status_lines(usage_reader=reader_for({
            "work": usage("work", 10.0, 10.0),
            "spare": usage("spare", 2.0, 3.0),
        }))
        marked = [line for line in lines if line.strip().startswith("spare")]
        assert marked and "in use" in marked[0]

    def test_a_profile_without_a_login_is_named_as_such(self, tmp_path):
        configure(tmp_path)
        lines = ccp.status_lines(usage_reader=reader_for({
            "work": usage("work", problem="no_login"),
            "spare": usage("spare", 2.0, 3.0),
        }))
        text = "\n".join(lines)
        assert "sign in" in text.lower()

    def test_the_stop_percentage_is_shown(self, tmp_path):
        configure(tmp_path)
        write_config({
            "stop_at_percent": 80,
            "profiles": [
                {"name": name, "config_dir": str(tmp_path / name)}
                for name in ("work", "spare")
            ],
        })
        text = "\n".join(ccp.status_lines(usage_reader=reader_for({
            "work": usage("work", 10.0, 10.0),
            "spare": usage("spare", 2.0, 3.0),
        })))
        assert "80" in text

    def test_it_reports_when_no_account_is_available(self, tmp_path):
        configure(tmp_path)
        text = "\n".join(ccp.status_lines(usage_reader=reader_for({
            "work": usage("work", 100.0, 10.0, five_hour_reset=SOON),
            "spare": usage("spare", 100.0, 10.0, five_hour_reset=SOON),
        })))
        assert "no claude code profile is available" in text.lower()

    def test_it_prints_no_token_and_no_address(self, tmp_path):
        configure(tmp_path)
        for name in ("work", "spare"):
            directory = tmp_path / name
            directory.mkdir()
            (directory / ".credentials.json").write_text(
                '{"claudeAiOauth": {"accessToken": "sk-ant-oat01-fake",'
                ' "expiresAt": 99999999999999}}'
            )
        text = "\n".join(ccp.status_lines(usage_fetcher=lambda _t: {
            "limits": [{"kind": "session", "percent": 5},
                       {"kind": "weekly_all", "percent": 6}]
        }))
        assert "sk-ant" not in text
        assert "@" not in text

    def test_reading_the_status_does_not_change_the_selection(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("spare")
        ccp.status_lines(usage_reader=reader_for({
            "work": usage("work", 1.0, 1.0),
            "spare": usage("spare", 100.0, 100.0, five_hour_reset=SOON),
        }))
        assert ccp.active_profile_name() == "spare"


class TestTheCommand:
    def test_it_prints_the_profile_table(self, tmp_path, capsys, monkeypatch):
        configure(tmp_path)
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: usage(p.name, 12.0, 13.0),
        )
        auth_status_command(SimpleNamespace(provider="claude-profiles"))
        printed = capsys.readouterr().out
        assert "work" in printed and "spare" in printed
        assert "12%" in printed

    def test_the_short_name_works_too(self, tmp_path, capsys, monkeypatch):
        configure(tmp_path)
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: usage(p.name, 12.0, 13.0),
        )
        auth_status_command(SimpleNamespace(provider="claude-profile"))
        assert "work" in capsys.readouterr().out

    def test_it_never_starts_a_login(self, tmp_path, capsys, monkeypatch):
        configure(tmp_path)
        monkeypatch.setattr(
            ccp, "read_profile_usage",
            lambda p, **_: usage(p.name, problem="no_login"),
        )
        import webbrowser

        monkeypatch.setattr(
            webbrowser, "open",
            lambda *_a, **_k: pytest.fail("the status command must not start a login"),
        )
        auth_status_command(SimpleNamespace(provider="claude-profiles"))
        assert "sign in" in capsys.readouterr().out.lower()
