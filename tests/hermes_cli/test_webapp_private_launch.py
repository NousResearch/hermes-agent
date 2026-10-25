"""``hermes webapp`` where nothing checks who redeems a one-use launch link (macOS, the BSDs, illumos, WSL).

The gateway marks such a link private (``gateway.webapp_listener._one_use_links_work``). The command never
hands it to a browser launcher, whose command line other local users can read, and never prints it when it
can open a browser: it writes the link into a page in the gateway home's user-only ``webapp`` folder and
opens that page, so the launcher's command line carries only the page's path. It prints the link only
where it cannot open a browser (``--no-open``, no graphical browser, a folder it cannot keep user-only)
and on WSL, whose Windows browser cannot read a page as this user only. These tests simulate those
hosts on Linux: a real macOS or BSD run is a live check.
"""
import html
import re
import stat
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from hermes_cli import webapp_launcher as launcher
from tests.hermes_cli import test_webapp_launcher as launcher_tests

LINK, _args = launcher_tests.LINK, launcher_tests._args
world = launcher_tests.world  # the launcher's recorded side effects, as those tests use them


@pytest.fixture
def private(world):
    world["answers"]["webapp-launch"] = {"ok": True, "result": {"url": LINK, "private": True,
                                                                "expires_in_seconds": 60}}
    return world


def _pages(world) -> list[Path]:
    return sorted((world["root"] / "webapp").glob("launch-*.html"))


def _page_path(url: str) -> Path:
    parts = urlsplit(url)
    assert parts.scheme == "file"
    return Path(unquote(parts.path))


@pytest.mark.platforms("posix")
def test_a_private_link_reaches_the_browser_in_a_page_only_the_user_can_read(private, capsys):
    assert launcher.cmd_webapp(_args()) == 0
    [opened] = private["calls"]["opened"]
    assert "ticket" not in opened
    page = _page_path(opened)
    assert _pages(private) == [page]
    assert stat.S_IMODE(page.stat().st_mode) == 0o600
    assert stat.S_IMODE(page.parent.stat().st_mode) == 0o700
    refresh = re.search(r'http-equiv="refresh" content="0;url=([^"]*)"', page.read_text(encoding="utf-8"))
    assert html.unescape(refresh[1]) == LINK
    out = capsys.readouterr().out
    assert "Opened the Webapp in your browser" in out and "ticket=" not in out


@pytest.mark.platforms("posix")
def test_a_folder_other_users_could_read_is_closed_before_the_page_is_written(private):
    folder = private["root"] / "webapp"
    folder.mkdir(mode=0o755)
    folder.chmod(0o755)
    assert launcher.cmd_webapp(_args()) == 0
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert len(_pages(private)) == 1


@pytest.mark.platforms("posix")
def test_a_relative_hermes_home_still_reaches_the_browser_in_a_page(private, monkeypatch, capsys):
    """``HERMES_HOME=./root`` is supported; the page's ``file:`` URI is built from its absolute path."""
    monkeypatch.chdir(private["tmp"])
    monkeypatch.setenv("HERMES_HOME", "./root")
    assert launcher.cmd_webapp(_args()) == 0
    [opened] = private["calls"]["opened"]
    page = _page_path(opened)
    assert page.is_absolute() and [page.resolve()] == [p.resolve() for p in _pages(private)]
    assert "ticket=" not in capsys.readouterr().out


def test_pages_of_earlier_launches_are_removed(private):
    folder = private["root"] / "webapp"
    folder.mkdir(mode=0o700)
    (folder / "launch-earlier.html").write_text("old", encoding="utf-8")
    assert launcher.cmd_webapp(_args()) == 0
    assert [p.name for p in _pages(private)] != ["launch-earlier.html"]
    assert len(_pages(private)) == 1


@pytest.mark.parametrize("case", ["no-open", "no browser", "wsl", "folder others can read"])
def test_where_no_user_only_page_can_carry_it_the_link_is_printed_and_no_page_is_left(private, capsys, case):
    monkeypatch = private["monkeypatch"]
    if case == "no browser":
        monkeypatch.setattr(launcher, "_open", lambda url: private["calls"]["opened"].append(url) and False)
    if case == "wsl":
        monkeypatch.setattr("hermes_constants.is_wsl", lambda: True)
    if case == "folder others can read":
        monkeypatch.setattr("gateway.webapp_listener._readable_by_other_users", lambda path: True)
    assert launcher.cmd_webapp(_args(no_open=case == "no-open")) == 0
    assert all("ticket" not in url for url in private["calls"]["opened"])
    if case != "no browser":
        assert private["calls"]["opened"] == []
    assert _pages(private) == []
    out = capsys.readouterr().out
    assert LINK in out and "keep it private" in out and "60 seconds" in out


def test_a_checked_link_still_goes_straight_to_the_browser(world):
    """Linux and Windows links (``private: False``) are unchanged: no page, the link itself is opened."""
    assert launcher.cmd_webapp(_args()) == 0
    assert world["calls"]["opened"] == [LINK]
    assert _pages(world) == []
