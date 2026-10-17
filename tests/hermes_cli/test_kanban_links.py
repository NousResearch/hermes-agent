"""``hermes_cli.kanban_links`` — canonical Kanban card deep links.

Covers the contract the CLI, the notifier and the browser board must agree on:
``<dashboard-base>/kanban?board=<slug>&task=<id>``.

``<dashboard-base>`` is the URL the dashboard is *reachable* at, which is usually NOT a
public one: a single-machine install is on loopback, an on-prem one on an RFC1918 address.
So the base resolves declared-URL-first, then the local host record — and the board param
must never be dropped, because an omitted board makes the browser fall back to its own
localStorage pin and open the wrong board.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_links


@pytest.fixture()
def declared_url(monkeypatch):
    """Operator-declared URL, as an internet-facing/reverse-proxied deploy would set."""
    monkeypatch.setattr(
        kanban_links, "_declared_public_url", lambda: "https://board.example.com/hermes"
    )


@pytest.fixture()
def local_only(monkeypatch):
    """No declared URL; a local dashboard is running (the plain single-machine install)."""
    monkeypatch.setattr(kanban_links, "_declared_public_url", lambda: None)
    monkeypatch.setattr(kanban_links, "_local_dashboard_base", lambda: "http://localhost:9119")
    monkeypatch.setattr(kanban_links, "_served_profiles", lambda: ("default", "git"))


@pytest.fixture()
def no_dashboard(monkeypatch):
    """Nothing configured and nothing running — no base can be built."""
    monkeypatch.setattr(kanban_links, "_declared_public_url", lambda: None)
    monkeypatch.setattr(kanban_links, "_local_dashboard_base", lambda: None)


@pytest.fixture()
def current_board(monkeypatch):
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "get_current_board", lambda: "wsl-master")


def test_declared_url_wins_and_includes_board_and_task(declared_url, current_board):
    url = kanban_links.task_url("t_abc123")
    assert url is not None
    assert url.startswith("https://board.example.com/hermes/kanban?")
    assert "board=wsl-master" in url
    assert "task=t_abc123" in url


def test_declared_url_keeps_reverse_proxy_prefix(declared_url, current_board):
    # The path prefix is part of the operator's declared authority; dropping it 404s.
    url = kanban_links.task_url("t_1")
    assert url is not None
    assert url.startswith("https://board.example.com/hermes/kanban?")


def test_local_dashboard_is_enough_without_any_config(local_only, current_board):
    # The single-machine case: nothing declared, the dashboard simply on loopback.
    url = kanban_links.task_url("t_1")
    assert url is not None
    assert url.startswith("http://localhost:9119/kanban?")
    assert "board=wsl-master" in url


def test_local_link_scopes_to_a_served_profile(local_only, current_board, monkeypatch):
    monkeypatch.setattr(kanban_links, "_link_profile", lambda: "git")
    url = kanban_links.task_url("t_1")
    assert url is not None
    assert "profile=git" in url


def test_local_link_omits_a_profile_the_dashboard_does_not_serve(local_only, current_board, monkeypatch):
    # The served set is default/git, but the active profile is "other" -> omit the param
    # rather than name a profile the server does not serve.
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "other", raising=False)
    url = kanban_links.task_url("t_1")
    assert url is not None
    assert "profile=" not in url


def test_loopback_bind_is_rendered_as_localhost(monkeypatch, current_board):
    # A browser on the host machine cannot reach the guest's 127.0.0.1 (WSL/container NAT);
    # "localhost" follows the port forward both sides agree on.
    class Rec:
        port = 9119
        host = "127.0.0.1"

    monkeypatch.setattr(kanban_links, "_declared_public_url", lambda: None)
    monkeypatch.setattr(kanban_links, "_host_record", lambda: Rec())
    monkeypatch.setattr(kanban_links, "_served_profiles", lambda: ())
    url = kanban_links.task_url("t_1")
    assert url is not None
    assert url.startswith("http://localhost:9119/kanban?")
    assert "127.0.0.1" not in url


def test_lan_bind_keeps_its_address(monkeypatch, current_board):
    # An on-prem install on an RFC1918 address must keep it: "localhost" would be wrong.
    class Rec:
        port = 9119
        host = "192.168.1.20"

    monkeypatch.setattr(kanban_links, "_declared_public_url", lambda: None)
    monkeypatch.setattr(kanban_links, "_host_record", lambda: Rec())
    monkeypatch.setattr(kanban_links, "_served_profiles", lambda: ())
    url = kanban_links.task_url("t_1")
    assert url is not None
    assert url.startswith("http://192.168.1.20:9119/kanban?")


def test_no_dashboard_means_no_link(no_dashboard, current_board):
    assert kanban_links.task_url("t_1") is None


def test_board_is_never_omitted(declared_url, current_board):
    url = kanban_links.task_url("t_1")
    assert url is not None
    assert "board=wsl-master" in url


def test_explicit_board_overrides_the_current_one(declared_url):
    url = kanban_links.task_url("t_1", board="other-board")
    assert url is not None
    assert "board=other-board" in url
    assert "wsl-master" not in url


def test_blank_task_id_is_not_a_link(declared_url, current_board):
    assert kanban_links.task_url("   ") is None


def test_task_id_is_url_encoded(declared_url, current_board):
    url = kanban_links.task_url("t/odd id")
    assert url is not None
    assert "task=t%2Fodd%20id" in url


def test_trailing_slash_on_declared_url_does_not_double_up(monkeypatch, current_board):
    # Patch the raw config source, not the normalizing wrapper, so the rstrip under test
    # actually runs.
    monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", "https://board.example.com/")
    url = kanban_links.task_url("t_1")
    assert url is not None
    assert "example.com//kanban" not in url
    assert url.startswith("https://board.example.com/kanban?")
