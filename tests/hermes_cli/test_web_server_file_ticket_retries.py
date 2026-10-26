"""A browser retry of the same download URL redeems its ticket again, a few times (#144667).

The loopback gate spends a download ticket's use before the file is opened, and the browser
retries a failed download by requesting that exact URL. The ticket stays bounded: it admits
``DOWNLOAD_MAX_REDEMPTIONS`` requests (a HEAD counts like a GET) within ``DOWNLOAD_TTL_SECONDS``
of minting. A different path, query or method is refused without using one up. Stream tickets
stay reusable until they expire. Signing out ends both (``test_webapp_sign_out_sockets.py``).
"""

import secrets
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server, web_server_file_tickets
from tests.hermes_cli.test_web_server_file_ticket_generations import _ticket
from tests.hermes_cli.test_web_server_files import forced_files_client


@pytest.fixture
def browser_download(forced_files_client, monkeypatch):
    """The Webapp's file routes under a fresh session token, a file in the managed root, and a
    browser that sends no session header (a download link cannot)."""
    client, root = forced_files_client
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", secrets.token_urlsafe(32))
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    root.mkdir()
    clip = root / "clip.mp4"
    clip.write_bytes(b"hello")
    browser = TestClient(web_server.app)
    try:
        yield client, browser, clip
    finally:
        browser.close()


def test_the_same_download_url_can_be_retried_a_few_times_then_dies(browser_download):
    client, browser, clip = browser_download
    route = web_server_file_tickets.ROUTES["download"]
    params = _ticket(client, "download", {"path": str(clip)})

    assert browser.get(route, params=params).content == b"hello"
    retry = browser.get(route, params=params)
    assert (retry.status_code, retry.content) == (200, b"hello")

    # Refused requests use none of the ticket's uses.
    assert browser.get(route, params={**params, "path": str(clip) + ".other"}).status_code == 401
    assert browser.get(route, params={**params, "profile": "other"}).status_code == 401
    assert browser.post(route, params=params).status_code == 401
    assert browser.get(web_server_file_tickets.ROUTES["stream"], params=params).status_code == 401

    # Three admitted requests, not a credential that lasts for the TTL: the GET
    # and its retry above, then a HEAD, which is admitted like a GET and uses one up.
    assert browser.head(route, params=params).status_code == 200
    assert browser.get(route, params=params).status_code == 401
    assert browser.head(route, params=params).status_code == 401

    stream = _ticket(client, "stream", {"path": str(clip)})
    for _ in range(web_server_file_tickets.DOWNLOAD_MAX_REDEMPTIONS + 1):
        partial = browser.get(
            web_server_file_tickets.ROUTES["stream"],
            params=stream,
            headers={"Range": "bytes=1-3"},
        )
        assert (partial.status_code, partial.content) == (206, b"ell")


def test_a_download_ticket_dies_its_ttl_after_minting_with_uses_left(browser_download, monkeypatch):
    client, browser, clip = browser_download
    route = web_server_file_tickets.ROUTES["download"]
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        web_server_file_tickets, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )

    params = _ticket(client, "download", {"path": str(clip)})
    assert browser.get(route, params=params).content == b"hello"
    clock["now"] += web_server_file_tickets.DOWNLOAD_TTL_SECONDS - 1
    assert browser.get(route, params=params).content == b"hello"

    # Uses are left; only the TTL, counted from minting, refuses it.
    assert web_server_file_tickets.DOWNLOAD_MAX_REDEMPTIONS > 2
    clock["now"] += 1
    assert browser.get(route, params=params).status_code == 401
