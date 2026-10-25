"""A browser retry of the same download URL redeems its ticket again, a few times (#144667).

The loopback gate spends a download ticket's use before the file is opened, and the browser
retries a failed download by requesting that exact URL. The ticket stays bounded: it admits
``DOWNLOAD_MAX_REDEMPTIONS`` requests (a HEAD counts like a GET) within ``DOWNLOAD_TTL_SECONDS``
of minting. A different path, query or method is refused without using one up, and signing out
drops a ticket minted under the replaced browser credential even with uses left. Stream tickets
stay reusable until they expire.
"""

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from hermes_cli import browser_credential, web_server, web_server_file_tickets


@pytest.fixture
def browser_download(monkeypatch, tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    clip = root / "clip.mp4"
    clip.write_bytes(b"hello")
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    for key, value in (
        ("auth_required", False),
        ("bound_host", None),
        ("ui_surface", "webapp"),
    ):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    browser = TestClient(web_server.app)
    try:
        yield client, browser, clip
    finally:
        browser.close()
        client.close()


def _mint(client, route, **query):
    minted = client.post("/api/files/ticket", json={"route": route, **query})
    assert minted.status_code == 200, minted.text
    return {**query, "ticket": minted.json()["ticket"]}


def test_the_same_download_url_can_be_retried_a_few_times_then_dies(browser_download):
    client, browser, clip = browser_download
    route = web_server_file_tickets.ROUTES["download"]
    params = _mint(client, "download", path=str(clip))

    assert browser.get(route, params=params).content == b"hello"
    retry = browser.get(route, params=params)
    assert (retry.status_code, retry.content) == (200, b"hello")

    # Refused requests use none of the ticket's uses.
    assert browser.get(route, params={**params, "path": str(clip) + ".other"}).status_code == 401
    assert browser.get(route, params={**params, "profile": "other"}).status_code == 401
    assert browser.post(route, params=params).status_code == 401
    assert browser.get(web_server_file_tickets.ROUTES["stream"], params=params).status_code == 401

    # A handful of uses, not a credential that lasts for the TTL. A HEAD of the
    # URL is admitted like a GET and uses one up.
    assert 2 < web_server_file_tickets.DOWNLOAD_MAX_REDEMPTIONS <= 5
    for _ in range(web_server_file_tickets.DOWNLOAD_MAX_REDEMPTIONS - 3):
        assert browser.get(route, params=params).content == b"hello"
    assert browser.head(route, params=params).status_code == 200
    assert browser.get(route, params=params).status_code == 401
    assert browser.head(route, params=params).status_code == 401

    stream = _mint(client, "stream", path=str(clip))
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

    params = _mint(client, "download", path=str(clip))
    assert browser.get(route, params=params).content == b"hello"
    clock["now"] += web_server_file_tickets.DOWNLOAD_TTL_SECONDS - 1
    assert browser.get(route, params=params).content == b"hello"

    # Uses are left; only the TTL, counted from minting, refuses it.
    assert web_server_file_tickets.DOWNLOAD_MAX_REDEMPTIONS > 2
    clock["now"] += 1
    assert browser.get(route, params=params).status_code == 401


def test_sign_out_drops_a_download_ticket_minted_under_the_browser_credential(
    browser_download, monkeypatch
):
    client, browser, clip = browser_download
    route = web_server_file_tickets.ROUTES["download"]
    params = _mint(client, "download", path=str(clip))
    assert browser.get(route, params=params).content == b"hello"
    other = web_server_file_tickets.mint(
        "download",
        {"path": str(clip)},
        authority=web_server_file_tickets.FileAuthority(clip, ()),
        session_token=None,
    )
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", web_server._SESSION_TOKEN)
    browser_credential.rotate("b" * 43)

    assert browser.get(route, params=params).status_code == 401
    kept = browser.get(route, params={"path": str(clip), "ticket": other})
    assert kept.content == b"hello"
