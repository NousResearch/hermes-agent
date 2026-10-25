"""A browser retry of the same download URL redeems that ticket again, a few times.

The download is fetched once, and the browser repeats that exact URL when the
download fails or is cancelled. Those retries succeed inside the download TTL.
The ticket is not a reusable credential: a different path, query, or method is
refused without spending a retry, an expired ticket is refused, and signing out
drops one minted under the replaced browser credential. Stream tickets stay
reusable until they expire.
"""

import time

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
    # A handful of browser retries, not a credential that lasts for the TTL.
    assert 1 < web_server_file_tickets.DOWNLOAD_MAX_REDEMPTIONS <= 5

    assert browser.get(route, params=params).content == b"hello"
    assert (
        browser.get(route, params={**params, "path": str(clip) + ".other"}).status_code
        == 401
    )
    assert browser.get(route, params={**params, "profile": "other"}).status_code == 401
    assert browser.post(route, params=params).status_code == 401
    assert (
        browser.get(web_server_file_tickets.ROUTES["stream"], params=params).status_code
        == 401
    )
    for _ in range(web_server_file_tickets.DOWNLOAD_MAX_REDEMPTIONS - 1):
        assert browser.get(route, params=params).content == b"hello"
    assert browser.get(route, params=params).status_code == 401

    stream = _mint(client, "stream", path=str(clip))
    for _ in range(web_server_file_tickets.DOWNLOAD_MAX_REDEMPTIONS + 1):
        partial = browser.get(
            web_server_file_tickets.ROUTES["stream"],
            params=stream,
            headers={"Range": "bytes=1-3"},
        )
        assert (partial.status_code, partial.content) == (206, b"ell")


def test_an_expired_download_ticket_is_refused_while_retries_remain(
    browser_download, monkeypatch
):
    client, browser, clip = browser_download
    route = web_server_file_tickets.ROUTES["download"]
    real = time.monotonic
    offset = {"seconds": 0.0}
    monkeypatch.setattr(
        web_server_file_tickets.time, "monotonic", lambda: real() + offset["seconds"]
    )

    fresh = _mint(client, "download", path=str(clip))
    assert browser.get(route, params=fresh).content == b"hello"
    offset["seconds"] = web_server_file_tickets.DOWNLOAD_TTL_SECONDS + 1
    assert browser.get(route, params=fresh).status_code == 401

    monkeypatch.setattr(web_server_file_tickets, "DOWNLOAD_TTL_SECONDS", 0)
    expired = _mint(client, "download", path=str(clip))
    assert browser.get(route, params=expired).status_code == 401


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
