"""Native-HTTP profile scope: the ``current`` sentinel is implicit on query/body selectors.

Desktop sends its ambient scope as ``?profile=current`` on per-profile routes. That sentinel can
never NAME another profile — the route resolves it through the ticket's home override — so it must
not be read as a profile id on ``/api/profiles/...`` either. The PATH segment stays strict: a path
segment is always a profile name.
"""
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode

import pytest
from starlette.requests import Request

from hermes_cli import profiles as profiles_mod
from hermes_cli.dashboard_auth import native_http

HOME = "/srv/hermes"
LUCY = "/srv/hermes/profiles/lucy"
JARVIS = "/srv/hermes/profiles/jarvis"


@pytest.fixture(autouse=True)
def profile_dirs(monkeypatch):
    mapping = {"default": HOME, "lucy": LUCY, "jarvis": JARVIS}
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda name: Path(mapping[name]))
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name in mapping)
    return mapping


def make_request(path, *, query=None, body=None, method="GET"):
    raw_query = urlencode(query or {}).encode()
    messages = []
    headers = []
    if body is not None:
        data = json.dumps(body).encode()
        messages.append({"type": "http.request", "body": data, "more_body": False})
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]
    else:
        messages.append({"type": "http.request", "body": b"", "more_body": False})
    scope = {
        "type": "http", "method": method, "path": path, "raw_path": path.encode(),
        "query_string": raw_query, "headers": headers, "scheme": "http",
        "server": ("test", 80), "client": ("127.0.0.1", 12345), "root_path": "",
    }

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    return Request(scope, receive)


def check(request, profile_home):
    asyncio.run(native_http._check_profile_scope(request, profile_home))


@pytest.mark.parametrize("sentinel", ["current", "", None])
def test_implicit_query_selector_on_a_profile_route_names_the_tickets_own_profile(sentinel):
    """`/api/profiles/<own>/soul?profile=current` is Desktop's normal per-profile read."""
    query = {} if sentinel is None else {"profile": sentinel}
    check(make_request("/api/profiles/default/soul", query=query), HOME)


@pytest.mark.parametrize("sentinel", ["current", ""])
def test_implicit_body_selector_on_a_profile_route_names_the_tickets_own_profile(sentinel):
    check(make_request("/api/profiles/default/soul", body={"profile": sentinel}, method="PUT"), HOME)


def test_implicit_selector_still_verifies_the_path_profile():
    """The sentinel relaxes the QUERY, never the path: another profile's SOUL stays refused."""
    with pytest.raises(PermissionError):
        check(make_request("/api/profiles/lucy/soul", query={"profile": "current"}), HOME)


def test_a_named_query_selector_still_has_to_be_the_tickets_profile():
    """R2: a real name in the query is checked against the ticket, sentinel or not."""
    with pytest.raises(PermissionError):
        check(make_request("/api/config", query={"profile": "default"}), LUCY)


def test_implicit_selector_on_a_current_profile_route_is_the_tickets_home():
    check(make_request("/api/config", query={"profile": "current"}), LUCY)
