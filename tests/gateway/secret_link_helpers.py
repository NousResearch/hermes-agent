"""Shared fixtures for the secret-link tests: a temp Hermes root with profiles, a fresh store,
a real aiohttp client over the real routes, and spies for the two mocked edges (Telegram send,
gateway restart)."""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import hermes_yaml as yaml

from gateway import secret_link_store
from gateway.secret_link_store import SecretLinkStore

BASE = "https://hermes.example.com"
SENTINEL = "sentinel-Secret-Value-93f1c0"
_NONCE_RE = re.compile(r'name="nonce" value="([^"]+)"')


class Clock:
    def __init__(self):
        self.now = 5000.0

    def __call__(self):
        return self.now


def make_profile(root: Path, name: str) -> Path:
    path = root / "profiles" / name
    path.mkdir(parents=True)
    (path / "config.yaml").write_text("{}\n")
    return path


@pytest.fixture(name="hermes_root")
def hermes_root_fixture(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text(yaml.safe_dump({"secret_links": {"public_base_url": BASE + "/"}}))
    make_profile(root, "alpha")
    make_profile(root, "beta")
    return root


@pytest.fixture(name="clock")
def clock_fixture():
    return Clock()


@pytest.fixture(name="store")
def store_fixture(monkeypatch, clock):
    fresh = SecretLinkStore(clock=clock)
    monkeypatch.setattr(secret_link_store, "STORE", fresh)
    return fresh


class Sent:
    """Records texts the request chat received."""

    def __init__(self):
        self.texts: list[str] = []

    async def __call__(self, text: str):
        self.texts.append(text)


@pytest.fixture(name="restarts")
def restarts_fixture(monkeypatch):
    """``_restart_gateway_after`` calls; ``restarts.running`` names the profiles whose gateway runs."""
    import hermes_cli.web_server_gateway as wsg
    from hermes_cli import profiles

    class _Restarts(list):
        running: set = set()

    calls = _Restarts()
    calls.running = set()

    def fake_restart(profile, *, what, label):
        calls.append(profile)
        return {"restart_started": True}

    def fake_running(profile_dir):
        return Path(profile_dir).name in calls.running or (
            "default" in calls.running and Path(profile_dir).name == ".hermes")

    monkeypatch.setattr(wsg, "_restart_gateway_after", fake_restart)
    monkeypatch.setattr(profiles, "_check_gateway_running", fake_running)
    monkeypatch.setattr(profiles, "_served_by_running_multiplexer", lambda _name: False)
    return calls


def new_link(store: SecretLinkStore, *, suggested: str = "", notify=None, on_done=None, owner_home=None) -> str:
    async def _drop(_text):
        return None
    token, _record = store.create(chat_key="telegram:42", owner_home=owner_home, suggested_key=suggested,
                                  notify=notify or _drop, on_done=on_done)
    return token


@asynccontextmanager
async def secret_client():
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from gateway.secret_link_routes import mount_secret_link_routes

    app = web.Application()
    mount_secret_link_routes(app)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


def link_path(token: str) -> str:
    return f"/secret-link/{token}"


async def get_form(client, token: str):
    resp = await client.get(link_path(token))
    html = await resp.text()
    match = _NONCE_RE.search(html)
    return resp, html, match.group(1) if match else ""


def form_nonce(html: str) -> str:
    match = _NONCE_RE.search(html)
    return match.group(1) if match else ""


async def submit(client, token: str, nonce: str, *, key: str = "STRIPE_SECRET_KEY", value: str = SENTINEL,
                 profiles=("alpha",), all_profiles: bool = False, replace: bool = False,
                 headers: dict | None = None):
    fields = [("nonce", nonce), ("key", key), ("secret", value)]
    fields += [("profile", p) for p in profiles]
    if all_profiles:
        fields.append(("all_profiles", "1"))
    if replace:
        fields.append(("replace", "1"))
    from urllib.parse import urlencode
    sent_headers = {"Origin": BASE, "Content-Type": "application/x-www-form-urlencoded"}
    sent_headers.update(headers or {})
    sent_headers = {k: v for k, v in sent_headers.items() if v is not None}
    resp = await client.post(link_path(token), data=urlencode(fields).encode("ascii"), headers=sent_headers)
    return resp, await resp.text()


def env_text(home: Path) -> str:
    path = home / ".env"
    return path.read_text() if path.exists() else ""


REQUIRED_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "Expires": "0",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": ("default-src 'none'; style-src 'self' 'unsafe-inline'; form-action 'self'; "
                                "frame-ancestors 'none'; base-uri 'none'"),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Strict-Transport-Security": "max-age=31536000",
}


def assert_secret_headers(resp):
    for name, value in REQUIRED_HEADERS.items():
        assert resp.headers.get(name) == value, name
    assert "Set-Cookie" not in resp.headers
    assert "Location" not in resp.headers
