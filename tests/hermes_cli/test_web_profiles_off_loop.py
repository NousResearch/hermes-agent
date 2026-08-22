"""Regression tests: ``/api/profiles`` handlers must not block the event loop.

``hermes_cli/web_routers/profiles.py`` holds handler bodies that were extracted
verbatim from ``web_server.py``, so the blocking library calls they inherited
run inline on the ASGI event loop. The worst of them are unbounded from the
dashboard's point of view: deleting a profile whose gateway is up sleeps up to
10 s in ``profiles._stop_gateway_process``, and ``describe-auto`` makes a
provider round-trip with a 60 s ceiling. While the loop is parked, the process
serves nothing else — including the ``/api/ws`` probes the desktop app and the
dashboard's own Chat tab depend on.

Two complementary assertions per site:

* a **loop probe** — the stubbed callee records whether an event loop is
  running in its own thread, mirroring
  ``tests/hermes_cli/test_cron_dashboard_off_loop.py``; and
* a **concurrency proof** — the stubbed callee blocks on a ``threading.Event``
  while an unrelated request is timed, which fails if the loop is parked.

The block is bounded by a timeout so a regression costs the suite a few
seconds rather than hanging it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# How long a stubbed blocking call holds its thread when nobody releases it.
BLOCK_SECONDS = 5.0
# A concurrently-served request has to land well inside that window. The gap is
# large (a served request takes milliseconds) so the bound is not timing-fragile.
CONCURRENT_BUDGET = BLOCK_SECONDS / 2


@pytest.fixture()
def profile_dir(tmp_path, monkeypatch) -> Path:
    """A real profile directory under a throwaway HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import profiles as profiles_mod

    d = profiles_mod.get_profile_dir("demo")
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def client(profile_dir):
    """One ``TestClient`` context == one portal == one event loop.

    This matters: entering the context manager pins a single blocking portal
    for every request, so a handler that parks the loop really does starve the
    concurrent request below. A bare ``TestClient(app)`` spins up a fresh loop
    per request and would pass these tests even unfixed.
    """
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        # web_server resolves _SESSION_TOKEN once, at import, so read it back
        # from the module rather than pinning a value from this file.
        c.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        yield c


@pytest.fixture()
def loop_probe():
    """Collect ``(tag, on_loop)`` proof from stubbed blocking callees."""
    seen: list[tuple[str, bool]] = []

    def probe(tag: str) -> None:
        try:
            asyncio.get_running_loop()
            seen.append((tag, True))
        except RuntimeError:
            seen.append((tag, False))

    return seen, probe


def assert_off_loop(seen, tag: str) -> None:
    assert (tag, False) in seen, (
        f"{tag} must run off the event loop; proof: {seen}"
    )


class _Blocker:
    """A stand-in for a slow library call, released by the test."""

    def __init__(self, result=None):
        self.entered = threading.Event()
        self.release = threading.Event()
        self._result = result

    def __call__(self, *args, **kwargs):
        self.entered.set()
        # Bounded on purpose: a regression must not hang the suite.
        self.release.wait(timeout=BLOCK_SECONDS)
        return self._result


def assert_serves_concurrently(client, blocker: _Blocker, fire) -> None:
    """Fire a request that blocks inside its handler, then time another one.

    ``fire`` issues the blocking request from a worker thread. Once the stub
    has been entered, an unrelated cheap route is timed on this thread: it can
    only answer quickly if the blocking work left the event loop.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        blocked = pool.submit(fire)
        try:
            assert blocker.entered.wait(timeout=BLOCK_SECONDS), (
                "the blocking stub was never reached"
            )
            start = time.monotonic()
            probe = client.get("/api/profiles/demo/setup-command")
            elapsed = time.monotonic() - start
        finally:
            blocker.release.set()
        blocked.result(timeout=BLOCK_SECONDS * 2)

    assert probe.status_code == 200, probe.text
    assert elapsed < CONCURRENT_BUDGET, (
        f"a concurrent request waited {elapsed:.2f}s while the handler was "
        f"busy — the event loop was parked by the blocking call"
    )


# ── DELETE /api/profiles/{name} — the 10 s gateway-stop sleep ────────────────


def test_delete_profile_runs_off_loop(client, monkeypatch, loop_probe, tmp_path):
    seen, probe = loop_probe
    from hermes_cli import profiles as profiles_mod

    def fake_delete(name, yes=False):
        probe("delete_profile")
        return tmp_path / "profiles" / name

    monkeypatch.setattr(profiles_mod, "delete_profile", fake_delete)

    resp = client.delete("/api/profiles/demo")

    assert resp.status_code == 200, resp.text
    assert_off_loop(seen, "delete_profile")


def test_delete_profile_does_not_block_the_dashboard(client, monkeypatch, tmp_path):
    """``delete_profile`` stops a running gateway by polling for up to 10 s.

    That is longer than the desktop's WebSocket ready-probe tolerates, so it
    must not hold the loop.
    """
    from hermes_cli import profiles as profiles_mod

    blocker = _Blocker(result=tmp_path / "profiles" / "demo")
    monkeypatch.setattr(profiles_mod, "delete_profile", blocker)

    assert_serves_concurrently(
        client, blocker, lambda: client.delete("/api/profiles/demo")
    )


# ── POST /api/profiles/{name}/describe-auto — the 60 s LLM round-trip ────────


def _outcome(ok=True, reason="described", description="a demo profile"):
    from hermes_cli.profile_describer import DescribeOutcome

    return DescribeOutcome("demo", ok, reason, description=description)


def test_describe_auto_runs_off_loop(client, monkeypatch, loop_probe):
    seen, probe = loop_probe
    from hermes_cli import profile_describer

    def fake_describe(name, overwrite=False, timeout=None):
        probe("describe_profile")
        return _outcome()

    monkeypatch.setattr(profile_describer, "describe_profile", fake_describe)

    resp = client.post("/api/profiles/demo/describe-auto", json={"overwrite": True})

    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True
    assert_off_loop(seen, "describe_profile")


def test_describe_auto_does_not_block_the_dashboard(client, monkeypatch):
    """The auxiliary provider call has a 60 s ceiling — six times the
    desktop's disconnect threshold."""
    from hermes_cli import profile_describer

    blocker = _Blocker(result=_outcome())
    monkeypatch.setattr(profile_describer, "describe_profile", blocker)

    assert_serves_concurrently(
        client,
        blocker,
        lambda: client.post(
            "/api/profiles/demo/describe-auto", json={"overwrite": True}
        ),
    )


# ── PATCH /api/profiles/{name} — rename walks and rewrites the profile tree ──


def test_rename_profile_runs_off_loop(client, monkeypatch, loop_probe, tmp_path):
    seen, probe = loop_probe
    from hermes_cli import profiles as profiles_mod

    def fake_rename(old, new):
        probe("rename_profile")
        return tmp_path / "profiles" / new

    monkeypatch.setattr(profiles_mod, "rename_profile", fake_rename)

    resp = client.patch("/api/profiles/demo", json={"new_name": "renamed"})

    assert resp.status_code == 200, resp.text
    assert_off_loop(seen, "rename_profile")


def test_rename_profile_does_not_block_the_dashboard(client, monkeypatch, tmp_path):
    from hermes_cli import profiles as profiles_mod

    blocker = _Blocker(result=tmp_path / "profiles" / "renamed")
    monkeypatch.setattr(profiles_mod, "rename_profile", blocker)

    assert_serves_concurrently(
        client,
        blocker,
        lambda: client.patch("/api/profiles/demo", json={"new_name": "renamed"}),
    )


# ── /api/profiles/active — sticky active-profile state file ──────────────────


def test_get_active_profile_runs_off_loop(client, monkeypatch, loop_probe):
    seen, probe = loop_probe
    from hermes_cli import profiles as profiles_mod

    def fake_get_active():
        probe("get_active_profile")
        return "demo"

    def fake_get_current():
        probe("get_active_profile_name")
        return "default"

    monkeypatch.setattr(profiles_mod, "get_active_profile", fake_get_active)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", fake_get_current)

    resp = client.get("/api/profiles/active")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"active": "demo", "current": "default"}
    assert_off_loop(seen, "get_active_profile")
    assert_off_loop(seen, "get_active_profile_name")


def test_set_active_profile_runs_off_loop(client, monkeypatch, loop_probe):
    seen, probe = loop_probe
    from hermes_cli import profiles as profiles_mod

    def fake_set_active(name):
        probe("set_active_profile")

    monkeypatch.setattr(profiles_mod, "set_active_profile", fake_set_active)

    resp = client.post("/api/profiles/active", json={"name": "demo"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["active"] == "demo"
    assert_off_loop(seen, "set_active_profile")


# ── PUT /api/profiles/{name}/description — profile.yaml read/modify/write ────


def test_update_description_runs_off_loop(client, monkeypatch, loop_probe):
    seen, probe = loop_probe
    from hermes_cli import profiles as profiles_mod

    def fake_write_meta(profile_dir, **kwargs):
        probe("write_profile_meta")

    monkeypatch.setattr(profiles_mod, "write_profile_meta", fake_write_meta)

    resp = client.put(
        "/api/profiles/demo/description", json={"description": "a demo profile"}
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["description_auto"] is False
    assert_off_loop(seen, "write_profile_meta")


# ── GET /api/profiles/{name}/desktop-overlay — reads desktop.json ────────────


def test_desktop_overlay_read_runs_off_loop(client, monkeypatch, loop_probe, profile_dir):
    seen, probe = loop_probe
    (profile_dir / "desktop.json").write_text('{"theme": "dark"}', encoding="utf-8")

    real_read_text = Path.read_text

    def probing_read_text(self, *args, **kwargs):
        if self.name == "desktop.json":
            probe("desktop.json read")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", probing_read_text)

    resp = client.get("/api/profiles/demo/desktop-overlay")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"exists": True, "desktop": {"theme": "dark"}}
    assert_off_loop(seen, "desktop.json read")


def test_desktop_overlay_absent_still_reports_missing(client):
    """The offloaded read must keep distinguishing "no file" from "no data"."""
    resp = client.get("/api/profiles/demo/desktop-overlay")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"exists": False, "desktop": None}


def test_desktop_overlay_null_document_still_reports_present(client, profile_dir):
    """A ``desktop.json`` holding literal ``null`` exists, it is just empty —
    it must not be reported the same as a missing file."""
    (profile_dir / "desktop.json").write_text("null", encoding="utf-8")

    resp = client.get("/api/profiles/demo/desktop-overlay")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"exists": True, "desktop": None}


def test_desktop_overlay_unreadable_document_is_a_500(client, profile_dir):
    """A malformed overlay still surfaces as a 500, not a silent success."""
    (profile_dir / "desktop.json").write_text("{not json", encoding="utf-8")

    resp = client.get("/api/profiles/demo/desktop-overlay")

    assert resp.status_code == 500, resp.text


# ── Status-code mapping must survive the move into a worker thread ───────────


def test_delete_missing_profile_is_still_404(client, monkeypatch):
    from hermes_cli import profiles as profiles_mod

    def fake_delete(name, yes=False):
        raise FileNotFoundError(f"Profile '{name}' does not exist.")

    monkeypatch.setattr(profiles_mod, "delete_profile", fake_delete)

    assert client.delete("/api/profiles/demo").status_code == 404


def test_rename_to_existing_profile_is_still_400(client, monkeypatch):
    from hermes_cli import profiles as profiles_mod

    def fake_rename(old, new):
        raise FileExistsError(f"Profile '{new}' already exists.")

    monkeypatch.setattr(profiles_mod, "rename_profile", fake_rename)

    resp = client.patch("/api/profiles/demo", json={"new_name": "taken"})
    assert resp.status_code == 400, resp.text


def test_set_active_missing_profile_is_still_404(client, monkeypatch):
    from hermes_cli import profiles as profiles_mod

    def fake_set_active(name):
        raise FileNotFoundError(f"Profile '{name}' does not exist.")

    monkeypatch.setattr(profiles_mod, "set_active_profile", fake_set_active)

    assert client.post("/api/profiles/active", json={"name": "demo"}).status_code == 404


def test_describe_auto_unknown_profile_is_still_404(client):
    """``_resolve_profile_dir`` still runs before the worker hop, so an
    unknown profile is a 404 rather than a 500 from the wrapped call."""
    assert (
        client.post("/api/profiles/nope/describe-auto", json={}).status_code == 404
    )
