"""GET/PUT /api/config carry the profile's config revision, so a client can order racing answers.

Desktop windows, the Webapp and other backends write the same ``config.yaml``; a PUT that only says
``{ok}`` left the renderer unable to tell which of two racing saves or reads was newer. The revision
is opt-in on GET (an envelope, so existing clients keep the bare config they PUT back whole), always
present on PUT, per profile, and strictly increasing across this server's saves.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient

JS_SAFE_INTEGER = 2**53 - 1


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes_home"
    named = root / "profiles" / "b"
    named.mkdir(parents=True)
    for home, theme in ((root, "ember"), (named, "everforest")):
        (home / "config.yaml").write_text(f"desktop:\n  theme: {theme}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    from agent import secret_scope
    from tui_gateway import launch_profile_policy

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    return root, named


@pytest.fixture
def client(homes):
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _read(client, profile=None):
    query = "with_revision=true" + (f"&profile={profile}" if profile else "")
    resp = client.get(f"/api/config?{query}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"config", "revision"}
    assert isinstance(body["revision"], int) and 0 < body["revision"] <= JS_SAFE_INTEGER
    return body["config"], body["revision"]


def _save(client, theme, profile=None):
    path = "/api/config" + (f"?profile={profile}" if profile else "")
    resp = client.put(path, json={"config": {"desktop": {"theme": theme}}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["revision"], int) and body["revision"] <= JS_SAFE_INTEGER
    return body["revision"]


def test_bare_get_is_unchanged_and_the_envelope_carries_the_same_config(client):
    bare = client.get("/api/config").json()
    assert "revision" not in bare and bare["desktop"]["theme"] == "ember"

    config, _ = _read(client)
    assert config == bare


def test_a_save_answers_a_newer_revision_that_the_next_read_reports(client):
    _, before = _read(client)
    saved = _save(client, "mono")
    assert saved > before

    config, after = _read(client)
    assert (config["desktop"]["theme"], after) == ("mono", saved)


def test_saves_strictly_advance_even_inside_one_timestamp_tick_or_behind_the_clock(client, homes):
    root, _ = homes
    # Another writer stamped the file a day ahead (a clock step, a synced copy): every later save
    # still has to read as newer, though the wall clock is behind the file.
    ahead = os.stat(root / "config.yaml").st_mtime_ns + 86_400 * 10**9
    os.utime(root / "config.yaml", ns=(ahead, ahead))
    _, last = _read(client)
    assert last == ahead // 1000

    # Back-to-back saves land within one coarse filesystem tick on most hosts.
    for theme in ("mono", "nous", "ember", "mono"):
        revision = _save(client, theme)
        assert revision > last
        last = revision

    config, reported = _read(client)
    assert (config["desktop"]["theme"], reported) == ("mono", last)


def test_revisions_are_per_profile(client):
    _, default_before = _read(client)
    _, named_before = _read(client, "b")

    saved = _save(client, "mono", profile="b")

    named, named_after = _read(client, "b")
    default, default_after = _read(client)
    assert saved > named_before and named_after == saved
    assert named["desktop"]["theme"] == "mono"
    assert (default["desktop"]["theme"], default_after) == ("ember", default_before)


def test_an_edit_by_another_process_reads_as_newer(client, homes):
    root, _ = homes
    saved = _save(client, "mono")
    path = root / "config.yaml"
    path.write_text("desktop:\n  theme: nous\n", encoding="utf-8")
    later = (saved + 1_000_000) * 1000  # one second after the save, whatever this host's tick
    os.utime(path, ns=(later, later))

    config, revision = _read(client)
    assert (config["desktop"]["theme"], revision) == ("nous", later // 1000)
    assert _save(client, "ember") > revision


def test_the_first_save_of_a_missing_config_has_a_revision(client, homes):
    root, _ = homes
    (root / "config.yaml").unlink()
    resp = client.get("/api/config?with_revision=true")
    assert resp.status_code == 200 and resp.json()["revision"] == 0

    saved = _save(client, "mono")
    config, revision = _read(client)
    assert saved > 0 and (config["desktop"]["theme"], revision) == ("mono", saved)
