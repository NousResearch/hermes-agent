"""M5 / D5 — History media projection: `/api/sessions/{id}/messages?include_media=true`.

Behavior contract (not change-detectors):

* Default calls are untouched — no ``media`` key appears without the opt-in.
* With ``include_media=true`` each message carries a deterministic ``media`` list
  derived server-side from STORED data (literal ``MEDIA:`` tags in the persisted
  reply text), re-running the delivery pipeline's own extraction + safety gates
  (``extract_media_from_reply``): the same file class the live event stream (D1)
  describes.
* Files that still exist project full metadata ``{path, kind, mime, size,
  available: true}`` (the D1 payload shape); missing files project fallback
  metadata ``{path, available: false, name, kind, mime}`` — enough for the
  desktop's never-silent fallback card, never a silent disappearance.
* Extraction is bounded to stored message text of the page actually returned —
  no re-scanning the whole transcript, no filesystem probing beyond one
  ``stat`` per extracted ref.
* Determinism: identical stores produce identical projections (stable
  first-occurrence order), so a reopened session renders the same way twice.
"""

from __future__ import annotations

import asyncio

import pytest

from hermes_cli import web_server
from hermes_cli.web_routers import sessions as sessions_router

pytest.importorskip("fastapi")


def _msg(role, content, **extra):
    row = {"role": role, "content": content, "timestamp": 1700000000.0}
    row.update(extra)
    return row


class _MediaSessionDB:
    """Minimal SessionDB fake: one session, stored rows handed straight back.

    ``get_messages`` mirrors the real read's pagination contract loosely — the
    endpoint caps the page at 500 and this fake returns what it is given.
    """

    rows = []

    def __init__(self, *args, **kwargs):
        pass

    def resolve_session_id(self, session_id):
        return "sess_media"

    def resolve_resume_session_id(self, session_id):
        return session_id

    def get_messages(self, session_id, limit=None, offset=0, latest=False, include_compacted=False):
        return [dict(r) for r in type(self).rows]

    def close(self):
        pass


@pytest.fixture
def fake_db(monkeypatch):
    _MediaSessionDB.rows = []
    monkeypatch.setattr("hermes_state.SessionDB", _MediaSessionDB)
    return _MediaSessionDB


# ── Projection helper unit contract ─────────────────────────────────────────


def test_projection_maps_only_messages_with_tags(tmp_path, monkeypatch):
    from hermes_cli.session_media_projection import build_media_refs_for_messages

    existing = tmp_path / "chart.png"
    existing.write_bytes(b"\x89PNG fake")

    messages = [
        _msg("user", "make me a chart"),
        _msg("assistant", f"Here it is.\n\nMEDIA:{existing}"),
        _msg("assistant", "no media here"),
        _msg("assistant", f"Also: `MEDIA:{existing}`\nDone."),
    ]

    refs = build_media_refs_for_messages(messages)
    assert set(refs) == {1, 3}
    assert refs[1][0]["path"] == str(existing)
    assert refs[1][0]["available"] is True


def test_projection_missing_file_gets_fallback_metadata(tmp_path):
    from hermes_cli.session_media_projection import build_media_refs_for_messages

    gone = tmp_path / "deleted.png"
    messages = [_msg("assistant", f"chart:\nMEDIA:{gone}")]

    refs = build_media_refs_for_messages(messages)

    row = refs[0][0]
    assert row["path"] == str(gone)
    assert row["available"] is False
    assert row["name"] == "deleted.png"
    assert row["mime"].startswith("image/")
    assert "size" not in row
    assert "kind" in row


def test_projection_is_deterministic_and_deduped(tmp_path):
    from hermes_cli.session_media_projection import build_media_refs_for_messages

    first = tmp_path / "a.png"
    first.write_bytes(b"x")
    second = tmp_path / "b.png"
    second.write_bytes(b"y")
    messages = [
        _msg("assistant", f"MEDIA:{second}\nMEDIA:{first}\nMEDIA:{first} `MEDIA:{first}`"),
    ]

    run_one = build_media_refs_for_messages(messages)
    run_two = build_media_refs_for_messages(messages)

    assert run_one == run_two  # reopened sessions render identically
    paths = [r["path"] for r in run_one[0]]
    assert paths == [str(second), str(first)]  # first-occurrence order, deduped


def test_projection_rejects_unsafe_paths_via_delivery_gates(tmp_path):
    from hermes_cli.session_media_projection import build_media_refs_for_messages

    # A credential store path: the delivery denylist rejects it even though it
    # exists on disk — the projection must never widen the live gate.
    import os

    from pathlib import Path as _P

    home = os.path.expanduser("~")
    ssh_key = _P(home) / ".ssh" / "id_rsa"
    messages = [_msg("assistant", f"MEDIA:{ssh_key}")]
    refs = build_media_refs_for_messages(messages)
    assert refs == {}


# ── Endpoint contract ───────────────────────────────────────────────────────


def test_endpoint_without_optin_has_no_media_key(monkeypatch, fake_db):
    fake_db.rows = [_msg("assistant", "MEDIA:/tmp/does-not-matter.png")]

    response = asyncio.run(
        web_server.get_session_messages(
            session_id="sess_media",
            profile=None,
            limit=None,
            offset=0,
            order=None,
            include_compacted=False,
        )
    )
    assert all("media" not in m for m in response["messages"])


def test_endpoint_include_media_projects_existing_and_missing(monkeypatch, fake_db, tmp_path):
    existing = tmp_path / "kept.png"
    existing.write_bytes(b"\x89PNG")
    gone = tmp_path / "lost.png"

    fake_db.rows = [
        _msg("user", "two charts please"),
        _msg("assistant", f"kept:\nMEDIA:{existing}\n\nlost:\nMEDIA:{gone}"),
    ]

    response = asyncio.run(
        web_server.get_session_messages(
            session_id="sess_media",
            profile=None,
            limit=None,
            offset=0,
            order=None,
            include_compacted=False,
            include_media=True,
        )
    )

    media = response["messages"][1]["media"]
    by_name = {r["path"]: r for r in media}
    assert by_name[str(existing)]["available"] is True
    assert by_name[str(existing)]["size"] == len(b"\x89PNG")
    assert by_name[str(gone)]["available"] is False
    assert by_name[str(gone)]["name"] == "lost.png"
    # The user message carries an (empty) media list — uniform shape.
    assert response["messages"][0]["media"] == []


def test_endpoint_include_media_empty_for_tagless_history(monkeypatch, fake_db):
    fake_db.rows = [_msg("assistant", "plain text only"), _msg("user", "hi")]

    response = asyncio.run(
        web_server.get_session_messages(
            session_id="sess_media",
            profile=None,
            limit=None,
            offset=0,
            order=None,
            include_compacted=False,
            include_media=True,
        )
    )
    # Every message gets a (here empty) media list under the opt-in.
    assert response["messages"][0]["media"] == []
    assert response["messages"][1]["media"] == []


def test_endpoint_media_survives_tool_result_context(monkeypatch, fake_db, tmp_path):
    """MEDIA refs inside stored tool-result text project too (real stores keep
    deliverable tags in tool rows via the delivery echo)."""
    existing = tmp_path / "from-tool.csv"
    existing.write_bytes(b"a,b\n1,2\n")

    fake_db.rows = [
        _msg("assistant", "running the export", tool_name="export"),
        _msg(
            "tool",
            f"export complete.\nMEDIA:{existing}",
            tool_name="export",
            tool_call_id="call_1",
        ),
    ]

    response = asyncio.run(
        web_server.get_session_messages(
            session_id="sess_media",
            profile=None,
            limit=None,
            offset=0,
            order=None,
            include_compacted=False,
            include_media=True,
        )
    )
    media = response["messages"][1]["media"]
    assert media and media[0]["path"] == str(existing)
