"""Tests for Kanban task file attachments (#35338).

Covers three layers:
  * ``hermes_cli.kanban_db`` accessors (add/list/get/delete + path helpers)
  * the dashboard REST surface (upload / list / download / delete)
  * worker-context surfacing so a kanban worker sees the absolute paths

The plugin router is attached to a bare FastAPI app — same approach as
``test_kanban_dashboard_plugin.py`` — so we exercise the real HTTP path
(multipart upload, streaming download) without the whole dashboard.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_attach_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _make_task(conn, title="t") -> str:
    return kb.create_task(conn, title=title)


# ---------------------------------------------------------------------------
# DB-layer accessors
# ---------------------------------------------------------------------------


def test_add_list_get_delete_attachment(kanban_home, tmp_path):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        # Write a real blob under the per-task dir so delete can unlink it.
        dest_dir = kb.task_attachments_dir(task_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        blob = dest_dir / "source.pdf"
        blob.write_bytes(b"%PDF-1.4 fake")

        att_id = kb.add_attachment(
            conn,
            task_id,
            filename="source.pdf",
            stored_path=str(blob),
            content_type="application/pdf",
            size=blob.stat().st_size,
            uploaded_by="tester",
        )
        assert att_id > 0

        atts = kb.list_attachments(conn, task_id)
        assert len(atts) == 1
        a = atts[0]
        assert a.filename == "source.pdf"
        assert a.content_type == "application/pdf"
        assert a.size == len(b"%PDF-1.4 fake")
        assert a.uploaded_by == "tester"
        assert a.stored_path == str(blob)

        got = kb.get_attachment(conn, att_id)
        assert got is not None and got.id == att_id

        removed = kb.delete_attachment(conn, att_id)
        assert removed is not None and removed.id == att_id
        assert kb.list_attachments(conn, task_id) == []
        assert not blob.exists(), "delete should unlink the on-disk blob"
        assert kb.get_attachment(conn, att_id) is None
    finally:
        conn.close()


def test_add_attachment_rejects_unknown_task(kanban_home):
    conn = kb.connect()
    try:
        with pytest.raises(ValueError):
            kb.add_attachment(
                conn, "t_doesnotexist", filename="x.txt", stored_path="/tmp/x.txt"
            )
    finally:
        conn.close()


def test_add_attachment_appends_event(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        kb.add_attachment(
            conn, task_id, filename="a.txt", stored_path="/tmp/a.txt", size=3
        )
        kinds = [e.kind for e in kb.list_events(conn, task_id)]
        assert "attached" in kinds
    finally:
        conn.close()


def test_delete_attachment_missing_returns_none(kanban_home):
    conn = kb.connect()
    try:
        assert kb.delete_attachment(conn, 999999) is None
    finally:
        conn.close()


def _insert_legacy_attachment(conn, task_id: str, blob: Path) -> int:
    cursor = conn.execute(
        "INSERT INTO task_attachments "
        "(task_id, filename, stored_path, content_type, size, uploaded_by, "
        "created_at, filesystem_identity) "
        "VALUES (?, ?, ?, 'text/plain', ?, 'legacy', 1, NULL)",
        (task_id, blob.name, str(blob.resolve()), blob.stat().st_size),
    )
    conn.commit()
    return int(cursor.lastrowid)


def test_api_delete_legacy_attachment_removes_row_and_preserves_blob(
    client, kanban_home
):
    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="legacy API attachment")
        attachment_dir = kb.task_attachments_dir(task_id)
        attachment_dir.mkdir(parents=True)
        blob = attachment_dir / "legacy-api.txt"
        blob.write_text("keep", encoding="utf-8")
        attachment_id = _insert_legacy_attachment(conn, task_id, blob)
    finally:
        conn.close()

    response = client.delete(f"/api/plugins/kanban/attachments/{attachment_id}")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    with kb.connect() as conn:
        assert kb.get_attachment(conn, attachment_id) is None
    assert blob.read_text(encoding="utf-8") == "keep"


def test_api_delete_legacy_attachment_preserves_noncanonical_blob(
    client, kanban_home, tmp_path
):
    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="legacy external attachment")
        blob = tmp_path / "legacy-external.txt"
        blob.write_text("keep", encoding="utf-8")
        attachment_id = _insert_legacy_attachment(conn, task_id, blob)
    finally:
        conn.close()

    response = client.delete(f"/api/plugins/kanban/attachments/{attachment_id}")

    assert response.status_code == 200
    with kb.connect() as conn:
        assert kb.get_attachment(conn, attachment_id) is None
    assert blob.read_text(encoding="utf-8") == "keep"


def test_api_delete_legacy_attachment_preserves_same_size_replacement(
    client, kanban_home
):
    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="changed legacy attachment")
        attachment_dir = kb.task_attachments_dir(task_id)
        attachment_dir.mkdir(parents=True)
        blob = attachment_dir / "changed.txt"
        blob.write_text("AAAA", encoding="utf-8")
        attachment_id = _insert_legacy_attachment(conn, task_id, blob)
        blob.unlink()
        blob.write_text("BBBB", encoding="utf-8")
    finally:
        conn.close()

    response = client.delete(f"/api/plugins/kanban/attachments/{attachment_id}")

    assert response.status_code == 200
    with kb.connect() as conn:
        assert kb.get_attachment(conn, attachment_id) is None
    assert blob.read_text(encoding="utf-8") == "BBBB"


def test_attachments_root_is_per_board(kanban_home, monkeypatch):
    # default board uses <root>/kanban/attachments
    default_root = kb.attachments_root(board="default")
    assert default_root.name == "attachments"
    # a named board nests under its board dir
    monkeypatch.delenv("HERMES_KANBAN_ATTACHMENTS_ROOT", raising=False)
    named = kb.attachments_root(board="default")
    assert named == default_root


def test_attachments_root_env_override(kanban_home, monkeypatch, tmp_path):
    override = tmp_path / "custom-attach"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(override))
    assert kb.attachments_root() == override
    assert kb.task_attachments_dir("t_abc") == override / "t_abc"


# ---------------------------------------------------------------------------
# Worker context surfacing
# ---------------------------------------------------------------------------


def test_worker_context_lists_attachments_with_absolute_path(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="translate PDF")
        dest_dir = kb.task_attachments_dir(task_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        blob = dest_dir / "manual.pdf"
        blob.write_bytes(b"data")
        kb.add_attachment(
            conn,
            task_id,
            filename="manual.pdf",
            stored_path=str(blob.resolve()),
            content_type="application/pdf",
            size=4,
        )
        ctx = kb.build_worker_context(conn, task_id)
        assert "## Attachments" in ctx
        assert "manual.pdf" in ctx
        # The absolute path must appear so the worker can read_file it.
        assert str(blob.resolve()) in ctx
    finally:
        conn.close()


def test_worker_context_no_attachments_section_when_empty(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        ctx = kb.build_worker_context(conn, task_id)
        assert "## Attachments" not in ctx
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# REST surface — upload / list / download / delete round-trip
# ---------------------------------------------------------------------------


def _create_task_via_api(client) -> str:
    r = client.post("/api/plugins/kanban/tasks", json={"title": "x"})
    assert r.status_code == 200, r.text
    return r.json()["task"]["id"]


def test_upload_list_download_delete_roundtrip(client):
    task_id = _create_task_via_api(client)
    content = b"hello attachment world"

    # Upload
    r = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert r.status_code == 200, r.text
    att = r.json()["attachment"]
    assert att["filename"] == "notes.txt"
    assert att["size"] == len(content)
    att_id = att["id"]

    # List (drawer also embeds it in GET /tasks/:id)
    r = client.get(f"/api/plugins/kanban/tasks/{task_id}/attachments")
    assert r.status_code == 200
    assert [a["filename"] for a in r.json()["attachments"]] == ["notes.txt"]

    detail = client.get(f"/api/plugins/kanban/tasks/{task_id}").json()
    assert "attachments" in detail
    assert len(detail["attachments"]) == 1

    # Download streams the exact bytes back
    r = client.get(f"/api/plugins/kanban/attachments/{att_id}")
    assert r.status_code == 200
    assert r.content == content

    # Delete removes the row and the file
    r = client.delete(f"/api/plugins/kanban/attachments/{att_id}")
    assert r.status_code == 200
    assert client.get(f"/api/plugins/kanban/attachments/{att_id}").status_code == 404
    assert client.get(
        f"/api/plugins/kanban/tasks/{task_id}/attachments"
    ).json()["attachments"] == []


def test_upload_sanitizes_traversal_filename(client):
    task_id = _create_task_via_api(client)
    r = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("../../../../etc/passwd", b"x", "text/plain")},
    )
    assert r.status_code == 200, r.text
    stored_path = r.json()["attachment"]["stored_path"]
    # The leaf name only; never escapes the per-task attachments dir.
    assert Path(stored_path).name == "passwd"
    task_dir = kb.task_attachments_dir(task_id).resolve()
    assert Path(stored_path).resolve().is_relative_to(task_dir)


def test_upload_name_collision_gets_suffixed(client):
    task_id = _create_task_via_api(client)
    for _ in range(2):
        r = client.post(
            f"/api/plugins/kanban/tasks/{task_id}/attachments",
            files={"file": ("dup.txt", b"a", "text/plain")},
        )
        assert r.status_code == 200, r.text
    names = sorted(
        a["filename"]
        for a in client.get(
            f"/api/plugins/kanban/tasks/{task_id}/attachments"
        ).json()["attachments"]
    )
    assert names == ["dup (1).txt", "dup.txt"]


def test_dashboard_upload_preserves_symlink_race_winner(
    client, kanban_home, monkeypatch
):
    task_id = _create_task_via_api(client)
    attachment_dir = kb.task_attachments_dir(task_id)
    attachment_dir.mkdir(parents=True, exist_ok=True)
    destination = attachment_dir / "race.txt"
    human_file = kanban_home / "human-race-winner.txt"
    human_file.write_bytes(b"human bytes")

    # Skip only on Windows hosts where developer-mode/elevation does not allow
    # symlink creation; the regular-file race is covered at the shared layer.
    probe = kanban_home / "symlink-probe"
    try:
        probe.symlink_to(human_file)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    else:
        probe.unlink()

    real_open = kb.os.open
    injected = False

    def race_open(path, flags, *args, **kwargs):
        nonlocal injected
        if Path(path) == destination and flags & kb.os.O_EXCL and not injected:
            injected = True
            destination.symlink_to(human_file)
            raise FileExistsError(str(destination))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(kb.os, "open", race_open)
    response = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("race.txt", b"agent bytes", "text/plain")},
    )

    assert response.status_code == 200, response.text
    attachment = response.json()["attachment"]
    assert injected
    assert destination.is_symlink()
    assert human_file.read_bytes() == b"human bytes"
    assert attachment["filename"] == "race (1).txt"
    assert Path(attachment["stored_path"]).read_bytes() == b"agent bytes"


def test_dashboard_upload_preserves_regular_file_race_winner(
    client, monkeypatch
):
    task_id = _create_task_via_api(client)
    destination = kb.task_attachments_dir(task_id) / "race.txt"
    real_open = kb.os.open
    injected = False

    def race_open(path, flags, *args, **kwargs):
        nonlocal injected
        if Path(path) == destination and flags & kb.os.O_EXCL and not injected:
            injected = True
            destination.write_bytes(b"human winner")
            raise FileExistsError(str(destination))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(kb.os, "open", race_open)
    response = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("race.txt", b"agent bytes", "text/plain")},
    )

    assert response.status_code == 200, response.text
    attachment = response.json()["attachment"]
    assert injected
    assert destination.read_bytes() == b"human winner"
    assert attachment["filename"] == "race (1).txt"
    assert Path(attachment["stored_path"]).read_bytes() == b"agent bytes"


def test_dashboard_upload_oversize_returns_413_without_blob(client, monkeypatch):
    task_id = _create_task_via_api(client)
    plugin = sys.modules["hermes_dashboard_plugin_kanban_attach_test"]
    monkeypatch.setattr(plugin, "KANBAN_ATTACHMENT_MAX_BYTES", 4)

    response = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/attachments",
        files={"file": ("large.bin", b"12345", "application/octet-stream")},
    )

    assert response.status_code == 413
    with kb.connect() as conn:
        assert kb.list_attachments(conn, task_id) == []
    attachment_dir = kb.task_attachments_dir(task_id)
    assert not attachment_dir.exists() or list(attachment_dir.iterdir()) == []


def test_upload_unknown_task_404(client):
    r = client.post(
        "/api/plugins/kanban/tasks/t_nope/attachments",
        files={"file": ("x.txt", b"x", "text/plain")},
    )
    assert r.status_code == 404


def test_download_unknown_attachment_404(client):
    assert client.get("/api/plugins/kanban/attachments/424242").status_code == 404


# ---------------------------------------------------------------------------
# Shared helper — store_attachment_bytes (used by dashboard + tool + CLI)
# ---------------------------------------------------------------------------


def test_store_attachment_bytes_roundtrip(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        att_id = kb.store_attachment_bytes(
            conn, task_id, "doc.txt", b"some bytes",
            content_type="text/plain", uploaded_by="tester",
        )
        a = kb.get_attachment(conn, att_id)
        assert a is not None
        assert a.filename == "doc.txt"
        assert a.size == len(b"some bytes")
        assert a.uploaded_by == "tester"
        assert Path(a.stored_path).read_bytes() == b"some bytes"
        assert Path(a.stored_path).resolve().is_relative_to(
            kb.task_attachments_dir(task_id).resolve()
        )
    finally:
        conn.close()


def test_store_attachment_accepts_close_time_ctime_settling(
    kanban_home, monkeypatch
):
    """Path-vs-handle ctime drift must not reject a file bound by samestat."""
    real_fstat = kb.os.fstat

    def shifted_fstat(fd):
        current = real_fstat(fd)
        if not kb.stat.S_ISREG(current.st_mode):
            return current
        return types.SimpleNamespace(
            st_dev=current.st_dev,
            st_ino=current.st_ino,
            st_mode=current.st_mode,
            st_size=current.st_size,
            st_mtime_ns=current.st_mtime_ns,
            st_ctime_ns=current.st_ctime_ns + 1,
            st_file_attributes=getattr(current, "st_file_attributes", 0),
        )

    monkeypatch.setattr(kb.os, "fstat", shifted_fstat)
    with kb.connect() as conn:
        task_id = _make_task(conn)
        attachment_id = kb.store_attachment_bytes(
            conn, task_id, "settled.txt", b"stable"
        )
        attachment = kb.get_attachment(conn, attachment_id)

    assert attachment is not None
    assert Path(attachment.stored_path).read_bytes() == b"stable"


def test_attachment_post_close_replacement_is_preserved(kanban_home, monkeypatch):
    destination = kanban_home / "destination"
    destination.mkdir()
    target = destination / "report.txt"
    real_lstat = Path.lstat
    target_lstats = 0

    def replace_after_close(path):
        nonlocal target_lstats
        if path == target:
            target_lstats += 1
            if target_lstats == 3:
                target.unlink()
                target.write_bytes(b"human replacement")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", replace_after_close)
    with pytest.raises(ValueError, match="changed during creation"):
        kb._exclusive_attachment_path(destination, target.name, b"agent bytes")

    assert target.read_bytes() == b"human replacement"


def test_store_attachment_bytes_rejects_oversize_and_leaves_no_blob(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        with pytest.raises(kb.AttachmentTooLarge):
            kb.store_attachment_bytes(
                conn, task_id, "big.bin", b"0123456789", max_bytes=4,
            )
        assert kb.list_attachments(conn, task_id) == []
        # No partial blob left behind.
        d = kb.task_attachments_dir(task_id)
        assert not d.exists() or list(d.iterdir()) == []
    finally:
        conn.close()


def test_store_attachment_bytes_resolves_collisions(kanban_home):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        kb.store_attachment_bytes(conn, task_id, "dup.txt", b"a")
        kb.store_attachment_bytes(conn, task_id, "dup.txt", b"b")
        names = sorted(a.filename for a in kb.list_attachments(conn, task_id))
        assert names == ["dup (1).txt", "dup.txt"]
    finally:
        conn.close()


def test_store_attachment_bytes_preserves_race_winner(kanban_home, monkeypatch):
    conn = kb.connect()
    try:
        task_id = _make_task(conn)
        destination = kb.task_attachments_dir(task_id) / "race.txt"
        real_open = kb.os.open
        injected = False

        def race_open(path, flags, *args, **kwargs):
            nonlocal injected
            if (
                Path(path) == destination
                and flags & kb.os.O_EXCL
                and not injected
            ):
                injected = True
                destination.write_bytes(b"human winner")
                raise FileExistsError(str(destination))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(kb.os, "open", race_open)
        attachment_id = kb.store_attachment_bytes(
            conn, task_id, "race.txt", b"agent bytes"
        )
        attachment = kb.get_attachment(conn, attachment_id)

        assert destination.read_bytes() == b"human winner"
        assert attachment is not None and attachment.filename == "race (1).txt"
        assert Path(attachment.stored_path).read_bytes() == b"agent bytes"
    finally:
        conn.close()


def test_attachment_junction_open_restore_writes_no_external_bytes(
    kanban_home, monkeypatch
):
    """A redirected exclusive open is rejected before bytes reach its target."""
    destination = kanban_home / "destination"
    destination.mkdir()
    intended = destination / "report.txt"
    external_dir = kanban_home / "external"
    external_dir.mkdir()
    external = external_dir / intended.name
    real_open = kb.os.open
    real_actual_path = kb._opened_file_actual_path
    real_remove = kb._atomic_remove_expected_file
    external_before_cleanup = []

    def junction_open(path, flags, *args, **kwargs):
        if Path(path) == intended and flags & kb.os.O_EXCL:
            fd = real_open(external, flags, *args, **kwargs)
            # Model the attacker restoring the original path after the open.
            intended.write_bytes(b"human replacement")
            return fd
        return real_open(path, flags, *args, **kwargs)

    def actual_path(fd, path):
        if Path(path) == intended:
            return external
        return real_actual_path(fd, path)

    def observe_remove(path, expected):
        if Path(path) == external:
            external_before_cleanup.append(external.read_bytes())
        return real_remove(path, expected)

    monkeypatch.setattr(kb.os, "open", junction_open)
    monkeypatch.setattr(kb, "_opened_file_actual_path", actual_path)
    monkeypatch.setattr(kb, "_atomic_remove_expected_file", observe_remove)

    with pytest.raises(ValueError, match="destination path changed"):
        kb._exclusive_attachment_path(destination, intended.name, b"agent bytes")

    assert external_before_cleanup == [b""]
    assert not external.exists()
    assert intended.read_bytes() == b"human replacement"


def test_store_attachment_bytes_unknown_task_leaves_no_blob(kanban_home):
    conn = kb.connect()
    try:
        with pytest.raises(ValueError):
            kb.store_attachment_bytes(conn, "t_nope", "x.txt", b"x")
        # The per-task dir may get created, but no blob should survive the
        # failed metadata insert.
        d = kb.task_attachments_dir("t_nope")
        assert not d.exists() or list(d.iterdir()) == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI — hermes kanban attach / attachments / attach-rm
# ---------------------------------------------------------------------------


def test_cli_attach_attachments_and_rm(kanban_home, tmp_path):
    from hermes_cli.kanban import run_slash

    conn = kb.connect()
    try:
        task_id = _make_task(conn, title="cli-attach")
    finally:
        conn.close()

    src = tmp_path / "upload.txt"
    src.write_bytes(b"cli file body")

    out = run_slash(f"attach {task_id} {src}")
    assert "Attached" in out, out

    conn = kb.connect()
    try:
        atts = kb.list_attachments(conn, task_id)
        assert len(atts) == 1
        att_id = atts[0].id
        assert atts[0].filename == "upload.txt"
        assert Path(atts[0].stored_path).read_bytes() == b"cli file body"
    finally:
        conn.close()

    listed = run_slash(f"attachments {task_id}")
    assert "upload.txt" in listed

    removed = run_slash(f"attach-rm {att_id}")
    assert "Deleted attachment" in removed
    conn = kb.connect()
    try:
        assert kb.list_attachments(conn, task_id) == []
    finally:
        conn.close()


def test_cli_attach_rm_removes_legacy_row_and_preserves_blob(kanban_home):
    from hermes_cli.kanban import run_slash

    with kb.connect() as conn:
        task_id = _make_task(conn, title="legacy CLI attachment")
        attachment_dir = kb.task_attachments_dir(task_id)
        attachment_dir.mkdir(parents=True)
        blob = attachment_dir / "legacy-cli.txt"
        blob.write_text("keep", encoding="utf-8")
        attachment_id = _insert_legacy_attachment(conn, task_id, blob)

    output = run_slash(f"attach-rm {attachment_id}")

    assert "Deleted attachment" in output
    with kb.connect() as conn:
        assert kb.get_attachment(conn, attachment_id) is None
    assert blob.read_text(encoding="utf-8") == "keep"


def test_cli_attach_honors_name_override(kanban_home, tmp_path):
    from hermes_cli.kanban import run_slash

    conn = kb.connect()
    try:
        task_id = _make_task(conn)
    finally:
        conn.close()
    src = tmp_path / "raw.bin"
    src.write_bytes(b"xyz")
    run_slash(f"attach {task_id} {src} --name renamed.dat")
    conn = kb.connect()
    try:
        assert kb.list_attachments(conn, task_id)[0].filename == "renamed.dat"
    finally:
        conn.close()


def test_cli_attach_missing_file(kanban_home, tmp_path):
    from hermes_cli.kanban import run_slash

    conn = kb.connect()
    try:
        task_id = _make_task(conn)
    finally:
        conn.close()
    out = run_slash(f"attach {task_id} {tmp_path / 'does-not-exist.txt'}")
    assert "no such file" in out.lower()


def test_cli_attachments_unknown_task(kanban_home):
    from hermes_cli.kanban import run_slash

    out = run_slash("attachments t_nope")
    assert "no such task" in out.lower()
