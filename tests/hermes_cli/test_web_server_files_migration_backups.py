"""The backend migration's backups (``backend_migration/``: the copy of ``.env``, the hand-started backend's
environment and command line) and its record are refused by the dashboard Files routes exactly as ``.env`` is:
never listed, never read, streamed or downloaded."""

from __future__ import annotations

from tests.hermes_cli.test_web_server_files import forced_files_client  # noqa: F401 - the fixture

_BACKUPS = ("env.bak", "config.yaml.bak", "launcher-env.json", "launcher-argv.json")


def test_the_migration_backups_and_record_are_refused_like_env(forced_files_client):  # noqa: F811
    client, root = forced_files_client
    folder = root / "backend_migration"
    folder.mkdir(parents=True)
    (root / ".env").write_text("OPENAI_API_KEY=fixture-only\n", encoding="utf-8")
    (root / "notes.txt").write_text("ordinary\n", encoding="utf-8")
    (root / "backend_migration.json").write_text('{"state": "done"}\n', encoding="utf-8")
    for name in _BACKUPS:
        (folder / name).write_text("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=fixture-only\n", encoding="utf-8")

    names = [e["name"] for e in client.get("/api/files", params={"path": str(root)}).json()["entries"]]
    assert "notes.txt" in names
    assert ".env" not in names and "backend_migration" not in names and "backend_migration.json" not in names
    listing = client.get("/api/files", params={"path": str(folder)})
    assert [e["name"] for e in listing.json().get("entries", [])] == []

    def refused(path):
        return [client.get(f"/api/files/{route}", params={"path": str(path)}).status_code
                for route in ("read", "download", "stream")]

    env = refused(root / ".env")
    assert env == [403, 403, 403]
    for path in (root / "backend_migration.json", *(folder / n for n in _BACKUPS)):
        assert refused(path) == env, path
    assert client.get("/api/files/read", params={"path": str(root / "notes.txt")}).status_code == 200
