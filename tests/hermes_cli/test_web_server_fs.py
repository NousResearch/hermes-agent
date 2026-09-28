import base64
import os
from pathlib import Path, PurePosixPath

import pytest

from hermes_cli import web_server
from hermes_cli.web_routers import files as files_router

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous_auth_required


def test_fs_list_sorts_and_hides_noise(client, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "b.txt").write_text("b")
    (root / "a_dir").mkdir()
    (root / "a.txt").write_text("a")
    (root / "node_modules").mkdir()
    (root / ".git").mkdir()

    response = client.get("/api/fs/list", params={"path": str(root)})

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert [entry["name"] for entry in entries] == ["a_dir", "a.txt", "b.txt"]
    assert entries[0] == {"name": "a_dir", "path": str(root / "a_dir"), "isDirectory": True}
    assert all(entry["name"] not in {".git", "node_modules"} for entry in entries)


def test_fs_read_data_url_rejects_over_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_DATA_URL_MAX_BYTES", 3)
    target = tmp_path / "image.png"
    target.write_bytes(b"1234")

    response = client.get("/api/fs/read-data-url", params={"path": str(target)})

    assert response.status_code == 413


def test_fs_download_streams_file_without_data_url_cap(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, "_FS_DATA_URL_MAX_BYTES", 3)
    target = tmp_path / "report with spaces.pdf"
    target.write_bytes(b"123456")

    response = client.get("/api/fs/download", params={"path": str(target)})

    assert response.status_code == 200
    assert response.content == b"123456"
    assert response.headers["content-type"].startswith("application/pdf")
    assert "report%20with%20spaces.pdf" in response.headers["content-disposition"]


def test_fs_download_rejects_sensitive_files(client, tmp_path):
    target = tmp_path / ".env"
    target.write_text("SECRET=1")

    response = client.get("/api/fs/download", params={"path": str(target)})

    assert response.status_code == 403


@pytest.mark.parametrize("endpoint", ["/api/fs/read-text", "/api/fs/read-data-url", "/api/fs/download"])
@pytest.mark.parametrize("relative", [".env", "auth.json", "mcp-tokens/github.json"])
def test_fs_readers_reject_sensitive_paths(client, tmp_path, endpoint, relative):
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SECRET=1")

    response = client.get(endpoint, params={"path": str(target)})

    assert response.status_code == 403
    assert "SECRET" not in response.text


def test_fs_list_hides_sensitive_entries(client, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("SECRET=1")
    (root / "auth.json").write_text("{}")
    (root / "mcp-tokens").mkdir()
    (root / "notes.txt").write_text("ok")

    response = client.get("/api/fs/list", params={"path": str(root)})

    assert response.status_code == 200
    assert [entry["name"] for entry in response.json()["entries"]] == ["notes.txt"]


_HERMES_CREDENTIAL_RELS = [
    "vault/vault.key",
    "vault/vault.json.enc",
    "browser-profile/default/Cookies",
    "sessions/abc123.jsonl",
    "state.db",
    "state.db-wal",
    "kanban.db",
    "kanban/boards/board1/kanban.db",
    "skills/.hub/cache.json",
]


@pytest.mark.parametrize("endpoint", ["/api/fs/read-text", "/api/fs/read-data-url", "/api/fs/download"])
@pytest.mark.parametrize("relative", _HERMES_CREDENTIAL_RELS)
def test_fs_readers_reject_hermes_credential_paths(client, tmp_path, monkeypatch, endpoint, relative):
    """#57505 follow-up: the credential trees the canonical read/delivery guards deny under a
    Hermes home (vault/, browser-profile/, sessions/, skills/.hub, the state.db and kanban.db
    SQLite stores) must be unreadable through every fs reader, not just the agent file tools."""
    hermes_home = tmp_path / "hermes"
    target = hermes_home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SECRET=1")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    response = client.get(endpoint, params={"path": str(target)})

    assert response.status_code == 403
    assert "SECRET" not in response.text


def test_fs_readers_reject_credentials_in_a_sibling_profile_home(client, tmp_path, monkeypatch):
    """The shared Hermes root covers every ``profiles/<name>`` home: a credential tree in a
    NON-active profile must be denied too."""
    profile_home = tmp_path / "hermes" / "profiles" / "other"
    target = profile_home / "vault" / "vault.key"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SECRET=1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    response = client.get("/api/fs/download", params={"path": str(target)})

    assert response.status_code == 403


@pytest.mark.parametrize("relative", ["vault/notes.md", "sessions/log.txt", "state.db"])
def test_fs_readers_do_not_overblock_common_names_outside_hermes(client, tmp_path, monkeypatch, relative):
    """The scoped names are ordinary directory/file names: outside every Hermes root the same
    paths must stay readable, or the denylist would break browsing e.g. ~/Documents/vault."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    target = tmp_path / "elsewhere" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ordinary user file")

    response = client.get("/api/fs/download", params={"path": str(target)})

    assert response.status_code == 200


def test_fs_list_hides_hermes_credential_entries(client, tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    for rel in ("vault", "sessions", "state.db", "notes.txt"):
        target = hermes_home / rel
        if rel == "notes.txt":
            target.write_text("ok")
        elif "." in rel:
            target.write_text("SECRET=1")
        else:
            target.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    response = client.get("/api/fs/list", params={"path": str(hermes_home)})

    assert response.status_code == 200
    assert [entry["name"] for entry in response.json()["entries"]] == ["notes.txt"]


def test_sensitive_sets_cover_the_canonical_credential_names():
    """Drift pin: every tree name the canonical guards deny must be covered by
    one of the dashboard denylists, so the mirror cannot silently lag again:
    agent.file_safety._READ_DENIED_DIRS, _CREDENTIAL_FILE_NAMES and
    _BLOCKED_PROJECT_ENV_BASENAMES, the delivery guard's _ROOT_CREDENTIAL_PATHS,
    and skills/.hub."""
    from agent.file_safety import (
        _BLOCKED_PROJECT_ENV_BASENAMES, _CREDENTIAL_FILE_NAMES, _READ_DENIED_DIRS,
    )
    from gateway.platforms.base import _ROOT_CREDENTIAL_PATHS
    from hermes_cli import web_server_files as server_files

    dir_names = (
        server_files._SENSITIVE_MANAGED_DIR_NAMES
        | {part for rel in server_files._HERMES_SCOPED_DIR_RELS for part in rel}
    )
    file_names = (
        server_files._SENSITIVE_MANAGED_FILE_BASENAMES | server_files._HERMES_SCOPED_FILE_BASENAMES
    )

    for subdir, _dir_msg, _file_msg in _READ_DENIED_DIRS:
        assert subdir in dir_names, subdir
    for rel in _ROOT_CREDENTIAL_PATHS:
        name = PurePosixPath(rel).name
        assert (server_files._is_sensitive_filename(name)
                or name in dir_names
                or name in file_names), rel
    for rel in _CREDENTIAL_FILE_NAMES:
        name = PurePosixPath(rel).name
        assert server_files._is_sensitive_filename(name), rel
    for name in _BLOCKED_PROJECT_ENV_BASENAMES:
        assert server_files._is_sensitive_filename(name), name
    assert ("skills", ".hub") in server_files._HERMES_SCOPED_DIR_RELS


@pytest.mark.parametrize("relative", _HERMES_CREDENTIAL_RELS + ["auth.json"])
def test_fs_write_text_rejects_sensitive_paths(client, tmp_path, monkeypatch, relative):
    """fs_write_text must enforce the same denylist as the readers: overwriting a
    credential store or planting inside a credential tree must 403 and leave the
    file untouched."""
    hermes_home = tmp_path / "hermes"
    target = hermes_home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ORIGINAL")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    response = client.post("/api/fs/write-text", json={"path": str(target), "content": "PWNED"})

    assert response.status_code == 403
    assert target.read_text() == "ORIGINAL"


def test_fs_write_text_enforces_the_canonical_write_guard(client, tmp_path, monkeypatch):
    """fs_write_text is a free-fs write primitive: it must consult the canonical
    write guard (agent.file_safety) so denied system/home credential paths and
    approval-gated paths fail closed. The guard's builders are patched to tmp
    paths so a regression can never clobber a real credential file."""
    denied = tmp_path / "denied.txt"
    denied.write_text("ORIGINAL")
    monkeypatch.setattr("agent.file_safety.build_write_denied_paths",
                        lambda home: {os.path.realpath(str(denied))})

    response = client.post("/api/fs/write-text", json={"path": str(denied), "content": "PWNED"})

    assert response.status_code == 403
    assert denied.read_text() == "ORIGINAL"


def test_fs_write_text_denies_inside_a_canonical_denied_prefix(client, tmp_path, monkeypatch):
    prefix = tmp_path / "credroot"
    prefix.mkdir()
    target = prefix / "key.txt"
    target.write_text("ORIGINAL")
    monkeypatch.setattr("agent.file_safety.build_write_denied_prefixes",
                        lambda home: [os.path.realpath(str(prefix)) + os.sep])

    response = client.post("/api/fs/write-text", json={"path": str(target), "content": "PWNED"})

    assert response.status_code == 403
    assert target.read_text() == "ORIGINAL"


def test_fs_write_text_fails_closed_on_approval_gated_paths(client, tmp_path, monkeypatch):
    """Approval-gated paths (canonical model: interactive tools prompt) must fail
    closed on the dashboard, which has no approval channel."""
    target = tmp_path / "approval.txt"
    target.write_text("ORIGINAL")
    monkeypatch.setattr("agent.file_safety.build_write_approval_paths",
                        lambda home: {os.path.realpath(str(target))})

    response = client.post("/api/fs/write-text", json={"path": str(target), "content": "PWNED"})

    assert response.status_code == 403
    assert target.read_text() == "ORIGINAL"


def test_fs_write_text_rejects_env_anywhere(client, tmp_path):
    """The dashboard denylist is stricter than canonical write here on purpose:
    an .env file must be unwritable wherever it sits, matching the read side."""
    target = tmp_path / "project" / ".env"
    target.parent.mkdir()
    target.write_text("SECRET=1")

    response = client.post("/api/fs/write-text", json={"path": str(target), "content": "SECRET=2"})

    assert response.status_code == 403
    assert target.read_text() == "SECRET=1"


def test_fs_write_text_allows_ordinary_paths(client, tmp_path, monkeypatch):
    """Over-block negative: a same-named dir/file outside every Hermes root and
    guard home stays writable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    target = tmp_path / "elsewhere" / "vault" / "note.txt"
    target.parent.mkdir(parents=True)

    response = client.post("/api/fs/write-text", json={"path": str(target), "content": "hello"})

    assert response.status_code == 200
    assert target.read_text() == "hello"


def test_fs_list_filters_sensitive_directory_contents(client, tmp_path, monkeypatch):
    """Listing a credential directory yields nothing exploitable: every child is
    filtered because the parent component is denied (same contract as the
    managed list of mcp-tokens/)."""
    vault = tmp_path / "hermes" / "vault"
    vault.mkdir(parents=True)
    (vault / "vault.key").write_text("SECRET")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    response = client.get("/api/fs/list", params={"path": str(vault)})

    assert response.status_code == 200
    assert response.json()["entries"] == []


def test_fs_download_rejects_symlink_into_sensitive_tree(client, tmp_path, monkeypatch):
    """Resolution happens before the guard, so a symlink planted outside the
    Hermes root cannot launder a credential path past it."""
    real = tmp_path / "hermes" / "state.db"
    real.parent.mkdir(parents=True)
    real.write_text("SECRET")
    link = tmp_path / "innocent.db"
    link.symlink_to(real)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    response = client.get("/api/fs/download", params={"path": str(link)})

    assert response.status_code == 403


def test_fs_download_rejects_case_variant_of_credential_tree(client, tmp_path, monkeypatch):
    tree = tmp_path / "hermes" / "VAULT"
    tree.mkdir(parents=True)
    (tree / "vault.KEY").write_text("SECRET")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    response = client.get("/api/fs/download", params={"path": str(tree / "vault.KEY")})

    assert response.status_code == 403


def test_fs_endpoints_reject_nt_namespace_paths(client):
    """NT/device-namespace prefixes must be refused on the raw string: resolving
    \\\\?\\UNC\\host\\share can trigger outbound SMB auth before any deny fires."""
    for endpoint in ("/api/fs/download", "/api/fs/read-text", "/api/fs/list"):
        response = client.get(endpoint, params={"path": "\\\\?\\UNC\\attacker.example\\share\\x"})
        assert response.status_code == 400, endpoint
    response = client.post(
        "/api/fs/write-text",
        json={"path": "\\??\\GLOBALROOT\\Device\\x", "content": "x"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("path", [
    "\\\\attacker.example\\share\\x",
    "//attacker.example/share/x",
    "file://attacker.example/share/x",
])
def test_fs_endpoints_reject_unc_and_remote_file_urls(client, path):
    """Bare UNC input and the ``file:`` netloc unwrap both reach ``resolve()`` as
    ``\\\\host\\share`` on Windows, an outbound SMB auth (NTLM leak) the raw
    NT check alone cannot catch, since the unwrap rewrites the string after it."""
    for endpoint in ("/api/fs/download", "/api/fs/read-text", "/api/fs/list"):
        response = client.get(endpoint, params={"path": path})
        assert response.status_code == 400, (endpoint, path)


def test_fs_file_url_for_local_path_still_works(client, tmp_path):
    target = tmp_path / "ok.txt"
    target.write_text("fine")

    response = client.get("/api/fs/read-text", params={"path": target.as_uri()})

    assert response.status_code == 200


def test_managed_routes_reject_nt_namespace_paths(client):
    """``_path_text`` carries the same raw-string namespace rejection as
    ``_fs_path``: managed upload, delete, mkdir and reads must all 400."""
    path = "\\\\?\\UNC\\attacker.example\\share\\x"
    assert client.get("/api/files", params={"path": path}).status_code == 400
    assert client.get("/api/files/read", params={"path": path}).status_code == 400
    assert client.request("DELETE", "/api/files", json={"path": path}).status_code == 400
    assert client.post("/api/files/mkdir", json={"path": path}).status_code == 400
    assert client.post(
        "/api/files/upload",
        json={"path": path, "data_url": "data:text/plain;base64,eA=="},
    ).status_code == 400


def test_scoped_check_is_case_insensitive_on_the_root_prefix():
    """On case-insensitive filesystems (default macOS APFS) ``resolve()`` keeps
    the caller's typed case: a case-variant root prefix must still match, or the
    whole scoped denylist is bypassed on exactly that platform."""
    from hermes_cli import web_server_files as server_files

    roots = [Path("/Users/dev/.hermes")]
    target = Path("/users/dev/.hermes/vault/vault.key")
    assert server_files._is_sensitive_path(target, roots)


def test_scoped_names_stay_denied_below_deeper_dirs_outside_first_level(client, tmp_path, monkeypatch):
    """Anchored semantics: the scoped names deny only the canonical positions
    (<home>/<name>); a vault/ or sessions/ nested deeper under the Hermes home
    is ordinary user data and must stay readable."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for rel in ("plugins/foo/sessions/log.txt", "backups/old/vault/notes.md", "plugins/foo/state.db"):
        target = hermes_home / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("plugin data")

        response = client.get("/api/fs/download", params={"path": str(target)})

        assert response.status_code == 200, rel


def test_managed_delete_refuses_credential_tree_ancestors(client, tmp_path, monkeypatch):
    """Recursive delete inspects only the target, so deleting a CONTAINER of a
    credential store (the Hermes home itself, or any ancestor) must 403 rather
    than rmtree the denied trees."""
    hermes_home = tmp_path / "hermes"
    (hermes_home / "vault").mkdir(parents=True)
    (hermes_home / "vault" / "vault.key").write_text("SECRET")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    for doomed in (hermes_home, tmp_path):
        response = client.request(
            "DELETE", "/api/files", json={"path": str(doomed), "recursive": True}
        )
        assert response.status_code == 403, doomed
    assert (hermes_home / "vault" / "vault.key").read_text() == "SECRET"


def test_managed_delete_still_removes_an_ordinary_tree(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    tree = tmp_path / "scratch"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "f.txt").write_text("x")

    response = client.request(
        "DELETE", "/api/files", json={"path": str(tree), "recursive": True}
    )

    assert response.status_code == 200
    assert not tree.exists()


def test_managed_write_does_not_follow_a_dangling_leaf_symlink(client, tmp_path, monkeypatch):
    """A pre-planted symlink must not launder a write past the denylist: the
    canonical write guard realpaths the target, so links to canonically-denied
    stores are already caught; the hole was dashboard-only basenames
    (``config.yaml``) outside every Hermes/home scope, judged by the link's
    name instead of the pointee."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    secret = elsewhere / "config.yaml"

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    link = scratch / "plain.txt"
    link.symlink_to(secret)  # dangling: the write itself would create config.yaml

    response = client.post(
        "/api/files/upload",
        json={"path": str(link), "data_url": "data:text/plain;base64,cHduZWQ=", "overwrite": True},
    )

    assert response.status_code == 403
    assert not secret.exists()
    assert link.is_symlink()


@pytest.mark.parametrize("call", [
    lambda c, path: c.post("/api/files/upload",
                           json={"path": path, "data_url": "data:text/plain;base64,cHduZWQ="}),
    lambda c, path: c.post("/api/files/upload-stream",
                           files={"file": ("x.txt", b"pwned")},
                           data={"path": path, "overwrite": "true"}),
    lambda c, path: c.request("DELETE", "/api/files", json={"path": path}),
])
def test_managed_writes_reject_hermes_credential_paths(client, tmp_path, monkeypatch, call):
    """The managed write endpoints share the resolver seam: uploading over,
    deleting, or planting into a Hermes credential path must 403."""
    hermes_home = tmp_path / "hermes"
    target = hermes_home / "state.db"
    target.parent.mkdir(parents=True)
    target.write_text("ORIGINAL")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    response = call(client, str(target))

    assert response.status_code == 403
    assert target.read_text() == "ORIGINAL"


def test_managed_mkdir_rejects_sensitive_directory(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    response = client.post(
        "/api/files/mkdir", json={"path": str(tmp_path / "hermes" / "sessions" / "new")},
    )

    assert response.status_code == 403


def test_managed_endpoints_reject_sensitive_reads_and_lists(client, tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    (hermes_home / "vault").mkdir(parents=True)
    (hermes_home / "vault" / "vault.key").write_text("SECRET")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert client.get("/api/files/read",
                      params={"path": str(hermes_home / "vault" / "vault.key")}).status_code == 403
    listing = client.get("/api/files", params={"path": str(hermes_home / "vault")})
    assert listing.status_code == 200
    assert listing.json()["entries"] == []


def test_fs_endpoints_require_auth(tmp_path):
    client = TestClient(web_server.app)
    target = tmp_path / "secret.txt"
    target.write_text("secret")

    list_response = client.get("/api/fs/list", params={"path": str(tmp_path)})
    read_response = client.get("/api/fs/read-text", params={"path": str(target)})
    default_response = client.get("/api/fs/default-cwd")

    assert list_response.status_code == 401
    assert read_response.status_code == 401
    assert default_response.status_code == 401
