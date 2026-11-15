"""A preview's MIME type and language come from the name the caller asked for, while the bytes and
the sensitive-path check stay on the file that name resolves to (follow-up to #153682)."""
import base64

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.web_routers import files as file_routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


HTML = b"<h1>Report</h1>"


def _assert_html_preview(endpoint, response):
    assert response.status_code == 200, response.text
    if endpoint.endswith("read-text"):
        assert response.json()["mimeType"] == "text/html"
        assert response.json()["language"] == "html"
        assert response.json()["text"] == HTML.decode()
    else:
        assert response.json()["dataUrl"] == "data:text/html;base64," + base64.b64encode(HTML).decode("ascii")


@pytest.mark.require_symlinks
@pytest.mark.parametrize("endpoint", ["/api/fs/read-text", "/api/fs/read-data-url"])
def test_a_host_link_previews_as_the_type_its_name_says(client, tmp_path, endpoint):
    """``report.html -> cache/7f24ab`` previews as HTML on this host, as it does over SSH."""
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "7f24ab").write_bytes(HTML)
    (tmp_path / "report.html").symlink_to(tmp_path / "cache" / "7f24ab")

    response = client.get(endpoint, params={"path": str(tmp_path / "report.html")})

    _assert_html_preview(endpoint, response)


@pytest.mark.require_symlinks
@pytest.mark.parametrize("endpoint", ["/api/fs/read-text", "/api/fs/read-data-url"])
def test_a_host_link_named_html_to_a_secret_is_still_refused(client, tmp_path, endpoint):
    """Typing by the visible name never lets a sensitive target through."""
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-secret\n")
    (tmp_path / "page.html").symlink_to(tmp_path / ".env")

    response = client.get(endpoint, params={"path": str(tmp_path / "page.html")})

    assert response.status_code == 403
    assert "sk-secret" not in response.text


@pytest.mark.require_symlinks
def test_a_host_link_without_an_extension_keeps_its_targets_type(client, tmp_path):
    """A name with nothing to say about type falls back to the file it resolves to."""
    (tmp_path / "page.html").write_bytes(HTML)
    (tmp_path / "latest").symlink_to(tmp_path / "page.html")

    response = client.get("/api/fs/read-text", params={"path": str(tmp_path / "latest")})

    assert response.status_code == 200
    assert response.json()["mimeType"] == "text/html"
    assert response.json()["language"] == "html"


@pytest.mark.require_symlinks
@pytest.mark.parametrize(
    ("link", "target", "field", "expected"),
    [
        ("report.draft", "report.html", "language", "html"),  # the link's extension names no type
        ("main.go", "cache/7f24ab", "language", "go"),  # it names a language only
        ("photo.png", "cache/7f24ab", "mimeType", "image/png"),  # it names a MIME type only
    ],
)
def test_a_host_link_previews_as_the_type_one_of_its_names_says(client, tmp_path, link, target, field, expected):
    (tmp_path / "cache").mkdir()
    (tmp_path / target).write_bytes(HTML)
    (tmp_path / link).symlink_to(tmp_path / target)

    response = client.get("/api/fs/read-text", params={"path": str(tmp_path / link)})

    assert response.status_code == 200
    assert response.json()[field] == expected


@pytest.mark.parametrize("endpoint", ["/api/fs/read-text", "/api/fs/read-data-url"])
@pytest.mark.parametrize("suffix", ["page.html#section", "page.h%74ml"])
def test_a_file_url_previews_as_the_type_of_the_path_it_names(client, tmp_path, endpoint, suffix):
    """The read decodes a ``file:`` URL (fragment dropped, %-escapes decoded once); the type does too."""
    (tmp_path / "page.html").write_bytes(HTML)

    response = client.get(endpoint, params={"path": (tmp_path / "x").as_uri()[:-1] + suffix})

    _assert_html_preview(endpoint, response)


class _FakeSshFs:
    def __init__(self, resolved):
        self.resolved = resolved
        self.reads = []

    def inspect_file(self, _path):
        return self.resolved, len(HTML)

    def read_bytes(self, path, *, max_bytes, read_limit=None):
        self.reads.append(path)
        return HTML, len(HTML), path


@pytest.mark.parametrize("endpoint", ["/api/fs/read-text", "/api/fs/read-data-url"])
def test_an_ssh_link_named_html_to_a_secret_is_refused_before_any_read(client, monkeypatch, endpoint):
    workspace = _FakeSshFs("/home/dev/.hermes/.env")
    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: workspace)

    response = client.get(endpoint, params={"path": "/srv/repos/page.html", "profile": "remote-dev"})

    assert response.status_code == 403
    assert workspace.reads == []


def test_an_ssh_link_without_an_extension_keeps_its_targets_type(client, monkeypatch):
    workspace = _FakeSshFs("/srv/repos/page.html")
    monkeypatch.setattr(file_routes, "_fs_backend", lambda profile=None: workspace)

    response = client.get("/api/fs/read-text", params={"path": "/srv/repos/latest", "profile": "remote-dev"})

    assert response.status_code == 200
    assert response.json()["mimeType"] == "text/html"
    assert workspace.reads == ["/srv/repos/page.html"]
