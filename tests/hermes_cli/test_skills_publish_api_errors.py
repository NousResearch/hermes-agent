"""Publishing must stop at the first failed GitHub stage (#37371)."""
import base64
import io
import json

import httpx
import pytest
from rich.console import Console

from hermes_cli.skills_hub import do_publish
from tools.skills_hub_github import GitHubAuth


@pytest.mark.parametrize("failure", ["fork_missing", "fork_empty", "fork_null", "fork_json_null", "fork_json_list", "fork_invalid_json", "branch", "upload", "network", "read"])
def test_publish_reports_failure_without_creating_pr(tmp_path, monkeypatch, failure):
    skill = tmp_path / "example"
    skill.mkdir()
    document = "---\nname: example\ndescription: A harmless example.\n---\n# Example\nHello.\n"
    (skill / "SKILL.md").write_text(document, encoding="utf-8")
    monkeypatch.setattr(GitHubAuth, "is_authenticated", lambda self: True)
    monkeypatch.setattr(GitHubAuth, "get_headers", lambda self: {})
    requests = []

    def respond(request):
        requests.append(request)
        path = request.url.path
        if path.endswith("/forks"):
            if failure == "fork_invalid_json":
                return httpx.Response(202, text="not JSON")
            if failure.startswith("fork_"):
                data = {"fork_missing": {}, "fork_empty": {"full_name": ""}, "fork_null": {"full_name": None}, "fork_json_null": None, "fork_json_list": []}[failure]
                return httpx.Response(202, text=json.dumps(data))
            return httpx.Response(202, json={"full_name": "contributor/project"})
        if path == "/repos/owner/project":
            return httpx.Response(200, json={"default_branch": "development"})
        if "/git/refs/heads/" in path:
            return httpx.Response(200, json={"object": {"sha": "abc"}})
        if path.endswith("/git/refs"):
            if failure == "network":
                raise httpx.ConnectError("fixture unavailable", request=request)
            if failure == "read":
                (skill / "SKILL.md").unlink()
                (skill / "SKILL.md").mkdir()
                (skill / "other.txt").write_text("content", encoding="utf8")
                original = type(skill).read_bytes
                def unreadable(path):
                    if path.name == "other.txt":
                        raise OSError("fixture unreadable")
                    return original(path)
                monkeypatch.setattr(type(skill), "read_bytes", unreadable)
            return httpx.Response(422 if failure == "branch" else 201, json={"message": "fixture"})
        if "/contents/" in path:
            return httpx.Response(403 if failure == "upload" else 201, json={"message": "fixture"})
        return httpx.Response(201, json={"html_url": "https://example.test/pr/1"})

    output = io.StringIO()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        for method in ("get", "post", "put"):
            monkeypatch.setattr(httpx, method, getattr(client, method))
        try:
            do_publish(str(skill), repo="owner/project", console=Console(file=output, width=160))
        except (KeyError, TypeError, AttributeError, OSError) as exc:
            pytest.fail(f"publish leaked {type(exc).__name__}: {exc}")
    assert not any(r.url.path.endswith("/pulls") for r in requests), output.getvalue()
    assert "PR created" not in output.getvalue()
    expected = "full_name" if failure.startswith("fork_") else ("branch" if failure in {"branch", "network"} else "upload")
    if failure == "fork_invalid_json":
        expected = "not valid JSON"
    assert expected in output.getvalue(), output.getvalue()
    if failure.startswith("fork_"):
        assert len(requests) == 1
    elif failure in {"branch", "network"}:
        assert not any("/contents/" in r.url.path for r in requests)


@pytest.mark.parametrize(
    "metadata, fork_branch, default_branch",
    [
        ({"default_branch": "development"}, "master", "development"),
        ({}, None, "main"),
        ({}, "master", "master"),
        ("http_error", "master", "master"),
        ("network_error", "master", "master"),
        ("invalid_json", "master", "master"),
        ({"default_branch": None}, "master", "master"),
        ({"default_branch": ""}, "master", "master"),
        ("network_error", None, "main"),
    ],
)
def test_publish_preserves_selected_repository_and_file_content(
    tmp_path, monkeypatch, metadata, fork_branch, default_branch
):
    document = "---\nname: example\ndescription: A harmless example.\n---\n# Example\nHello.\n"
    (tmp_path / "SKILL.md").write_text(document, encoding="utf8")
    monkeypatch.setattr(GitHubAuth, "is_authenticated", lambda self: True)
    monkeypatch.setattr(GitHubAuth, "get_headers", lambda self: {})
    requests = []
    def respond(request):
        requests.append(request)
        path = request.url.path
        if path.endswith("/forks"):
            return httpx.Response(202, json={"full_name": "contributor/project", "default_branch": fork_branch})
        if path == "/repos/owner/project":
            if metadata == "network_error":
                raise httpx.ConnectError("fixture unavailable", request=request)
            if metadata == "invalid_json":
                return httpx.Response(200, text="not JSON")
            if metadata == "http_error":
                return httpx.Response(503, json={"message": "unavailable"})
            return httpx.Response(200, json=metadata)
        if "/git/refs/heads/" in path:
            if not path.endswith("/" + default_branch):
                return httpx.Response(404, json={"message": "unknown branch"})
            return httpx.Response(200, json={"object": {"sha": "abc"}})
        return httpx.Response(201, json={"html_url": "https://example.test/pr/1"})
    output = io.StringIO()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        for method in ("get", "post", "put"):
            monkeypatch.setattr(httpx, method, getattr(client, method))
        do_publish(str(tmp_path), repo="owner/project", console=Console(file=output, width=160))
    assert "PR created: https://example.test/pr/1" in output.getvalue(), output.getvalue()
    upload = next(r for r in requests if r.method == "PUT")
    assert base64.b64decode(json.loads(upload.content)["content"]) == (tmp_path / "SKILL.md").read_bytes()
    pull = requests[-1]
    assert pull.url.path == "/repos/owner/project/pulls"
    assert json.loads(pull.content)["base"] == default_branch
    assert json.loads(pull.content)["head"] == "contributor:add-skill-example"
    assert "PR created: https://example.test/pr/1" in output.getvalue()
