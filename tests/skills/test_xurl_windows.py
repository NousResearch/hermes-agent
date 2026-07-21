import sys

import yaml

from agent.skill_utils import parse_frontmatter, skill_matches_platform
from scripts import xurl_windows


def test_xurl_skill_is_available_on_windows(monkeypatch):
    content = (
        xurl_windows.Path("skills/social-media/xurl/SKILL.md")
        .read_text(encoding="utf-8")
    )
    frontmatter, _ = parse_frontmatter(content)

    monkeypatch.setattr(sys, "platform", "win32")

    assert "windows" in frontmatter["platforms"]
    assert skill_matches_platform(frontmatter)


def test_windows_shim_auth_status_without_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))

    rc = xurl_windows.main(["auth", "status"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "No apps registered" in captured.out
    assert "token" not in captured.out.lower()


def test_windows_shim_apps_add_prompts_for_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(xurl_windows.getpass, "getpass", lambda prompt: "prompt-secret")

    rc = xurl_windows.main(
        [
            "auth",
            "apps",
            "add",
            "test-app",
            "--client-id",
            "client-id",
            "--prompt-client-secret",
        ]
    )

    captured = capsys.readouterr()
    store = yaml.safe_load((tmp_path / ".xurl").read_text(encoding="utf-8"))
    assert rc == 0
    assert 'App "test-app" registered successfully.' in captured.out
    assert store["apps"]["test-app"]["client_secret"] == "prompt-secret"
    assert "prompt-secret" not in captured.out


def test_windows_shim_apps_add_allows_public_client_without_secret(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("HOME", str(tmp_path))

    def fail_getpass(prompt):
        raise AssertionError("public client registration should not prompt")

    monkeypatch.setattr(xurl_windows.getpass, "getpass", fail_getpass)

    rc = xurl_windows.main(["auth", "apps", "add", "test-app", "--client-id", "client-id"])

    captured = capsys.readouterr()
    store = yaml.safe_load((tmp_path / ".xurl").read_text(encoding="utf-8"))
    assert rc == 0
    assert 'App "test-app" registered successfully.' in captured.out
    assert store["apps"]["test-app"]["client_secret"] == ""


def test_windows_shim_public_client_token_exchange_omits_basic_auth(monkeypatch):
    observed = {}

    def fake_http_json(method, url, headers=None, form=None, **kwargs):
        observed["method"] = method
        observed["headers"] = headers
        observed["form"] = form
        return {"access_token": "access", "refresh_token": "refresh", "expires_in": 7200}

    monkeypatch.setattr(xurl_windows, "_http_json", fake_http_json)

    payload = xurl_windows._exchange_token(
        {"client_id": "client-id", "client_secret": ""},
        {"grant_type": "authorization_code", "code": "code", "code_verifier": "verifier"},
    )

    assert payload["access_token"] == "access"
    assert "Authorization" not in observed["headers"]
    assert observed["form"]["client_id"] == "client-id"


def test_windows_shim_uses_minimum_posting_scopes():
    assert xurl_windows.OAUTH2_SCOPES == [
        "tweet.read",
        "tweet.write",
        "users.read",
        "offline.access",
    ]


def test_windows_shim_rejects_placeholder_values(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(xurl_windows.getpass, "getpass", lambda prompt: "client-secret")

    rc = xurl_windows.main(
        ["auth", "apps", "add", "test-app", "--client-id", "YOUR_CLIENT_ID"]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "placeholder" in captured.err
    assert not (tmp_path / ".xurl").exists()


def test_windows_shim_apps_remove(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = {
        "default_app": "test-app",
        "apps": {
            "test-app": {
                "client_id": "client-id",
                "client_secret": "client-secret",
                "oauth2_tokens": {},
            }
        },
    }
    (tmp_path / ".xurl").write_text(yaml.safe_dump(store), encoding="utf-8")

    rc = xurl_windows.main(["auth", "apps", "remove", "test-app"])

    captured = capsys.readouterr()
    saved = yaml.safe_load((tmp_path / ".xurl").read_text(encoding="utf-8"))
    assert rc == 0
    assert 'App "test-app" removed.' in captured.out
    assert saved["apps"] == {}
    assert saved["default_app"] == ""


def test_windows_shim_post_uses_store_without_printing_token(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    token_value = "fake-access-token"
    store = {
        "default_app": "test-app",
        "apps": {
            "test-app": {
                "client_id": "client-id",
                "client_secret": "client-secret",
                "oauth2_tokens": {
                    "alice": {
                        "type": "oauth2",
                        "oauth2": {
                            "access_token": token_value,
                            "refresh_token": "fake-refresh-token",
                            "expiration_time": 4102444800,
                        },
                    }
                },
                "default_user": "alice",
            }
        },
    }
    (tmp_path / ".xurl").write_text(yaml.safe_dump(store), encoding="utf-8")

    observed = {}

    def fake_post(endpoint, body, access_token):
        observed["endpoint"] = endpoint
        observed["body"] = body
        observed["access_token"] = access_token
        return {"data": {"id": "123", "text": body["text"]}}

    monkeypatch.setattr(xurl_windows, "_post_json", fake_post)

    rc = xurl_windows.main(["post", "hello from hermes"])

    captured = capsys.readouterr()
    assert rc == 0
    assert observed == {
        "endpoint": "/2/tweets",
        "body": {"text": "hello from hermes"},
        "access_token": token_value,
    }
    assert "hello from hermes" in captured.out
    assert token_value not in captured.out
    assert "fake-refresh-token" not in captured.out
