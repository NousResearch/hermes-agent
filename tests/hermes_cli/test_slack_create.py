"""``hermes slack create``: one App Configuration token creates the app, re-runs sync it.

The Slack Web API is stubbed at the HTTP boundary (``httpx.post``); everything else (rotation
bookkeeping, .env and config.yaml writes, manifest generation) is the real code against the
isolated test HERMES_HOME.
"""
import argparse
import json

from hermes_cli import slack_create
from hermes_cli.config import get_config_path, get_env_path, load_env, read_raw_config, save_config
from hermes_cli.slack_cli import _build_full_manifest


class _FakeSlack:
    """Minimal Slack Web API: records each call's form fields, answers per method."""

    def __init__(self, exported=None):
        self.calls = []
        self.exported = exported
        self.rotations = 0

    def post(self, url, data=None, timeout=None):
        method = url.rsplit("/", 1)[-1]
        self.calls.append((method, dict(data or {})))
        if method == "tooling.tokens.rotate":
            self.rotations += 1
            body = {"ok": True, "token": f"xoxe.xoxp-access-{self.rotations}",
                    "refresh_token": f"xoxe-1-rotated-{self.rotations}"}
        elif method == "apps.manifest.create":
            body = {"ok": True, "app_id": "A0NEWAPP01",
                    "credentials": {"client_secret": "never-stored"}}
        elif method == "apps.manifest.export":
            body = {"ok": True, "manifest": self.exported}
        else:
            body = {"ok": True}
        return _Response(body)

    def methods(self):
        return [m for m, _ in self.calls]

    def fields(self, method):
        return next(f for m, f in self.calls if m == method)


class _Response:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def _answers(monkeypatch, *values):
    queue = list(values)
    seen = []

    def fake_prompt(question, default=None, password=False):
        seen.append((question, password))
        return queue.pop(0) if queue else ""

    monkeypatch.setattr(slack_create, "prompt", fake_prompt)
    return seen


def _args(**kw):
    return argparse.Namespace(app_id=kw.get("app_id"), name=None, description=None,
                              messaging=kw.get("messaging", "agent"))


def test_create_rotates_creates_full_manifest_and_keeps_secrets_out_of_config(
        _isolate_hermes_home, monkeypatch):
    slack = _FakeSlack()
    monkeypatch.setattr(slack_create.httpx, "post", slack.post)
    seen = _answers(monkeypatch, "xoxe-1-pasted", "xoxb-bot-123", "xapp-1-app-456")

    assert slack_create.slack_create_command(_args()) == 0

    assert slack.methods() == ["tooling.tokens.rotate", "apps.manifest.create"]
    assert slack.fields("tooling.tokens.rotate") == {"refresh_token": "xoxe-1-pasted"}
    created = slack.fields("apps.manifest.create")
    assert created["token"] == "xoxe.xoxp-access-1"
    assert json.loads(created["manifest"]) == _build_full_manifest(
        "Hermes", "Your Hermes agent on Slack", messaging_experience="agent")
    # Every secret prompt is masked.
    assert seen and all(password for _question, password in seen)

    env = load_env()
    assert env[slack_create.REFRESH_TOKEN_ENV] == "xoxe-1-rotated-1"
    assert env["SLACK_BOT_TOKEN"] == "xoxb-bot-123"
    assert env["SLACK_APP_TOKEN"] == "xapp-1-app-456"
    assert read_raw_config()["slack"]["app_id"] == "A0NEWAPP01"
    config_text = get_config_path().read_text(encoding="utf-8-sig")
    for secret in ("xox", "never-stored"):
        assert secret not in config_text
    assert "never-stored" not in get_env_path().read_text(encoding="utf-8-sig")


def test_rerun_with_stored_app_updates_manifest_and_preserves_user_edits(
        _isolate_hermes_home, monkeypatch):
    save_config({"slack": {"app_id": "A0EXISTING"}})
    from hermes_cli.config import save_env_value
    save_env_value(slack_create.REFRESH_TOKEN_ENV, "xoxe-1-stored")
    save_env_value("SLACK_BOT_TOKEN", "xoxb-kept")
    save_env_value("SLACK_APP_TOKEN", "xapp-kept")
    stale = _build_full_manifest("My Bot", "custom words", messaging_experience="assistant")
    stale["features"]["slash_commands"] = [
        {"command": "/old-gone", "description": "removed upstream", "url": "https://x"}]
    stale["oauth_config"]["scopes"]["bot"] = ["chat:write", "custom:scope"]
    slack = _FakeSlack(exported=stale)
    monkeypatch.setattr(slack_create.httpx, "post", slack.post)
    seen = _answers(monkeypatch)

    assert slack_create.slack_create_command(_args()) == 0

    assert seen == []  # stored refresh token, stored bot/app tokens: nothing to ask
    assert slack.methods() == [
        "tooling.tokens.rotate", "apps.manifest.export", "apps.manifest.update"]
    assert slack.fields("tooling.tokens.rotate") == {"refresh_token": "xoxe-1-stored"}
    updated = slack.fields("apps.manifest.update")
    assert updated["app_id"] == "A0EXISTING"
    manifest = json.loads(updated["manifest"])
    current = _build_full_manifest("Hermes", "", messaging_experience="assistant")
    assert manifest["features"]["slash_commands"] == current["features"]["slash_commands"]
    assert manifest["display_information"] == stale["display_information"]
    assert "custom:scope" in manifest["oauth_config"]["scopes"]["bot"]
    assert set(current["oauth_config"]["scopes"]["bot"]) <= set(
        manifest["oauth_config"]["scopes"]["bot"])
    assert load_env()[slack_create.REFRESH_TOKEN_ENV] == "xoxe-1-rotated-1"
