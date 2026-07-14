"""Tests for hermes_cli/webhook.py — webhook subscription CLI."""

import json
import os
import pytest
import stat
from argparse import Namespace

from hermes_cli.webhook import (
    _load_subscriptions,
    _save_subscriptions,
    _setup_hint,
    _subscriptions_path,
    webhook_command,
)


def test_setup_hint_defaults_to_loopback_and_documents_public_consent():
    hint = _setup_hint()
    assert 'host: "127.0.0.1"' in hint
    assert "allow_public_bind: true" in hint


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Default: webhooks enabled (most tests need this)
    monkeypatch.setattr(
        "hermes_cli.webhook._is_webhook_enabled", lambda: True
    )


def _make_args(**kwargs):
    defaults = {
        "webhook_action": None,
        "name": "",
        "prompt": "",
        "events": "",
        "description": "",
        "skills": "",
        "deliver": "log",
        "deliver_chat_id": "",
        "secret": "",
        "payload": "",
        "script": "",
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestSubscribe:
    def test_basic_create(self, capsys):
        webhook_command(_make_args(webhook_action="subscribe", name="test-hook"))
        out = capsys.readouterr().out
        assert "Created" in out
        assert "/webhooks/test-hook" in out
        subs = _load_subscriptions()
        assert "test-hook" in subs

    def test_with_options(self, capsys):
        webhook_command(_make_args(
            webhook_action="subscribe",
            name="gh-issues",
            events="issues,pull_request",
            prompt="Issue: {issue.title}",
            deliver="telegram",
            deliver_chat_id="12345",
            description="Watch GitHub",
        ))
        subs = _load_subscriptions()
        route = subs["gh-issues"]
        assert route["events"] == ["issues", "pull_request"]
        assert route["prompt"] == "Issue: {issue.title}"
        assert route["deliver"] == "telegram"
        assert route["deliver_extra"] == {"chat_id": "12345"}

    def test_custom_secret(self):
        webhook_command(_make_args(
            webhook_action="subscribe", name="s", secret="my-secret"
        ))
        assert _load_subscriptions()["s"]["secret"] == "my-secret"

    def test_script_option_is_persisted(self):
        webhook_command(_make_args(
            webhook_action="subscribe", name="s", script="todoist_filter.py"
        ))
        assert _load_subscriptions()["s"]["script"] == "todoist_filter.py"

    def test_auto_secret(self):
        webhook_command(_make_args(webhook_action="subscribe", name="s"))
        secret = _load_subscriptions()["s"]["secret"]
        assert len(secret) > 20

    def test_update(self, capsys):
        webhook_command(_make_args(webhook_action="subscribe", name="x", prompt="v1"))
        webhook_command(_make_args(webhook_action="subscribe", name="x", prompt="v2"))
        out = capsys.readouterr().out
        assert "Updated" in out
        assert _load_subscriptions()["x"]["prompt"] == "v2"

    def test_invalid_name(self, capsys):
        webhook_command(_make_args(webhook_action="subscribe", name="bad name!"))
        out = capsys.readouterr().out
        assert "Error" in out or "Invalid" in out
        assert _load_subscriptions() == {}


class TestList:
    def test_empty(self, capsys):
        webhook_command(_make_args(webhook_action="list"))
        out = capsys.readouterr().out
        assert "No dynamic" in out

    def test_with_entries(self, capsys):
        webhook_command(_make_args(webhook_action="subscribe", name="a"))
        webhook_command(_make_args(webhook_action="subscribe", name="b"))
        capsys.readouterr()  # clear
        webhook_command(_make_args(webhook_action="list"))
        out = capsys.readouterr().out
        assert "2 webhook" in out
        assert "a" in out
        assert "b" in out


class TestRemove:
    def test_remove_existing(self, capsys):
        webhook_command(_make_args(webhook_action="subscribe", name="temp"))
        webhook_command(_make_args(webhook_action="remove", name="temp"))
        out = capsys.readouterr().out
        assert "Removed" in out
        assert _load_subscriptions() == {}

    def test_remove_nonexistent(self, capsys):
        webhook_command(_make_args(webhook_action="remove", name="nope"))
        out = capsys.readouterr().out
        assert "No subscription" in out

    def test_selective_remove(self):
        webhook_command(_make_args(webhook_action="subscribe", name="keep"))
        webhook_command(_make_args(webhook_action="subscribe", name="drop"))
        webhook_command(_make_args(webhook_action="remove", name="drop"))
        subs = _load_subscriptions()
        assert "keep" in subs
        assert "drop" not in subs


class TestPersistence:
    def test_file_written(self):
        webhook_command(_make_args(webhook_action="subscribe", name="persist"))
        path = _subscriptions_path()
        assert path.exists()
        data = json.loads(path.read_text())
        assert "persist" in data

    def test_corrupted_file(self):
        path = _subscriptions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("broken{{{")
        assert _load_subscriptions() == {}

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are platform-specific")
    def test_save_creates_secret_file_owner_only_under_permissive_umask(self):
        old_umask = os.umask(0o022)
        try:
            _save_subscriptions({"demo": {"secret": "TOPSECRET", "prompt": "x"}})
        finally:
            os.umask(old_umask)

        path = _subscriptions_path()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert "TOPSECRET" in path.read_text(encoding="utf-8")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are platform-specific")
    def test_save_narrows_existing_broad_secret_file_mode(self):
        # Simulate a pre-existing 0o644 file from before this hardening landed.
        path = _subscriptions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"old": {"secret": "stale", "prompt": "x"}}))
        path.chmod(0o644)

        _save_subscriptions({"demo": {"secret": "FRESH", "prompt": "x"}})

        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert "FRESH" in path.read_text(encoding="utf-8")


class TestWebhookEnabledGate:
    def test_blocks_when_disabled(self, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.webhook._is_webhook_enabled", lambda: False)
        webhook_command(_make_args(webhook_action="subscribe", name="blocked"))
        out = capsys.readouterr().out
        assert "not enabled" in out.lower()
        assert "hermes gateway setup" in out
        assert _load_subscriptions() == {}

    def test_blocks_list_when_disabled(self, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.webhook._is_webhook_enabled", lambda: False)
        webhook_command(_make_args(webhook_action="list"))
        out = capsys.readouterr().out
        assert "not enabled" in out.lower()

    def test_allows_when_enabled(self, capsys):
        # _is_webhook_enabled already patched to True by autouse fixture
        webhook_command(_make_args(webhook_action="subscribe", name="allowed"))
        out = capsys.readouterr().out
        assert "Created" in out
        assert "allowed" in _load_subscriptions()

    def test_real_check_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.webhook._get_webhook_config",
            lambda: {},
        )
        monkeypatch.setattr(
            "hermes_cli.webhook._is_webhook_enabled",
            lambda: bool({}.get("enabled")),
        )
        import hermes_cli.webhook as wh_mod
        assert wh_mod._is_webhook_enabled() is False

    def test_real_check_enabled(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.webhook._is_webhook_enabled",
            lambda: True,
        )
        import hermes_cli.webhook as wh_mod
        assert wh_mod._is_webhook_enabled() is True
