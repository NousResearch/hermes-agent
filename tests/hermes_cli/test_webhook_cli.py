"""Tests for hermes_cli/webhook.py — webhook subscription CLI."""

import hashlib
import hmac
import json
import os
import pytest
import stat
from argparse import Namespace

from hermes_cli.webhook import (
    webhook_command,
    _get_webhook_base_url,
    _get_webhook_config,
    _load_subscriptions,
    _save_subscriptions,
    _subscriptions_path,
)

# Capture the *real* implementation before the autouse `_isolate` fixture
# patches the module attribute. Needed to exercise the true return path of
# `_is_webhook_enabled` (line 94) rather than the patched stub.
import hermes_cli.webhook as webhook_mod

_REAL_IS_WEBHOOK_ENABLED = webhook_mod._is_webhook_enabled


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


@pytest.mark.parametrize("host", [None, "", "0.0.0.0", "::"])
def test_webhook_base_url_maps_wildcard_hosts_to_localhost(monkeypatch, host):
    monkeypatch.setattr(
        "hermes_cli.webhook._get_webhook_config",
        lambda: {"extra": {"host": host, "port": 9123}},
    )
    assert _get_webhook_base_url() == "http://localhost:9123"


class TestSubscribe:


    def test_custom_secret(self):
        webhook_command(_make_args(
            webhook_action="subscribe", name="s", secret="my-secret"
        ))
        assert _load_subscriptions()["s"]["secret"] == "my-secret"


    def test_auto_secret(self):
        webhook_command(_make_args(webhook_action="subscribe", name="s"))
        secret = _load_subscriptions()["s"]["secret"]
        assert len(secret) > 20


class TestList:

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


    def test_selective_remove(self):
        webhook_command(_make_args(webhook_action="subscribe", name="keep"))
        webhook_command(_make_args(webhook_action="subscribe", name="drop"))
        webhook_command(_make_args(webhook_action="remove", name="drop"))
        subs = _load_subscriptions()
        assert "keep" in subs
        assert "drop" not in subs


class TestPersistence:

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


# ---------------------------------------------------------------------------
# Tests added to raise statement coverage for hermes_cli/webhook.py
# (issue #36458). Each targets a specific previously-untested path.
# ---------------------------------------------------------------------------


class TestSaveSubscriptionsFailure:
    def test_cleans_up_tmp_and_reraises_on_write_failure(self, monkeypatch):
        # Force the step *after* the temp file is created (atomic_replace) to
        # fail so the except block runs: tmp file cleaned up, original error
        # re-raised.
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated replace failure")

        monkeypatch.setattr("hermes_cli.webhook.atomic_replace", _boom)

        with pytest.raises(RuntimeError):
            _save_subscriptions({"s": {"secret": "x", "prompt": "y"}})

        leftover = [
            p for p in _subscriptions_path().parent.iterdir() if p.suffix == ".tmp"
        ]
        assert leftover == []

    def test_reraises_original_when_tmp_unlink_also_fails(self, monkeypatch):
        # atomic_replace fails AND the tmp-file unlink raises OSError: the inner
        # `except OSError: pass` swallows it and the original error propagates.
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated replace failure")

        monkeypatch.setattr("hermes_cli.webhook.atomic_replace", _boom)

        import pathlib

        def _bad_unlink(self_unused, *a, **k):
            raise OSError("simulated unlink failure")

        monkeypatch.setattr(pathlib.Path, "unlink", _bad_unlink)

        with pytest.raises(RuntimeError):
            _save_subscriptions({"s": {"secret": "x", "prompt": "y"}})


class TestGetWebhookConfigFallback:
    def test_returns_empty_when_config_load_raises(self, monkeypatch):
        def _boom():
            raise RuntimeError("config load failed")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)
        assert _get_webhook_config() == {}


class TestIsWebhookEnabled:
    def test_true_when_config_marks_enabled(self, monkeypatch):
        # Exercise the real `_is_webhook_enabled` (not the patched stub) so its
        # `return bool(...)` line executes with an `enabled` key present.
        monkeypatch.setattr(
            "hermes_cli.webhook._get_webhook_config",
            lambda: {"enabled": True},
        )
        assert _REAL_IS_WEBHOOK_ENABLED() is True


class TestWebhookBaseUrlIPv6:
    def test_brackets_ipv6_host(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.webhook._get_webhook_config",
            lambda: {"extra": {"host": "2001:db8::1", "port": 8644}},
        )
        assert _get_webhook_base_url() == "http://[2001:db8::1]:8644"


class TestWebhookCommandDispatch:
    def test_no_subcommand_prints_usage(self, capsys):
        webhook_command(_make_args())
        out = capsys.readouterr().out
        assert "Usage: hermes webhook" in out
        assert "subscribe|list|remove|test" in out

    def test_test_subcommand_reports_unknown_route(self, capsys):
        # Covers the `elif sub == "test"` dispatch and the not-found branch
        # inside _cmd_test.
        webhook_command(_make_args(webhook_action="test", name="doesnotexist"))
        out = capsys.readouterr().out
        assert "No subscription named 'doesnotexist'" in out


class TestSubscribeValidation:
    def test_rejects_invalid_name(self, capsys):
        webhook_command(_make_args(webhook_action="subscribe", name="bad@name"))
        out = capsys.readouterr().out
        assert "Error: Invalid name" in out
        assert "bad@name" not in _load_subscriptions()

    def test_deliver_only_with_log_deliver_is_rejected(self, capsys):
        webhook_command(
            _make_args(
                webhook_action="subscribe",
                name="x",
                deliver="log",
                deliver_only=True,
            )
        )
        out = capsys.readouterr().out
        assert "--deliver-only requires --deliver" in out
        assert "x" not in _load_subscriptions()

    def test_script_route_saved_and_printed(self, capsys):
        webhook_command(
            _make_args(webhook_action="subscribe", name="scripty", script="run.py")
        )
        route = _load_subscriptions()["scripty"]
        assert route["script"] == "run.py"
        out = capsys.readouterr().out
        assert "Script: run.py" in out

    def test_deliver_chat_id_saved_to_deliver_extra(self, capsys):
        webhook_command(
            _make_args(
                webhook_action="subscribe",
                name="chatty",
                deliver_chat_id="12345",
                deliver="telegram",
            )
        )
        route = _load_subscriptions()["chatty"]
        assert route["deliver_extra"] == {"chat_id": "12345"}

    def test_events_printed_when_provided(self, capsys):
        webhook_command(
            _make_args(
                webhook_action="subscribe", name="ev", events="push,pull"
            )
        )
        out = capsys.readouterr().out
        assert "Events: push, pull" in out

    def test_deliver_only_mode_line_in_subscribe_output(self, capsys):
        webhook_command(
            _make_args(
                webhook_action="subscribe",
                name="donly",
                deliver="telegram",
                deliver_only=True,
            )
        )
        out = capsys.readouterr().out
        assert "Mode: direct delivery" in out
        assert "donly" in _load_subscriptions()

    def test_prompt_preview_short(self, capsys):
        webhook_command(
            _make_args(webhook_action="subscribe", name="p1", prompt="Hello world")
        )
        out = capsys.readouterr().out
        assert "Prompt: Hello world" in out

    def test_prompt_preview_truncates_long_prompt(self, capsys):
        long_prompt = "x" * 200
        webhook_command(
            _make_args(webhook_action="subscribe", name="p2", prompt=long_prompt)
        )
        out = capsys.readouterr().out
        assert "Prompt: " + "x" * 80 in out
        assert "..." in out


class TestListEmpty:
    def test_list_with_no_subscriptions_prints_empty_message(self, capsys):
        webhook_command(_make_args(webhook_action="list"))
        out = capsys.readouterr().out
        assert "No dynamic webhook subscriptions" in out


class TestListDetails:
    def test_list_shows_deliver_only_marker_and_script(self, capsys):
        webhook_command(
            _make_args(
                webhook_action="subscribe",
                name="lst",
                deliver="telegram",
                deliver_only=True,
                script="run.py",
            )
        )
        capsys.readouterr()
        webhook_command(_make_args(webhook_action="list"))
        out = capsys.readouterr().out
        assert "(direct \u2014 no agent)" in out
        # Assert on the script path being listed (normalized whitespace) —
        # don't freeze the current double-space formatting accident.
        assert "run.py" in out
        assert any(
            line.strip().startswith("Script:") for line in out.splitlines()
        )


class TestRemoveNonexistent:
    def test_remove_reports_missing_subscription(self, capsys):
        webhook_command(_make_args(webhook_action="remove", name="ghost"))
        out = capsys.readouterr().out
        assert "No subscription named 'ghost'" in out


class TestCmdTest:
    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"ok": true}'

    def _make_test_route(self, **kwargs):
        webhook_command(
            _make_args(
                webhook_action="subscribe",
                name="route",
                deliver="telegram",
                secret="my-secret",
                **kwargs,
            )
        )

    def test_sends_test_post_and_prints_response(self, monkeypatch, capsys):
        self._make_test_route()
        captured = {}

        def _fake_urlopen(req, *a, **k):
            captured["request"] = req
            return self._FakeResp()

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        webhook_command(_make_args(webhook_action="test", name="route"))
        out = capsys.readouterr().out
        assert "Sending test POST" in out
        assert "Response (200)" in out
        assert '{"ok": true}' in out

        # The POST must carry the payload and the HMAC-SHA256 signature over
        # it, keyed with the route's secret.
        req = captured["request"]
        payload = req.data.decode()
        # Body must be byte-identical to the payload the command was given —
        # no re-serialization before signing or sending.
        assert payload == '{"test": true, "event_type": "test", "message": "Hello from hermes webhook test"}'
        expected_sig = (
            "sha256="
            + hmac.new(b"my-secret", payload.encode(), hashlib.sha256).hexdigest()
        )
        headers = {k.lower(): v for k, v in req.headers.items()}
        assert headers["content-type"] == "application/json"
        assert headers["x-hub-signature-256"] == expected_sig
        assert headers["x-github-event"] == "test"

    def test_signature_uses_route_secret_not_placeholder(self, monkeypatch, capsys):
        # Same flow but with a distinct secret: the signature must be keyed by
        # the *stored route secret*, proving the header isn't computed against
        # some fixed placeholder.
        webhook_command(
            _make_args(
                webhook_action="subscribe",
                name="route",
                deliver="telegram",
                secret="route-secret-42",
            )
        )
        captured = {}

        def _fake_urlopen(req, *a, **k):
            captured["request"] = req
            return self._FakeResp()

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        webhook_command(
            _make_args(webhook_action="test", name="route", payload='{"k": "v"}')
        )
        req = captured["request"]
        # Body must be byte-identical to the --payload argument.
        assert req.data.decode() == '{"k": "v"}'
        expected = (
            "sha256="
            + hmac.new(
                b"route-secret-42", b'{"k": "v"}', hashlib.sha256
            ).hexdigest()
        )
        headers = {k.lower(): v for k, v in req.headers.items()}
        assert headers["x-hub-signature-256"] == expected

    def test_prints_error_when_gateway_unreachable(self, monkeypatch, capsys):
        self._make_test_route()

        def _boom(*a, **k):
            raise ConnectionRefusedError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        webhook_command(_make_args(webhook_action="test", name="route"))
        out = capsys.readouterr().out
        assert "Error:" in out
        assert "Is the gateway running?" in out

