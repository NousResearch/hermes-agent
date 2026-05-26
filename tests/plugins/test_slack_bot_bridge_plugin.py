"""Tests for the slack-bot-bridge plugin.

The plugin sits on ``pre_gateway_dispatch`` and rewrites a tagged JSON
envelope posted by an external bot in a single dedicated Slack channel
into a canonical prompt. None of the tests touch Slack — they exercise
the plugin's parsing, allow/deny, dedupe, signature, and rewrite logic
directly.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Loader — import the plugin module under hermes_plugins.* like the runtime does
# ---------------------------------------------------------------------------


def _load_plugin():
    """Import plugins/slack-bot-bridge/__init__.py as hermes_plugins.slack_bot_bridge."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "slack-bot-bridge"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    mod_name = "hermes_plugins.slack_bot_bridge"
    # Always re-load so test-time environment changes (HERMES_HOME) take effect
    # cleanly without state from another test bleeding in.
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(
        mod_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = mod_name
    mod.__path__ = [str(plugin_dir)]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """Fresh plugin module with HERMES_HOME pointing at a tempdir."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return _load_plugin()


# ---------------------------------------------------------------------------
# Event fakes — mimic the shape gateway/run.py hands to the hook
# ---------------------------------------------------------------------------

CHANNEL = "C0BRIDGE0"
HERMES_UID = "U_HERMES"


def _slack_envelope_text(envelope: dict, *, mention: str = f"<@{HERMES_UID}>", marker: str = "hermes-bridge") -> str:
    body = json.dumps(envelope)
    return f"{mention} {marker}\n```json\n{body}\n```"


def _make_slack_event(
    *,
    text: str,
    channel: str = CHANNEL,
    bot_id: str = "B0RELAY",
    app_id: str = "A0RELAYAPP",
    team: str = "T0WORKSPACE",
    subtype: str = "bot_message",
    user_id: str = "U_RELAY",
    platform: str = "slack",
):
    raw = {
        "type": "message",
        "subtype": subtype,
        "bot_id": bot_id,
        "app_id": app_id,
        "team": team,
        "channel": channel,
        "user": user_id,
        "text": text,
        "ts": "1700000000.000001",
    }
    source = SimpleNamespace(
        platform=SimpleNamespace(value=platform),
        chat_id=channel,
        user_id=user_id,
        chat_type="group",
    )
    return SimpleNamespace(source=source, raw_message=raw, text=text)


def _hmac_signature(secret: str, request_id: str, actor: str, prompt: str, metadata: dict | None = None) -> str:
    metadata_text = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hmac.new(
        secret.encode("utf-8"),
        f"{request_id}|{actor}|{prompt}|{metadata_text}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Envelope-parsing tests (low level)
# ---------------------------------------------------------------------------


class TestExtractEnvelopeText:
    def test_finds_fenced_block(self, plugin):
        text = _slack_envelope_text({"request_id": "r1", "prompt": "do thing"})
        body = plugin._extract_envelope_text(text)
        assert body is not None
        assert json.loads(body) == {"request_id": "r1", "prompt": "do thing"}

    def test_accepts_inline_json_without_fence(self, plugin):
        body = '{"request_id": "r1", "prompt": "do thing"}'
        text = f"<@{HERMES_UID}> hermes-bridge {body}"
        extracted = plugin._extract_envelope_text(text)
        assert extracted == body

    def test_requires_marker(self, plugin):
        text = f"<@{HERMES_UID}> please run this ```json\n{{}}\n```"
        assert plugin._extract_envelope_text(text) is None

    def test_marker_is_case_insensitive(self, plugin):
        text = f"<@{HERMES_UID}> HERMES-BRIDGE\n```json\n{{\"request_id\":\"r1\",\"prompt\":\"x\"}}\n```"
        assert plugin._extract_envelope_text(text) is not None

    def test_requires_some_mention(self, plugin):
        # Marker present but no <@...> mention → ignored.
        text = "hermes-bridge\n```json\n{\"request_id\":\"r1\",\"prompt\":\"x\"}\n```"
        assert plugin._extract_envelope_text(text) is None

    def test_handles_balanced_braces_inside_strings(self, plugin):
        # Prompt with embedded braces inside a quoted string must not
        # confuse the brace-counting fallback parser.
        body = '{"request_id":"r1","prompt":"if (x) { y(); }","actor":""}'
        text = f"<@{HERMES_UID}> hermes-bridge {body} trailing-garbage"
        extracted = plugin._extract_envelope_text(text)
        assert extracted == body


class TestValidateEnvelopeFields:
    def test_accepts_full_envelope(self, plugin):
        ok = plugin._validate_envelope_fields(
            {"request_id": "r1", "prompt": "hi", "actor": "ci", "metadata": {"k": 1}}
        )
        assert ok == ("r1", "ci", "hi", {"k": 1})

    def test_rejects_missing_request_id(self, plugin):
        assert plugin._validate_envelope_fields({"prompt": "hi"}) is None

    def test_rejects_blank_prompt(self, plugin):
        assert plugin._validate_envelope_fields({"request_id": "r1", "prompt": "   "}) is None

    def test_rejects_oversize_prompt(self, plugin):
        big = "x" * 40_000
        assert plugin._validate_envelope_fields({"request_id": "r1", "prompt": big}) is None

    def test_rejects_invalid_id_charset(self, plugin):
        # Spaces are disallowed in request_id to keep the canonical signing
        # input unambiguous.
        assert plugin._validate_envelope_fields({"request_id": "r 1", "prompt": "hi"}) is None

    def test_rejects_non_dict_metadata(self, plugin):
        bad = {"request_id": "r1", "prompt": "hi", "metadata": ["nope"]}
        assert plugin._validate_envelope_fields(bad) is None


# ---------------------------------------------------------------------------
# Dedup store
# ---------------------------------------------------------------------------


class TestDedup:
    def test_first_claim_succeeds_second_rejects(self, plugin):
        assert plugin._claim_request_id("rid-1") is True
        assert plugin._claim_request_id("rid-1") is False

    def test_expired_entry_can_be_reused(self, plugin, monkeypatch):
        monkeypatch.setenv("HERMES_SLACK_BRIDGE_DEDUP_TTL_SECONDS", "10")
        assert plugin._claim_request_id("rid-1", now=1000.0) is True
        # After the TTL has passed, the same id reclaims successfully.
        assert plugin._claim_request_id("rid-1", now=1000.0 + 11) is True

    def test_state_survives_module_reload(self, plugin):
        plugin._claim_request_id("rid-shared")
        reloaded = _load_plugin()  # same HERMES_HOME — reads the file
        assert reloaded._claim_request_id("rid-shared") is False

    def test_memory_fallback_rejects_duplicates_when_disk_write_fails(self, plugin, monkeypatch):
        monkeypatch.setattr(plugin, "_write_dedup_state", lambda path, state: False)

        assert plugin._claim_request_id("rid-memory-fallback") is True
        assert plugin._claim_request_id("rid-memory-fallback") is False


# ---------------------------------------------------------------------------
# HMAC validation
# ---------------------------------------------------------------------------


class TestSignature:
    def test_accepts_correct_signature(self, plugin):
        sig = _hmac_signature("topsecret", "r1", "ci", "hi")
        ok = plugin._verify_signature(
            "topsecret",
            {"signature": sig},
            "r1",
            "ci",
            "hi",
            {},
        )
        assert ok is True

    def test_rejects_missing_signature(self, plugin):
        assert plugin._verify_signature("topsecret", {}, "r1", "ci", "hi", {}) is False

    def test_rejects_wrong_signature(self, plugin):
        assert (
            plugin._verify_signature("topsecret", {"signature": "deadbeef"}, "r1", "ci", "hi", {})
            is False
        )

    def test_rejects_when_payload_tampered(self, plugin):
        sig = _hmac_signature("topsecret", "r1", "ci", "hi")
        # Even though the signature is valid for "hi", supplying a
        # different prompt must invalidate it.
        assert (
            plugin._verify_signature(
                "topsecret",
                {"signature": sig},
                "r1",
                "ci",
                "DIFFERENT",
                {},
            )
            is False
        )

    def test_rejects_when_metadata_tampered(self, plugin):
        original_metadata = {"repo": "org/repo", "run_id": "1"}
        sig = _hmac_signature("topsecret", "r1", "ci", "hi", original_metadata)
        assert (
            plugin._verify_signature(
                "topsecret",
                {"signature": sig},
                "r1",
                "ci",
                "hi",
                {"repo": "org/repo", "run_id": "2"},
            )
            is False
        )


# ---------------------------------------------------------------------------
# End-to-end hook behavior
# ---------------------------------------------------------------------------


@pytest.fixture
def configure_bridge(monkeypatch):
    """Apply the minimum env vars needed for the plugin to consider an event."""
    def _configure(
        *,
        bot_ids: str = "B0RELAY",
        app_ids: str = "",
        teams: str = "",
        secret: str = "",
        channel: str = CHANNEL,
    ):
        monkeypatch.setenv("HERMES_SLACK_BRIDGE_CHANNEL", channel)
        if bot_ids:
            monkeypatch.setenv("HERMES_SLACK_BRIDGE_ALLOWED_BOT_IDS", bot_ids)
        else:
            monkeypatch.delenv("HERMES_SLACK_BRIDGE_ALLOWED_BOT_IDS", raising=False)
        if app_ids:
            monkeypatch.setenv("HERMES_SLACK_BRIDGE_ALLOWED_APP_IDS", app_ids)
        else:
            monkeypatch.delenv("HERMES_SLACK_BRIDGE_ALLOWED_APP_IDS", raising=False)
        if teams:
            monkeypatch.setenv("HERMES_SLACK_BRIDGE_ALLOWED_TEAMS", teams)
        else:
            monkeypatch.delenv("HERMES_SLACK_BRIDGE_ALLOWED_TEAMS", raising=False)
        if secret:
            monkeypatch.setenv("HERMES_SLACK_BRIDGE_HMAC_SECRET", secret)
        else:
            monkeypatch.delenv("HERMES_SLACK_BRIDGE_HMAC_SECRET", raising=False)
    return _configure


class TestHookEndToEnd:
    def test_rewrites_canonical_prompt(self, plugin, configure_bridge):
        configure_bridge()
        envelope = {
            "request_id": "r-ok",
            "prompt": "Triage failure in PR #1",
            "actor": "github-actions",
            "metadata": {"pr": 1, "repo": "org/r"},
        }
        event = _make_slack_event(text=_slack_envelope_text(envelope))

        result = plugin.on_pre_gateway_dispatch(event=event)

        assert isinstance(result, dict)
        assert result["action"] == "rewrite"
        # Header carries request_id and metadata; original prompt follows.
        assert "request_id: r-ok" in result["text"]
        assert "actor: github-actions" in result["text"]
        assert '"pr": 1' in result["text"]
        assert "Triage failure in PR #1" in result["text"]

    def test_returns_none_for_non_slack_event(self, plugin, configure_bridge):
        configure_bridge()
        envelope = {"request_id": "r1", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope), platform="telegram")
        assert plugin.on_pre_gateway_dispatch(event=event) is None

    def test_returns_none_when_channel_does_not_match(self, plugin, configure_bridge):
        configure_bridge()
        envelope = {"request_id": "r1", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope), channel="C_OTHER")
        assert plugin.on_pre_gateway_dispatch(event=event) is None

    def test_inert_without_channel_env(self, plugin, monkeypatch):
        # No HERMES_SLACK_BRIDGE_CHANNEL set → plugin should be a no-op
        # even if everything else looks right.
        monkeypatch.delenv("HERMES_SLACK_BRIDGE_CHANNEL", raising=False)
        envelope = {"request_id": "r1", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        assert plugin.on_pre_gateway_dispatch(event=event) is None

    def test_skips_human_bridge_attempt_in_bridge_channel(self, plugin, configure_bridge):
        configure_bridge()
        envelope = {"request_id": "r1", "prompt": "hi"}
        # No bot_id and not a bot_message subtype → plain user post.
        event = _make_slack_event(
            text=_slack_envelope_text(envelope),
            bot_id="",
            subtype="",
        )
        assert plugin.on_pre_gateway_dispatch(event=event) == {
            "action": "skip",
            "reason": "bridge-attempt-not-from-bot",
        }

    def test_rejects_unallowlisted_bot_id(self, plugin, configure_bridge):
        configure_bridge(bot_ids="B_OTHER_ONLY")
        envelope = {"request_id": "r1", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        assert plugin.on_pre_gateway_dispatch(event=event) == {
            "action": "skip",
            "reason": "sender-not-allowlisted",
        }

    def test_app_id_allowlist_is_alternative_to_bot_id(self, plugin, configure_bridge):
        # No bot_id allowlist, but app_id does match → accept.
        configure_bridge(bot_ids="", app_ids="A0RELAYAPP")
        envelope = {"request_id": "r-app", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        result = plugin.on_pre_gateway_dispatch(event=event)
        assert result is not None and result["action"] == "rewrite"

    def test_team_allowlist_is_combined_with_sender_identity(self, plugin, configure_bridge):
        configure_bridge(bot_ids="B_OTHER_ONLY", app_ids="", teams="T0WORKSPACE")
        envelope = {"request_id": "r-team-sender", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        assert plugin.on_pre_gateway_dispatch(event=event) == {
            "action": "skip",
            "reason": "sender-not-allowlisted",
        }

    def test_team_allowlist_only_accepts_matching_team(self, plugin, configure_bridge):
        configure_bridge(bot_ids="", app_ids="", teams="T0WORKSPACE")
        envelope = {"request_id": "r-team-only", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        result = plugin.on_pre_gateway_dispatch(event=event)
        assert result is not None and result["action"] == "rewrite"

    def test_empty_allowlists_fall_back_to_channel_only(self, plugin, configure_bridge):
        configure_bridge(bot_ids="", app_ids="", teams="")
        envelope = {"request_id": "r-anonymous", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        result = plugin.on_pre_gateway_dispatch(event=event)
        assert result is not None and result["action"] == "rewrite"

    def test_dedup_skips_duplicate_request_id(self, plugin, configure_bridge):
        configure_bridge()
        envelope = {"request_id": "r-dup", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))

        first = plugin.on_pre_gateway_dispatch(event=event)
        second = plugin.on_pre_gateway_dispatch(event=event)

        assert first is not None and first["action"] == "rewrite"
        assert second == {"action": "skip", "reason": "duplicate-request-id"}

    def test_signature_required_when_secret_set(self, plugin, configure_bridge):
        configure_bridge(secret="topsecret")
        envelope = {"request_id": "r-nosig", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        # No signature in the envelope → fail closed.
        assert plugin.on_pre_gateway_dispatch(event=event) == {
            "action": "skip",
            "reason": "bad-signature",
        }

    def test_bad_signature_fails_closed(self, plugin, configure_bridge):
        configure_bridge(secret="topsecret")
        envelope = {"request_id": "r-badsig", "prompt": "hi", "signature": "deadbeef"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        assert plugin.on_pre_gateway_dispatch(event=event) == {
            "action": "skip",
            "reason": "bad-signature",
        }

    def test_signature_accepted_when_correct(self, plugin, configure_bridge):
        configure_bridge(secret="topsecret")
        sig = _hmac_signature("topsecret", "r-sig", "", "hi")
        envelope = {"request_id": "r-sig", "prompt": "hi", "signature": sig}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        result = plugin.on_pre_gateway_dispatch(event=event)
        assert result is not None and result["action"] == "rewrite"

    def test_malformed_json_fails_closed(self, plugin, configure_bridge):
        configure_bridge()
        text = f"<@{HERMES_UID}> hermes-bridge\n```json\nNOT JSON\n```"
        event = _make_slack_event(text=text)
        assert plugin.on_pre_gateway_dispatch(event=event) == {
            "action": "skip",
            "reason": "invalid-json-envelope",
        }

    def test_invalid_envelope_fields_fail_closed(self, plugin, configure_bridge):
        configure_bridge()
        envelope = {"request_id": "not valid", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        assert plugin.on_pre_gateway_dispatch(event=event) == {
            "action": "skip",
            "reason": "invalid-envelope-fields",
        }

    def test_handler_swallows_unexpected_errors(self, plugin, configure_bridge, monkeypatch):
        configure_bridge()
        # Force the inner function to raise; the outer wrapper must catch.
        def _boom(_event):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(plugin, "_on_pre_gateway_dispatch_inner", _boom)
        envelope = {"request_id": "r1", "prompt": "hi"}
        event = _make_slack_event(text=_slack_envelope_text(envelope))
        assert plugin.on_pre_gateway_dispatch(event=event) is None


# ---------------------------------------------------------------------------
# Plugin registration plumbing
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_attaches_pre_gateway_dispatch(self, plugin):
        recorded: list[tuple[str, object]] = []

        class _Ctx:
            def register_hook(self, name, cb):
                recorded.append((name, cb))

        plugin.register(_Ctx())

        assert [name for name, _ in recorded] == ["pre_gateway_dispatch"]
        # And the recorded callback is the one we expose.
        assert recorded[0][1] is plugin.on_pre_gateway_dispatch
