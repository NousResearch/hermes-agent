"""P3b — model override persistence across restart (secret-safe, config-backed-only).

The persisted model override stores ONLY {model, provider, api_mode} — never the
api_key/base_url (re-resolved from provider config on boot). Only a
provider-config-resolvable override is persisted; an ad-hoc/unresolvable one is
skipped. All clear-sites route through the single _set_session_model_override door.
"""

import json
import re
import pathlib
import pytest
from unittest.mock import MagicMock, patch

import gateway.run as gateway_run
from gateway.session import SessionEntry, SessionStore
from datetime import datetime, timezone


def _entry(session_key="agent:main:discord:c1:c1", session_id="s1"):
    now = datetime.now(timezone.utc)
    return SessionEntry(session_key=session_key, session_id=session_id, created_at=now, updated_at=now)


def _make_store(tmp_path):
    store = object.__new__(SessionStore)
    store._entries = {}
    store.sessions_dir = tmp_path
    store._lock = __import__("threading").RLock()
    return store


def _runner(store):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner.session_store = store
    return runner


REPO = pathlib.Path(__file__).resolve().parents[2]


class TestPersistedModelOverrideHasNoSecret:
    def test_no_api_key_or_base_url_on_disk(self, tmp_path):
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key)
        runner = _runner(store)
        # Force "persistable": patch the resolvability check to return an identity.
        with patch.object(
            gateway_run.GatewayRunner, "_model_override_is_persistable",
            return_value={"model": "claude-fable-5", "provider": "claude-app", "api_mode": "anthropic_messages"},
        ):
            runner._set_session_model_override(key, {
                "model": "claude-fable-5", "provider": "claude-app",
                "api_key": "sk-SECRET-should-not-persist",
                "base_url": "https://user:tok@endpoint",
                "api_mode": "anthropic_messages",
            })
        saved_text = (tmp_path / "sessions.json").read_text()
        assert "sk-SECRET-should-not-persist" not in saved_text
        assert "user:tok@endpoint" not in saved_text
        ident = json.loads(saved_text)[key]["model_override_identity"]
        assert ident == {"model": "claude-fable-5", "provider": "claude-app", "api_mode": "anthropic_messages"}
        assert "api_key" not in ident and "base_url" not in ident


class TestAdhocNotPersisted:
    def test_unresolvable_override_not_persisted(self, tmp_path):
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key)
        runner = _runner(store)
        # Simulate an ad-hoc / unresolvable override: persistability returns None.
        with patch.object(gateway_run.GatewayRunner, "_model_override_is_persistable", return_value=None):
            runner._set_session_model_override(key, {
                "model": "some-adhoc-model", "provider": "ad-hoc-provider",
                "api_key": "inline-key", "base_url": "https://adhoc", "api_mode": "openai",
            })
        # In-memory override IS set (session still works this run) ...
        assert key in runner._session_model_overrides
        # ... but nothing persisted to disk.
        assert store._entries[key].model_override_identity is None
        saved = json.loads((tmp_path / "sessions.json").read_text())
        assert saved[key].get("model_override_identity") is None


class TestRehydrateReresolves:
    def test_rehydrate_reresolves_credentials(self, tmp_path):
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        e = _entry(key)
        e.model_override_identity = {"model": "claude-fable-5", "provider": "claude-app", "api_mode": "anthropic_messages"}
        store._entries[key] = e
        store._ensure_loaded = lambda: None
        runner = _runner(store)
        with patch.object(
            gateway_run.GatewayRunner, "_reresolve_model_override_credentials",
            return_value={"model": "claude-fable-5", "provider": "claude-app",
                          "api_key": "re-resolved-key", "base_url": "https://real", "api_mode": "anthropic_messages"},
        ):
            runner._rehydrate_session_overrides()
        got = runner._session_model_overrides.get(key)
        assert got and got["api_key"] == "re-resolved-key" and got["base_url"] == "https://real"

    def test_rehydrate_skips_when_provider_gone(self, tmp_path):
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        e = _entry(key)
        e.model_override_identity = {"model": "x", "provider": "retired", "api_mode": None}
        store._entries[key] = e
        store._ensure_loaded = lambda: None
        runner = _runner(store)
        with patch.object(gateway_run.GatewayRunner, "_reresolve_model_override_credentials", return_value=None):
            runner._rehydrate_session_overrides()  # must not raise
        assert key not in runner._session_model_overrides


class TestSingleDoorGrep:
    def test_all_model_clear_sites_route_through_single_door(self):
        # Grep gate (RC-2): no bare _session_model_overrides.pop( outside the single
        # door (_set_session_model_override) and the deliberately-excluded MoA
        # transient restore (annotated NOTE(P3b/RC-2)).
        offenders = []
        for rel in ("gateway/run.py", "gateway/slash_commands.py"):
            text = (REPO / rel).read_text().splitlines()
            for i, line in enumerate(text):
                if "_session_model_overrides.pop(" in line:
                    # allow: the pop INSIDE _set_session_model_override (the door
                    # itself), and the MoA transient restore (annotated NOTE).
                    # Find the nearest enclosing def above this line.
                    enclosing_def = ""
                    for j in range(i, max(0, i - 60), -1):
                        m = re.match(r"    (?:async )?def (\w+)", text[j])
                        if m:
                            enclosing_def = m.group(1)
                            break
                    ctx = "\n".join(text[max(0, i - 8):i + 1])
                    if enclosing_def == "_set_session_model_override" or "NOTE(P3b/RC-2)" in ctx:
                        continue
                    offenders.append(f"{rel}:{i+1}: {line.strip()}")
        assert not offenders, "bare model-override pop outside the single door:\n" + "\n".join(offenders)

    def test_model_command_has_no_inline_credential_flag(self):
        # C7/INV-PROV: parse_model_flags accepts no --api-key/--base-url. If a future
        # edit adds one, this fails loudly (persistence provenance must be revisited).
        # Check the executable body only (strip the docstring, which mentions the flag
        # names in prose explaining WHY they're forbidden).
        src = (REPO / "hermes_cli" / "model_switch.py").read_text()
        m = re.search(r"def parse_model_flags\(.*?\n(?:.*?\n)*?    return ", src)
        body = m.group(0) if m else ""
        # Drop the triple-quoted docstring.
        body_no_doc = re.sub(r'"""(?:.|\n)*?"""', "", body)
        low = body_no_doc.lower()
        assert "api-key" not in low and "api_key" not in low, body_no_doc
        assert "base-url" not in low and "base_url" not in low, body_no_doc
