"""Tests for the vault-backed password-blind browser autofill feature.

Covers:
- VaultStore: encrypt/decrypt round-trip, file perms, identifier-as-metadata
  (login secret payload is password-only)
- login-control classifier: scoring + new-password/one-time-code exclusion,
  password-only fill selection
- origin-binding refusal (pre-check + in-script TOCTOU assert)
- fail-closed secret eval (no argv fallback)
- vault-value redaction registry (browser_cdp read-back regression)
- tool gating: check_fn False when the vault is empty
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.vault_login_classifier import (  # noqa: E402
    ClassifiedLoginControl,
    LoginControl,
    build_fill_js,
    classify_login_control,
    select_password_fill,
)
from agent.vault_store import (  # noqa: E402
    VaultError,
    VaultStore,
    normalize_origin,
    scrub_secret_from_text,
)


@pytest.fixture()
def store(tmp_path):
    return VaultStore(base_dir=tmp_path / "vault")


def _add_login(store, origin="https://example.com", password="s3cret-pw"):
    return store.add_item(
        kind="login",
        label="Example login",
        origin=origin,
        secret={
            "identifier_type": "email",
            "identifier": "user@example.com",
            "password": password,
            "origin": origin,
        },
    )


# ---------------------------------------------------------------------------
# VaultStore
# ---------------------------------------------------------------------------

class TestVaultStore:
    def test_roundtrip_encrypt_decrypt(self, store):
        meta = _add_login(store)
        secret = store.resolve_secret(meta.id)
        # Design: login secret payload is password-only; identifier is metadata.
        assert secret == {"password": "s3cret-pw"}
        assert meta.identifier == "user@example.com"
        assert meta.identifier_type == "email"

    def test_vault_file_never_contains_password(self, store, tmp_path):
        _add_login(store)
        blob = (tmp_path / "vault" / "vault.json.enc").read_bytes()
        assert b"s3cret-pw" not in blob

    def test_file_permissions_0600(self, store, tmp_path):
        _add_login(store)
        for name in ("vault.json.enc", "vault.key"):
            mode = stat.S_IMODE(os.stat(tmp_path / "vault" / name).st_mode)
            assert mode == 0o600, f"{name} has mode {oct(mode)}"

    def test_listing_is_password_free(self, store):
        meta = _add_login(store)
        items = store.list_items()
        assert len(items) == 1
        dumped = json.dumps(items[0].to_dict())
        assert "s3cret-pw" not in dumped
        assert "password" not in dumped
        # Identifier IS visible metadata now.
        assert items[0].identifier == "user@example.com"
        assert items[0].identifier_type == "email"
        assert items[0].id == meta.id
        assert items[0].origin == "https://example.com"

    def test_remove_item(self, store):
        meta = _add_login(store)
        assert store.remove_item(meta.id) is True
        assert store.remove_item(meta.id) is False
        assert store.list_items() == []

    def test_login_requires_origin(self, store):
        with pytest.raises(VaultError):
            store.add_item(
                kind="login",
                label="x",
                secret={
                    "identifier_type": "email",
                    "identifier": "a@b.c",
                    "password": "p",
                },
            )

    def test_all_kinds_supported(self, store):
        store.add_item(kind="payment", label="Card", secret={"number": "4111"})
        store.add_item(kind="address", label="Home", secret={"street": "1 Main St"})
        kinds = {m.kind for m in store.list_items()}
        assert kinds == {"payment", "address"}

    def test_unknown_kind_rejected(self, store):
        with pytest.raises(VaultError):
            store.add_item(kind="totp", label="x", secret={})

    def test_has_items(self, store):
        assert store.has_items() is False
        _add_login(store)
        assert store.has_items() is True

    def test_normalize_origin(self):
        assert normalize_origin("https://Example.com:443/login?x=1") == "https://example.com"
        assert normalize_origin("http://localhost:8931/") == "http://localhost:8931"
        assert normalize_origin("http://site.test:80") == "http://site.test"
        with pytest.raises(VaultError):
            normalize_origin("example.com")

    def test_scrub_secret_from_text(self):
        secret = {"password": "hunter22x", "identifier": "me@x.io"}
        out = scrub_secret_from_text("boom hunter22x at me@x.io", secret)
        assert "hunter22x" not in out
        assert "me@x.io" not in out


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _ctrl(**kw):
    base = dict(autocomplete="", form_index=0, index=0, label="", name="", type="text")
    base.update(kw)
    return LoginControl(**base)


class TestClassifier:
    def test_autocomplete_exact_match_scores_100(self):
        for token in ("username", "email", "tel", "current-password"):
            res = classify_login_control(_ctrl(autocomplete=token))
            assert res is not None and res.score == 100 and res.token == token

    def test_new_password_autocomplete_excluded(self):
        assert classify_login_control(
            _ctrl(autocomplete="new-password", type="password")
        ) is None

    def test_one_time_code_excluded(self):
        assert classify_login_control(_ctrl(autocomplete="one-time-code")) is None

    def test_label_new_password_excluded(self):
        for label in ("New password", "Confirm Password", "create-password", "Repeat  password"):
            assert classify_login_control(_ctrl(type="password", label=label)) is None, label

    def test_password_type_scores_90(self):
        res = classify_login_control(_ctrl(type="password"))
        assert res.score == 90 and res.token == "current-password"

    def test_email_tel_types_score_85(self):
        assert classify_login_control(_ctrl(type="email")).score == 85
        res = classify_login_control(_ctrl(type="tel"))
        assert res.score == 85 and res.token == "tel"

    def test_label_heuristics(self):
        assert classify_login_control(_ctrl(label="E-mail address")).token == "email"
        assert classify_login_control(_ctrl(name="mobile_number")).token == "tel"
        res = classify_login_control(_ctrl(label="Username or account"))
        assert res.token == "username" and res.score == 70

    def test_unmatched_returns_none(self):
        assert classify_login_control(_ctrl(label="Search the docs")) is None

    def test_select_password_fill_picks_best_password(self):
        user = ClassifiedLoginControl(_ctrl(index=0, form_index=0, autocomplete="username"), 100, "username")
        pw_heur = ClassifiedLoginControl(_ctrl(index=1, form_index=0, type="password"), 90, "current-password")
        pw_exact = ClassifiedLoginControl(_ctrl(index=3, form_index=0, autocomplete="current-password"), 100, "current-password")
        fills = select_password_fill([user, pw_heur, pw_exact], "p")
        # Password only — the identifier field is never filled by the vault.
        assert [(f["index"], f["token"]) for f in fills] == [(3, "current-password")]

    def test_select_password_fill_requires_password_field(self):
        user = ClassifiedLoginControl(_ctrl(index=0, autocomplete="username"), 100, "username")
        assert select_password_fill([user], "p") == []

    def test_select_password_fill_single_field_only(self):
        pw1 = ClassifiedLoginControl(_ctrl(index=1, type="password"), 90, "current-password")
        pw2 = ClassifiedLoginControl(_ctrl(index=2, type="password"), 90, "current-password")
        fills = select_password_fill([pw1, pw2], "p")
        assert len(fills) == 1 and fills[0]["index"] == 1

    def test_build_fill_js_contains_events(self):
        js = build_fill_js(
            [{"index": 0, "token": "current-password", "value": "x"}],
            expected_origin="https://example.com",
        )
        assert "InputEvent" in js and '"change"' in js and "filled" in js

    def test_build_fill_js_has_no_dom_marker(self):
        # P1-1: no deterministic selector for filled controls.
        js = build_fill_js(
            [{"index": 0, "token": "current-password", "value": "x"}],
            expected_origin="https://example.com",
        )
        assert "vaultSecret" not in js
        assert "data-vault-secret" not in js

    def test_build_fill_js_asserts_origin_before_any_write(self):
        # P1-2: the origin assert must run inside the SAME script, before
        # any element write.
        js = build_fill_js(
            [{"index": 0, "token": "current-password", "value": "x"}],
            expected_origin="https://example.com",
        )
        assert '"https://example.com"' in js
        assert "window.location.origin" in js
        assert "origin_changed" in js
        assert js.index("origin_changed") < js.index("querySelectorAll")


# ---------------------------------------------------------------------------
# Browser tool: origin binding + gating
# ---------------------------------------------------------------------------

class TestBrowserVaultTools:
    def test_check_fn_false_when_vault_empty(self, tmp_path):
        from tools import browser_vault_tool

        empty = VaultStore(base_dir=tmp_path / "empty-vault")
        with patch("agent.vault_store.get_vault_store", return_value=empty):
            assert browser_vault_tool._check_vault_available() is False

    def test_check_fn_true_with_items(self, store):
        from tools import browser_vault_tool

        _add_login(store)
        with patch("agent.vault_store.get_vault_store", return_value=store):
            assert browser_vault_tool._check_vault_available() is True

    def test_list_returns_identifier_never_password(self, store):
        from tools import browser_vault_tool

        _add_login(store)
        with patch("agent.vault_store.get_vault_store", return_value=store):
            out = json.loads(browser_vault_tool.browser_vault_list())
        assert out["success"] is True
        assert out["items"][0]["handle"].startswith("vault_")
        # Design change: identifier is agent-visible metadata.
        assert out["items"][0]["identifier"] == "user@example.com"
        assert out["items"][0]["identifier_type"] == "email"
        assert "s3cret-pw" not in json.dumps(out)

    def test_fill_refused_on_origin_mismatch(self, store):
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_current_page_origin", return_value="https://evil.com"):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
        assert out["success"] is False
        assert "Refused" in out["error"]
        assert "s3cret-pw" not in json.dumps(out)

    def test_fill_unknown_handle(self, store):
        from tools import browser_vault_tool

        with patch("agent.vault_store.get_vault_store", return_value=store):
            out = json.loads(browser_vault_tool.browser_vault_fill("vault_nope"))
        assert out["success"] is False

    def test_fill_success_returns_counts_only(self, store):
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [
            {"autocomplete": "email", "formIndex": 0, "index": 0, "label": "", "name": "email", "type": "email"},
            {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        secret_exprs = []

        def fake_eval_secret(task_id, expression):
            secret_exprs.append(expression)
            return {"success": True, "result": json.dumps({"filled": 1})}

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        # Password-only fill: exactly one field.
        assert out == {
            "success": True,
            "filled_fields": 1,
            "kind": "login",
            "origin": "https://example.com",
        }
        assert "s3cret-pw" not in raw
        # The secret expression only ever goes through the secret eval path,
        # and it targets only the password field (index 1).
        assert len(secret_exprs) == 1
        assert "s3cret-pw" in secret_exprs[0]
        assert '"index": 0' not in secret_exprs[0]
        assert "user@example.com" not in secret_exprs[0]

    def test_fill_toctou_navigation_writes_nothing(self, store):
        """P1-2 schedule regression: inspection passes on the allowed origin,
        the page navigates before the fill script runs, the in-script origin
        assert refuses, and zero credential bytes are written."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [
            {"autocomplete": "current-password", "formIndex": 0, "index": 0, "label": "", "name": "pw", "type": "password"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                # Pre-check sees the allowed origin.
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            # The evaluated script itself must carry the origin assert.
            assert "window.location.origin" in expression
            assert '"https://example.com"' in expression
            # Simulate the page having navigated cross-origin by the time
            # the fill script executes: the script's own assert fires.
            return {
                "success": True,
                "result": json.dumps(
                    {"refused": "origin_changed", "found": "https://evil.com"}
                ),
            }

        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is False
        assert out["error_type"] == "origin_changed"
        assert out.get("filled_fields", 0) == 0
        assert "s3cret-pw" not in raw

    def test_secret_eval_fails_closed_without_supervisor(self, store):
        """P1-1: the secret-bearing eval NEVER falls back to the argv path."""
        from tools import browser_vault_tool

        meta = _add_login(store, origin="https://example.com")
        controls = [
            {"autocomplete": "current-password", "formIndex": 0, "index": 0, "label": "", "name": "pw", "type": "password"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        # No supervisor registered → _eval_js_secret must refuse without
        # ever touching _run_browser_command.
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch("tools.browser_supervisor.SUPERVISOR_REGISTRY") as reg, \
             patch("tools.browser_tool._run_browser_command") as run_cmd:
            reg.get.return_value = None
            raw = browser_vault_tool.browser_vault_fill(meta.id)
        out = json.loads(raw)
        assert out["success"] is False
        assert out["error_type"] == "supervisor_required"
        assert "supervis" in out["error"].lower()
        run_cmd.assert_not_called()
        assert "s3cret-pw" not in raw

    def test_nonsecret_eval_fallback_still_works(self):
        """_eval_js (non-secret) may still fall back to the CLI eval path."""
        from tools import browser_vault_tool

        with patch("tools.browser_supervisor.SUPERVISOR_REGISTRY") as reg, \
             patch("tools.browser_tool._last_session_key", return_value="k"), \
             patch("tools.browser_tool._run_browser_command") as run_cmd:
            reg.get.return_value = None
            run_cmd.return_value = {"success": True, "data": {"result": "https://x.test"}}
            res = browser_vault_tool._eval_js("t", "window.location.href")
        assert res == {"success": True, "result": "https://x.test"}
        run_cmd.assert_called_once()

    def test_vault_canary_redacted_from_browser_cdp_results(self, store):
        """P1-1 regression: a filled, non-token-shaped canary password must be
        unrecoverable through a model-facing browser_cdp-style result."""
        from agent import redact
        from agent.redact import redact_sensitive_text
        from tools import browser_vault_tool
        from tools.browser_cdp_tool import _redact_cdp_output

        canary = "plain sentence nobody would flag 7"
        meta = _add_login(store, origin="https://example.com", password=canary)
        controls = [
            {"autocomplete": "current-password", "formIndex": 0, "index": 0, "label": "", "name": "pw", "type": "password"},
        ]

        def fake_eval(task_id, expression):
            if "location.href" in expression:
                return {"success": True, "result": "https://example.com/login"}
            return {"success": True, "result": json.dumps(controls)}

        def fake_eval_secret(task_id, expression):
            return {"success": True, "result": json.dumps({"filled": 1})}

        try:
            with patch("agent.vault_store.get_vault_store", return_value=store), \
                 patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
                 patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_eval_secret):
                out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
            assert out["success"] is True

            # Simulate a browser_cdp Runtime.evaluate sibling read echoing
            # the canary back (e.g. reading the input's value from the DOM).
            cdp_result = {
                "result": {"type": "string", "value": canary},
                "description": f"input value is {canary}",
            }
            scrubbed = _redact_cdp_output(cdp_result)
            assert canary not in json.dumps(scrubbed)
            assert "«redacted-vault-secret»" in json.dumps(scrubbed, ensure_ascii=False)

            # And the generic browser-result scrub catches it too, even with
            # user-level redaction preferences irrelevant (unconditional).
            assert canary not in redact_sensitive_text(f"page text: {canary}")
        finally:
            with redact._VAULT_REDACTION_LOCK:
                redact._VAULT_REDACTION_VALUES.discard(canary)

    def test_fill_rejects_non_login_kind(self, store):
        from tools import browser_vault_tool

        meta = store.add_item(kind="payment", label="Card", secret={"number": "4111"})
        with patch("agent.vault_store.get_vault_store", return_value=store):
            out = json.loads(browser_vault_tool.browser_vault_fill(meta.id))
        assert out["success"] is False
        assert "login" in out["error"]


class TestVaultHardening:
    """Read-deny, backup perms-tightening, canonical dir securing.

    Mirrors the browser-profile snapshot hardening (f1d05c): the vault dir
    holds key + ciphertext side by side, so it gets the same treatment.
    """

    def test_read_block_vault_dir_and_contents(self, tmp_path, monkeypatch):
        import agent.file_safety as fs

        home = tmp_path / "hermes_home"
        vault = home / "vault"
        vault.mkdir(parents=True)
        (vault / "vault.key").write_text("k")
        (vault / "vault.json.enc").write_text("blob")
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)

        for target in (vault, vault / "vault.key", vault / "vault.json.enc"):
            err = fs.get_read_block_error(str(target))
            assert err is not None, f"expected read deny for {target}"
            assert "vault" in err.lower()

    def test_read_block_leaves_sibling_dirs_alone(self, tmp_path, monkeypatch):
        import agent.file_safety as fs

        home = tmp_path / "hermes_home"
        other = home / "vaults-notes"
        other.mkdir(parents=True)
        f = other / "notes.txt"
        f.write_text("hi")
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
        assert fs.get_read_block_error(str(f)) is None

    def test_backup_secret_names_include_vault_files(self):
        from hermes_cli.backup import _SECRET_FILE_NAMES

        assert "vault.key" in _SECRET_FILE_NAMES
        assert "vault.json.enc" in _SECRET_FILE_NAMES

    def test_ensure_dir_uses_canonical_secure_dir(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import hermes_cli.config as cfg
        from agent.vault_store import VaultStore

        called = MagicMock()
        monkeypatch.setattr(cfg, "_secure_dir", called)
        store = VaultStore(base_dir=tmp_path / "vault")
        store._ensure_dir()
        assert called.call_count == 1
