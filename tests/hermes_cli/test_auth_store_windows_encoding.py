"""Regression tests for auth store encoding on Windows.

``_load_auth_store`` and the Codex/Nous shared-store readers previously called
``Path.read_text()`` with no ``encoding=``, so the bytes were decoded with
``locale.getpreferredencoding()`` — cp1252 on Windows. The store is *written* as
UTF-8 (``_save_auth_store`` uses ``encoding="utf-8"``), so any non-ASCII byte
(e.g. a CJK or emoji credential label) raised ``UnicodeDecodeError`` on read.

Worst case: ``_load_auth_store``'s broad ``except`` then copied the file to
``.json.corrupt`` and returned an *empty* store — silently wiping every
provider credential. These tests pin the round-trip so a non-ASCII label
survives a save→load cycle, and assert the readers pass an explicit UTF-8
encoding (the fix) instead of relying on the locale default.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

import hermes_cli.auth as auth


# --- helpers ---------------------------------------------------------------

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a tmp dir so we never touch the real auth store.

    Required because ``_auth_file_path()`` has a seat belt that refuses to
    resolve to the real user's ~/.hermes/auth.json under pytest.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_utf8(path: Path, payload: dict) -> None:
    """Write JSON as UTF-8, mirroring _save_auth_store's encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- the bug: a non-ASCII label must survive save → load -------------------

class TestAuthStoreEncodingRoundTrip:
    def test_load_reads_utf8_with_non_ascii_label(self, hermes_home):
        """A UTF-8 store with a CJK/emoji label loads intact (not wiped)."""
        store = {
            "version": auth.AUTH_STORE_VERSION,
            "providers": {
                "openai-codex": {
                    "auth_mode": "chatgpt",
                    "label": "工作账号 🔥",  # non-ASCII: CJK + emoji
                    "tokens": {"access_token": "a", "refresh_token": "r"},
                }
            },
        }
        auth_path = hermes_home / "auth.json"
        _write_utf8(auth_path, store)

        loaded = auth._load_auth_store(auth_path)

        # The label round-trips exactly — the provider is NOT lost.
        assert "openai-codex" in loaded["providers"]
        assert loaded["providers"]["openai-codex"]["label"] == "工作账号 🔥"

    def test_load_does_not_corrupt_store_on_non_ascii(self, hermes_home):
        """The pre-fix bug wiped the store to empty on a UnicodeDecodeError.

        After the fix, loading a valid UTF-8 store must never produce the empty
        fallback, and must never write a .json.corrupt sidecar.
        """
        store = {
            "version": auth.AUTH_STORE_VERSION,
            "providers": {"x": {"label": "José's key"}},
        }
        auth_path = hermes_home / "auth.json"
        _write_utf8(auth_path, store)

        auth._load_auth_store(auth_path)

        assert not (hermes_home / "auth.json.corrupt").exists()
        # original file untouched — read it back and compare structurally
        # (json.dumps may escape non-ASCII, so compare parsed values, not text)
        on_disk = json.loads(auth_path.read_text(encoding="utf-8"))
        assert on_disk["providers"]["x"]["label"] == "José's key"

    def test_load_handles_utf8_with_bom(self, hermes_home):
        """A BOM (e.g. from Notepad editing) must not break the read."""
        store = {"version": auth.AUTH_STORE_VERSION, "providers": {"x": {"label": "café"}}}
        auth_path = hermes_home / "auth.json"
        payload = json.dumps(store)
        # write with utf-8-sig to prepend the BOM
        auth_path.write_text(payload, encoding="utf-8-sig")

        loaded = auth._load_auth_store(auth_path)
        assert loaded["providers"]["x"]["label"] == "café"


# --- the fix: readers pass an explicit encoding ---------------------------

class TestExplicitEncodingPassed:
    """The readers must not rely on the locale default (cp1252 on Windows).

    We assert read_text is called with an explicit UTF-8 encoding. This is the
    regression guard: a future refactor that drops the encoding kwarg would
    reintroduce the Windows data-loss bug.
    """

    def test_load_auth_store_passes_utf8_encoding(self, hermes_home):
        auth_path = hermes_home / "auth.json"
        _write_utf8(auth_path, {"version": auth.AUTH_STORE_VERSION, "providers": {}})

        with mock.patch.object(
            Path, "read_text", wraps=Path.read_text
        ) as spy:
            auth._load_auth_store(auth_path)

        assert spy.call_count == 1
        kwargs = spy.call_args.kwargs
        assert "encoding" in kwargs, "read_text() must pass an explicit encoding"
        assert "utf-8" in str(kwargs["encoding"]).lower()

    def test_codex_store_reader_passes_utf8_encoding(self, tmp_path, monkeypatch):
        """The ~/.codex/auth.json reader must pass an explicit UTF-8 encoding."""
        codex_home = tmp_path / "codex"
        codex_home.mkdir()
        (codex_home / "auth.json").write_text(
            json.dumps({"tokens": {"access_token": "a", "refresh_token": "r"}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        # Bypass the JWT-expiry check so a fake token doesn't short-circuit.
        monkeypatch.setattr(auth, "_codex_access_token_is_expiring", lambda *a, **k: False)

        with mock.patch.object(Path, "read_text", wraps=Path.read_text) as spy:
            auth._import_codex_cli_tokens()

        # _import_codex_cli_tokens reads exactly one file; assert that read
        # carried an explicit UTF-8 encoding. (The bound-method spy captures
        # kwargs but not the bound `self`, so we check the single read directly.)
        assert spy.call_count >= 1, "expected a read of the codex auth.json"
        for call in spy.call_args_list:
            assert "encoding" in call.kwargs, "codex read_text() must pass encoding"
            assert "utf-8" in str(call.kwargs["encoding"]).lower()
