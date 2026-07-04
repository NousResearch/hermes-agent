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
    """Write JSON as UTF-8 with non-ASCII chars as real UTF-8 bytes.

    Uses ``ensure_ascii=False`` so the file actually contains non-ASCII bytes
    (the trigger for the Windows cp1252 bug). The default ``ensure_ascii=True``
    would escape them to ``\\uXXXX`` ASCII, hiding the bug.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def windows_default_encoding(monkeypatch):
    """Simulate the Windows locale default for ``Path.read_text()``.

    On Windows, ``locale.getpreferredencoding()`` is ``cp1252``, so a bare
    ``read_text()`` decodes UTF-8 bytes as cp1252 and raises
    ``UnicodeDecodeError`` on any non-ASCII byte. POSIX test runners default to
    UTF-8, so the bug is invisible there — this fixture makes a no-encoding
    ``read_text()`` behave like Windows (cp1252), while leaving calls that pass
    an explicit encoding untouched. That lets the regression tests actually
    catch the bug on any platform.
    """
    real_read_text = Path.read_text

    def _windows_read_text(self, *args, **kwargs):
        if "encoding" not in kwargs and not args:
            # No explicit encoding → force the Windows default.
            kwargs["encoding"] = "cp1252"
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _windows_read_text)




# --- the bug: a non-ASCII label must survive save → load -------------------

class TestAuthStoreEncodingRoundTrip:
    def test_load_reads_utf8_with_non_ascii_label(self, hermes_home, windows_default_encoding):
        """A UTF-8 store with a CJK/emoji label loads intact (not wiped).

        Under the Windows-default-encoding fixture, a no-encoding read_text()
        raises UnicodeDecodeError on the non-ASCII bytes and the broad except
        wipes the store — so this test catches the bug on any platform.
        """
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

    def test_load_does_not_corrupt_store_on_non_ascii(self, hermes_home, windows_default_encoding):
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


# --- sibling readers of the same ~/.hermes/auth.json in other modules -------
#
# _load_auth_store lives in hermes_cli/auth.py, but several other modules read
# the same ~/.hermes/auth.json directly. They had the same UTF-8-vs-cp1252
# asymmetry on Windows — these tests pin the sibling reads too.

class TestAuthJsonSiblingReaders:
    def test_has_xai_credentials_reads_non_ascii_store(self, hermes_home, windows_default_encoding, monkeypatch):
        """tools/xai_http.has_xai_credentials must read a non-ASCII auth.json.

        Pre-fix, a UTF-8 store with a non-ASCII label raised UnicodeDecodeError
        here (cp1252 default on Windows), the broad except swallowed it, and
        xAI OAuth silently looked absent.
        """
        # No XAI_API_KEY env → force the auth.json code path.
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        store = {
            "version": auth.AUTH_STORE_VERSION,
            "providers": {
                "xai-oauth": {
                    # CJK label → UTF-8 bytes (e.g. 0xE5..) that cp1252 cannot
                    # decode, so a no-encoding read raises UnicodeDecodeError.
                    "label": "工作账号",
                    "tokens": {"access_token": "tok"},
                }
            },
        }
        _write_utf8(hermes_home / "auth.json", store)

        from tools.xai_http import has_xai_credentials

        assert has_xai_credentials() is True

    def test_auxiliary_nous_provider_reads_non_ascii_store(self, hermes_home, windows_default_encoding, monkeypatch):
        """agent/auxiliary_client's Nous-provider lookup reads the same store.

        The lookup returns None on any read failure, silently disabling Nous as
        the auxiliary (vision/summarization) provider. A non-ASCII label must
        not trigger that path.
        """
        store = {
            "version": auth.AUTH_STORE_VERSION,
            "active_provider": "nous",
            "providers": {"nous": {"agent_key": "k", "label": "工作账号"}},
        }
        _write_utf8(hermes_home / "auth.json", store)

        import agent.auxiliary_client as aux

        # _read_nous_auth consults the credential pool FIRST and returns early
        # when a pool entry exists, never reaching the auth.json read. Force the
        # pool-absent path so the auth.json code under test actually runs.
        monkeypatch.setattr(aux, "_select_pool_entry", lambda _provider: (False, None))

        provider = aux._read_nous_auth()
        assert provider is not None
        assert provider.get("agent_key") == "k"

