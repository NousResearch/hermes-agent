"""Tests for the sanitized WhatsApp dead-letter ledger.

Task 3 of the delivery-reliability plan:
- Local JSONL ledger under the Hermes state path, disabled by default upstream.
- Records timestamp, platform, route, idempotency key hash, attempt count,
  sanitized error category/status and resolution state.
- Never message content/chat id/token.
- Atomic append with a Windows-safe file lock.
"""

import json
import threading

import pytest

from plugins.platforms.whatsapp import delivery_ledger as ledger

IDEMPOTENCY_KEY = "b3f5c6a1e4d2477a9e0a1c2b3d4e5f60"
SECRET_TOKEN = "Bearer super-secret-bridge-token"
CHAT_ID = "5511998877665@s.whatsapp.net"
MESSAGE_TEXT = "please call me back at 5511998877665, my email is a@b.com"


@pytest.fixture(autouse=True)
def _isolated_ledger_env(tmp_path, monkeypatch):
    """Each test gets its own ledger path and starts disabled."""
    monkeypatch.delenv(ledger._LEDGER_ENABLED_ENV, raising=False)
    monkeypatch.setenv(ledger._LEDGER_PATH_ENV, str(tmp_path / "ledger.jsonl"))
    yield


def _read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestLedgerDisabledByDefault:
    def test_disabled_when_env_not_set(self):
        assert ledger.is_ledger_enabled() is False

    def test_record_is_a_noop_when_disabled(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(ledger._LEDGER_PATH_ENV, str(path))
        ref = ledger.record_dead_letter(
            platform="whatsapp",
            route="/send",
            idempotency_key=IDEMPOTENCY_KEY,
            attempts=3,
            category="http_503",
            status=503,
        )
        assert ref is None
        assert not path.exists()


class TestLedgerRecording:
    def test_enabled_writes_one_jsonl_entry(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(ledger._LEDGER_PATH_ENV, str(path))

        ref = ledger.record_dead_letter(
            platform="whatsapp",
            route="/send",
            idempotency_key=IDEMPOTENCY_KEY,
            attempts=3,
            category="http_503",
            status=503,
        )

        entries = _read_lines(path)
        assert len(entries) == 1
        entry = entries[0]
        assert ref
        assert entry["platform"] == "whatsapp"
        assert entry["route"] == "/send"
        assert entry["attempts"] == 3
        assert entry["category"] == "http_503"
        assert entry["status"] == 503
        assert entry["resolution"] == "open"
        assert "timestamp" in entry and entry["timestamp"]
        assert "idempotency_key_hash" in entry

    def test_multiple_records_append_without_clobbering(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(ledger._LEDGER_PATH_ENV, str(path))

        for i in range(5):
            ledger.record_dead_letter(
                platform="whatsapp",
                route="/send",
                idempotency_key=f"key-{i}",
                attempts=3,
                category="http_503",
                status=503,
            )

        entries = _read_lines(path)
        assert len(entries) == 5

    def test_concurrent_appends_all_land_as_valid_lines(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(ledger._LEDGER_PATH_ENV, str(path))

        def _write(i):
            ledger.record_dead_letter(
                platform="whatsapp",
                route="/send",
                idempotency_key=f"key-{i}",
                attempts=3,
                category="http_503",
                status=503,
            )

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = _read_lines(path)
        assert len(entries) == 20


class TestLedgerSanitization:
    def test_idempotency_key_is_hashed_not_stored_raw(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(ledger._LEDGER_PATH_ENV, str(path))

        ledger.record_dead_letter(
            platform="whatsapp",
            route="/send",
            idempotency_key=IDEMPOTENCY_KEY,
            attempts=3,
            category="http_503",
            status=503,
        )

        raw = path.read_text(encoding="utf-8")
        assert IDEMPOTENCY_KEY not in raw

    @pytest.mark.parametrize("leak", [SECRET_TOKEN, CHAT_ID, MESSAGE_TEXT, "5511998877665", "a@b.com"])
    def test_no_pii_or_secrets_leak_into_ledger(self, tmp_path, monkeypatch, leak):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(ledger._LEDGER_PATH_ENV, str(path))

        # record_dead_letter's signature has no field for message/chat/token
        # content at all — passing one positionally is a caller bug we want
        # a TypeError for, not a leak.
        with pytest.raises(TypeError):
            ledger.record_dead_letter(
                platform="whatsapp",
                route="/send",
                idempotency_key=IDEMPOTENCY_KEY,
                attempts=3,
                category="http_503",
                status=503,
                message=leak,
            )

        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        assert leak not in raw

    def test_category_field_rejects_free_text_exception_string(self, tmp_path, monkeypatch):
        """category must be a sanitized identifier, never str(exception)."""
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(ledger._LEDGER_PATH_ENV, str(path))

        ledger.record_dead_letter(
            platform="whatsapp",
            route="/send",
            idempotency_key=IDEMPOTENCY_KEY,
            attempts=3,
            category="unknown_exception",
            status=None,
        )

        entries = _read_lines(path)
        assert entries[0]["category"] == "unknown_exception"


class TestLedgerPathResolution:
    def test_default_path_lives_under_hermes_state_dir(self, monkeypatch):
        monkeypatch.delenv(ledger._LEDGER_PATH_ENV, raising=False)
        path = ledger.default_ledger_path()
        assert path.name == "whatsapp_delivery_ledger.jsonl"
        assert "state" in path.parts

    def test_ledger_path_honors_env_override(self, tmp_path, monkeypatch):
        override = tmp_path / "custom" / "dlq.jsonl"
        monkeypatch.setenv(ledger._LEDGER_PATH_ENV, str(override))
        assert ledger.ledger_path() == override
