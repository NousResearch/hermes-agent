"""Synthetic, non-private Telegram delivery canary tests."""

import hashlib
import json
import marshal
import os
import py_compile
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from hermes_cli.telegram_canary import (
    CANARY_SCHEMA,
    append_private_receipt,
    build_live_payload,
    claim_live_canary,
    finalize_live_canary,
    strict_single_owner_id,
    verify_running_runtime_sha,
)
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    prepare_legacy_text_chunks,
)


class _Message:
    def __init__(self, message_id: int):
        self.message_id = message_id


class _Bot:
    def __init__(self):
        self.calls = 0

    async def send_message(self, **_kwargs):
        self.calls += 1
        return _Message(1000 + self.calls)


@pytest.fixture
def strict_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner-1")
    for key in (
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_strict_owner_rejects_wildcard_and_multiple_users(strict_allowlist, monkeypatch):
    assert strict_single_owner_id() == "owner-1"
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner-1,owner-2")
    assert strict_single_owner_id() is None
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    assert strict_single_owner_id() is None


def test_private_receipt_is_append_only(tmp_path):
    path = tmp_path / "private" / "receipt.jsonl"
    first = {
        "schema": CANARY_SCHEMA,
        "checks": {"delivery": {"message_ids": ["1", "2"]}},
    }
    second = {
        "schema": CANARY_SCHEMA,
        "checks": {"delivery": {"message_ids": ["3"]}},
    }

    append_private_receipt(path, first)
    append_private_receipt(path, second)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first, second]
    assert path.stat().st_mode & 0o777 == 0o600


def test_concurrent_claims_do_not_overwrite_state(tmp_path):
    state_path = tmp_path / "private" / "telegram-canary-state.json"

    def claim(index: int):
        return claim_live_canary(
            runtime_sha="a" * 40,
            destination_alias="owner",
            message_id=f"message-{index}",
            update_id=f"update-{index}",
            state_path=state_path,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(24)))

    assert all(item is not None and duplicate is False for item, duplicate in results)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(state["runs"]) == 24


def test_old_canary_claim_is_never_evicted_from_replay_protection(tmp_path):
    state_path = tmp_path / "private" / "telegram-canary-state.json"
    first, duplicate = claim_live_canary(
        runtime_sha="b" * 40,
        destination_alias="owner",
        message_id="old-message",
        update_id="old-update",
        state_path=state_path,
        created_at="2000-01-01T00:00:00Z",
    )
    assert first is not None and duplicate is False

    for index in range(100):
        claim, duplicate = claim_live_canary(
            runtime_sha="b" * 40,
            destination_alias="owner",
            message_id=f"new-message-{index}",
            update_id=f"new-update-{index}",
            state_path=state_path,
        )
        assert claim is not None and duplicate is False

    replay, duplicate = claim_live_canary(
        runtime_sha="b" * 40,
        destination_alias="owner",
        message_id="old-message",
        update_id="old-update",
        state_path=state_path,
    )
    assert replay is None
    assert duplicate is True


def test_runtime_sha_verification_requires_exact_clean_checkout(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    tracked_module = tmp_path / "tracked_module.py"
    tracked_module.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt", "tracked_module.py", ".gitignore"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Hermes Test",
            "-c",
            "user.email=hermes@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert verify_running_runtime_sha(sha, source_root=tmp_path) is True
    assert verify_running_runtime_sha("f" * 40, source_root=tmp_path) is False
    ordinary_cache = Path(
        py_compile.compile(str(tracked_module), doraise=True)
    )
    assert ordinary_cache.parent.name == "__pycache__"
    assert verify_running_runtime_sha(sha, source_root=tmp_path) is True

    # Marshal byte streams are not canonical: different valid encodings can
    # load to the exact same code object. The verifier must accept that normal
    # tracked-source cache while still rejecting changed executable content.
    ordinary_bytes = ordinary_cache.read_bytes()
    equivalent_code = compile(
        tracked_module.read_bytes(),
        str(tracked_module),
        "exec",
        dont_inherit=True,
        optimize=0,
    )
    alternate_payload = marshal.dumps(equivalent_code, 3)
    assert alternate_payload != ordinary_bytes[16:]
    ordinary_cache.write_bytes(ordinary_bytes[:16] + alternate_payload)
    assert marshal.loads(alternate_payload) == equivalent_code
    assert verify_running_runtime_sha(sha, source_root=tmp_path) is True
    ordinary_cache.unlink()
    ordinary_cache.parent.rmdir()

    original_source = tracked_module.read_bytes()
    original_stat = tracked_module.stat()
    malicious_source = b"PWNED = 1\n"
    assert len(malicious_source) == len(original_source)
    tracked_module.write_bytes(malicious_source)
    os.utime(
        tracked_module,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    forged_cache = Path(py_compile.compile(str(tracked_module), doraise=True))
    tracked_module.write_bytes(original_source)
    os.utime(
        tracked_module,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    probe_env = os.environ.copy()
    probe_env["PYTHONPATH"] = str(tmp_path)
    probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import tracked_module; "
            "print(getattr(tracked_module, 'PWNED', 0), "
            "getattr(tracked_module, 'VALUE', 0))",
        ],
        cwd=tmp_path,
        env=probe_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "1 0"
    assert verify_running_runtime_sha(sha, source_root=tmp_path) is False
    forged_cache.unlink()
    forged_cache.parent.rmdir()

    (tmp_path / "sitecustomize.py").write_text(
        "raise RuntimeError('untracked runtime injection')\n",
        encoding="utf-8",
    )
    assert verify_running_runtime_sha(sha, source_root=tmp_path) is False
    (tmp_path / "sitecustomize.py").unlink()
    pyc_source = tmp_path / "ignored_pyc_source.py"
    pyc_source.write_text("print('IGNORED_PYC_EXECUTED')\n", encoding="utf-8")
    py_compile.compile(
        str(pyc_source),
        cfile=str(tmp_path / "sitecustomize.pyc"),
        doraise=True,
    )
    pyc_source.unlink()
    assert verify_running_runtime_sha(sha, source_root=tmp_path) is False
    (tmp_path / "sitecustomize.pyc").unlink()
    tracked.write_text("dirty\n", encoding="utf-8")
    assert verify_running_runtime_sha(sha, source_root=tmp_path) is False


@pytest.mark.asyncio
async def test_partial_multichunk_send_never_retries_full_payload(monkeypatch):
    from telegram.error import NetworkError

    class _PartialBot:
        def __init__(self):
            self.calls = 0

        async def send_message(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Message(1001)
            raise NetworkError("synthetic later-chunk connection failure")

    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock()
    )
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", AsyncMock())
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
    bot = _PartialBot()
    adapter._bot = bot

    result = await adapter._send_with_retry("@fixture", "A" * 5000)

    assert result.success is False
    assert result.retryable is False
    assert result.raw_response["partial_send"] is True
    assert result.raw_response["delivered_chunks"] == 1
    assert result.raw_response["total_chunks"] == 2
    assert result.raw_response["message_ids"] == ["1001"]
    # One successful first chunk plus the adapter's three attempts for chunk 2.
    # The base retry layer must not start the payload from chunk 1 again.
    assert bot.calls == 4


@pytest.mark.asyncio
async def test_live_retry_probe_stays_inside_existing_telegram_send(monkeypatch):
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock()
    )
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
    bot = _Bot()
    adapter._bot = bot

    result = await adapter.send(
        "@fixture",
        build_live_payload("c" * 40),
        metadata={"hermes_synthetic_pre_send_connect_failure": True},
    )

    assert result.success is True
    assert result.raw_response["synthetic_pre_send_failures"] == 1
    assert result.raw_response["attempt_counts"][0] == 2
    assert result.raw_response["chunk_count"] >= 2
    # The injected failure happens before the Bot API call. Every real Bot
    # call therefore corresponds to one successfully acknowledged chunk.
    assert bot.calls == result.raw_response["chunk_count"]


def test_live_receipt_hashes_ids_and_marks_external_acceptance_non_qualifying(tmp_path):
    state_path = tmp_path / "private" / "telegram-canary-state.json"
    receipt_path = tmp_path / "private" / "telegram-canary.jsonl"
    claim, duplicate = claim_live_canary(
        runtime_sha="d" * 40,
        destination_alias="owner",
        message_id="raw-live-message-id-never-persist",
        update_id="raw-live-update-id-never-persist",
        state_path=state_path,
        created_at="2026-07-15T12:00:00Z",
    )
    assert duplicate is False
    assert claim is not None
    duplicate_claim, duplicate = claim_live_canary(
        runtime_sha="d" * 40,
        destination_alias="owner",
        message_id="raw-live-message-id-never-persist",
        update_id="raw-live-update-id-never-persist",
        state_path=state_path,
        created_at="2026-07-15T12:00:00Z",
    )
    assert duplicate_claim is None
    assert duplicate is True

    payload = build_live_payload("d" * 40)
    chunks = prepare_legacy_text_chunks(payload)
    result = SendResult(
        success=True,
        message_id="raw-delivery-id-never-persist",
        raw_response={
            "message_ids": [
                f"raw-delivery-id-{index}" for index in range(len(chunks))
            ],
            "attempt_counts": [2, *([1] * (len(chunks) - 1))],
            "synthetic_pre_send_failures": 1,
            "chunk_count": len(chunks),
            "chunk_sha256": [
                "sha256:" + hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                for chunk in chunks
            ],
            "chunk_utf16_units": [
                len(chunk.encode("utf-16-le")) // 2 for chunk in chunks
            ],
            "thread_fallback": False,
        },
    )
    receipt, file_sha = finalize_live_canary(
        claim,
        result=result,
        payload=payload,
        authentication={
            "gateway_path_exercised": True,
            "strict_single_owner_allowlist": True,
            "owner_allowed": True,
            "unknown_denied": True,
            "source_authorized": True,
        },
        duplicate_probe_suppressed=True,
        receipt_path=receipt_path,
        state_path=state_path,
    )

    assert receipt["result"] == "pass"
    assert receipt["private_data"] is False
    assert receipt["qualifies_for_external_acceptance"] is False
    assert receipt["checks"]["idempotency"]["scope"] == "canary_producer"
    assert receipt["checks"]["delivery"]["confirmation"] == "api_acknowledged"
    assert all(
        item.startswith("sha256:")
        for item in receipt["checks"]["delivery"]["message_id_hashes"]
    )
    persisted = receipt_path.read_text(encoding="utf-8") + state_path.read_text(
        encoding="utf-8"
    )
    assert "raw-live-message-id-never-persist" not in persisted
    assert "raw-live-update-id-never-persist" not in persisted
    assert "raw-delivery-id-never-persist" not in persisted
    assert len(file_sha) == 64
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert state_path.stat().st_mode & 0o777 == 0o600


def test_live_receipt_rejects_fabricated_chunk_hashes(tmp_path):
    state_path = tmp_path / "private" / "state.json"
    claim, _ = claim_live_canary(
        runtime_sha="e" * 40,
        destination_alias="owner",
        message_id="message",
        update_id="update",
        state_path=state_path,
    )
    payload = build_live_payload("e" * 40)
    chunks = prepare_legacy_text_chunks(payload)
    result = SendResult(
        success=True,
        raw_response={
            "message_ids": [str(index) for index in range(len(chunks))],
            "attempt_counts": [2, *([1] * (len(chunks) - 1))],
            "synthetic_pre_send_failures": 1,
            "chunk_count": len(chunks),
            "chunk_sha256": ["sha256:" + ("0" * 64)] * len(chunks),
            "chunk_utf16_units": [
                len(chunk.encode("utf-16-le")) // 2 for chunk in chunks
            ],
        },
    )

    receipt, _ = finalize_live_canary(
        claim,
        result=result,
        payload=payload,
        authentication={
            "gateway_path_exercised": True,
            "strict_single_owner_allowlist": True,
            "owner_allowed": True,
            "unknown_denied": True,
            "source_authorized": True,
        },
        duplicate_probe_suppressed=True,
        receipt_path=tmp_path / "private" / "receipt.jsonl",
        state_path=state_path,
    )

    assert receipt["result"] == "fail"
    assert receipt["checks"]["length"]["all_chunks_acknowledged"] is False
