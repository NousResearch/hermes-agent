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

import hermes_cli.telegram_canary as telegram_canary
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


def _passing_canary_result(payload: str) -> SendResult:
    chunks = prepare_legacy_text_chunks(payload)
    return SendResult(
        success=True,
        raw_response={
            "message_ids": [str(index) for index in range(len(chunks))],
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
        },
    )


def _init_runtime_fixture(path: Path) -> tuple[str, Path]:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    tracked = path / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=path, check=True)
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
        cwd=path,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return sha, tracked


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


def test_private_receipt_recovers_a_bounded_partial_tail(tmp_path):
    path = tmp_path / "private" / "receipt.jsonl"
    path.parent.mkdir(mode=0o700)
    path.write_bytes(b'{"schema":"hermes.telegram-canary/v1"')
    path.chmod(0o600)

    receipt = {"schema": CANARY_SCHEMA, "result": "pass"}
    append_private_receipt(path, receipt)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [receipt]
    quarantined = list(path.parent.glob(f".{path.name}.partial-*.bin"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b'{"schema":"hermes.telegram-canary/v1"'
    assert quarantined[0].stat().st_mode & 0o777 == 0o600


def test_private_receipt_frames_a_complete_missing_newline_record(tmp_path):
    path = tmp_path / "private" / "receipt.jsonl"
    path.parent.mkdir(mode=0o700)
    first = {"schema": CANARY_SCHEMA, "result": "fail"}
    second = {"schema": CANARY_SCHEMA, "result": "pass"}
    path.write_text(
        json.dumps(first, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)

    append_private_receipt(path, second)

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        first,
        second,
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX special-file boundary")
def test_private_receipt_rejects_fifo_without_blocking(tmp_path):
    path = tmp_path / "private" / "receipt.jsonl"
    path.parent.mkdir(mode=0o700)
    os.mkfifo(path, mode=0o600)
    repo_root = Path(__file__).resolve().parents[2]
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "from hermes_cli.telegram_canary import append_private_receipt; "
            f"append_private_receipt(Path({str(path)!r}), "
            "{'schema': 'hermes.telegram-canary/v1'})",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert probe.returncode != 0
    assert "regular single-link file" in probe.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX hard-link boundary")
def test_private_receipt_rejects_hardlink_without_mutating_target(tmp_path):
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    target = parent / "unrelated.txt"
    target.write_text("do-not-change\n", encoding="utf-8")
    target.chmod(0o600)
    path = parent / "receipt.jsonl"
    os.link(target, path)

    with pytest.raises(ValueError, match="regular single-link file"):
        append_private_receipt(path, {"schema": CANARY_SCHEMA})

    assert target.read_text(encoding="utf-8") == "do-not-change\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX dirfd boundary")
def test_private_receipt_rejects_ancestor_swap_without_redirect(
    tmp_path, monkeypatch
):
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    attacker = tmp_path / "attacker"
    attacker.mkdir(mode=0o700)
    attacker_receipt = attacker / "receipt.jsonl"
    attacker_receipt.write_text("attacker\n", encoding="utf-8")
    attacker_receipt.chmod(0o600)
    moved = tmp_path / "private-original"
    original_open = telegram_canary.os.open
    swapped = False

    def swap_then_open(file, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and Path(os.fsdecode(file)).name == "receipt.jsonl":
            swapped = True
            parent.rename(moved)
            parent.symlink_to(attacker, target_is_directory=True)
        if dir_fd is None:
            return original_open(file, flags, mode)
        return original_open(file, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(telegram_canary.os, "open", swap_then_open)
    with pytest.raises(ValueError, match="changed during canary file access"):
        append_private_receipt(parent / "receipt.jsonl", {"schema": CANARY_SCHEMA})

    assert attacker_receipt.read_text(encoding="utf-8") == "attacker\n"


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
    tests_init = tmp_path / "tests" / "__init__.py"
    tests_init.parent.mkdir()
    tests_init.write_text("", encoding="utf-8")
    tracked_test = tmp_path / "tests" / "test_tracked.py"
    tracked_test.write_text(
        'RUNTIME_MARKER = "tracked-source"\n\n'
        "def test_value():\n    assert 1 == 1\n",
        encoding="utf-8",
    )
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "add",
            "tracked.txt",
            "tracked_module.py",
            "tests/__init__.py",
            "tests/test_tracked.py",
            ".gitignore",
        ],
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
    subprocess.run(
        [sys.executable, "-m", "pytest", str(tracked_test), "-q"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    pytest_caches = list((tracked_test.parent / "__pycache__").glob("*-pytest-*.pyc"))
    assert len(pytest_caches) == 1
    malicious_pytest_source = tmp_path / "malicious_pytest_cache.py"
    malicious_pytest_source.write_text(
        'RUNTIME_MARKER = "forged-pytest-cache"\n',
        encoding="utf-8",
    )
    py_compile.compile(
        str(malicious_pytest_source),
        cfile=str(pytest_caches[0]),
        doraise=True,
    )
    malicious_pytest_source.unlink()
    assert verify_running_runtime_sha(sha, source_root=tmp_path) is True
    standard_import_env = os.environ.copy()
    standard_import_env["PYTHONPATH"] = str(tmp_path)
    standard_import = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from tests import test_tracked; "
            "print(test_tracked.RUNTIME_MARKER); "
            "print(test_tracked.__cached__)",
        ],
        cwd=tmp_path,
        env=standard_import_env,
        check=True,
        capture_output=True,
        text=True,
    )
    standard_lines = standard_import.stdout.strip().splitlines()
    assert standard_lines[0] == "tracked-source"
    assert Path(standard_lines[1]).name == (
        f"test_tracked.{sys.implementation.cache_tag}.pyc"
    )
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

    stale_source = tmp_path / "stale_cache_source.py"
    stale_source.write_text(
        'PWNED = "stale ordinary cache payload must not execute"\n',
        encoding="utf-8",
    )
    for invalidation_mode in (
        py_compile.PycInvalidationMode.TIMESTAMP,
        py_compile.PycInvalidationMode.CHECKED_HASH,
    ):
        py_compile.compile(
            str(stale_source),
            cfile=str(ordinary_cache),
            doraise=True,
            invalidation_mode=invalidation_mode,
        )
        assert verify_running_runtime_sha(sha, source_root=tmp_path) is False
        stale_source.unlink()
        assert verify_running_runtime_sha(sha, source_root=tmp_path) is True
        stale_import = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import tracked_module; "
                "print(getattr(tracked_module, 'PWNED', 0), tracked_module.VALUE)",
            ],
            cwd=tmp_path,
            env=standard_import_env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert stale_import.stdout.strip() == "0 1"
        if invalidation_mode is py_compile.PycInvalidationMode.CHECKED_HASH:
            never_check_env = standard_import_env.copy()
            repo_root = Path(__file__).resolve().parents[2]
            never_check_env["PYTHONPATH"] = (
                f"{tmp_path}{os.pathsep}{repo_root}"
            )
            never_check = subprocess.run(
                [
                    sys.executable,
                    "--check-hash-based-pycs",
                    "never",
                    "-B",
                    "-c",
                    "from pathlib import Path; "
                    "from hermes_cli.telegram_canary import "
                    "verify_running_runtime_sha; "
                    "import tracked_module; "
                    "print(bool(getattr(tracked_module, 'PWNED', 0)), "
                    "getattr(tracked_module, 'VALUE', 0), "
                    "verify_running_runtime_sha("
                    f"'{sha}', source_root=Path.cwd()))",
                ],
                cwd=tmp_path,
                env=never_check_env,
                check=True,
                capture_output=True,
                text=True,
            )
            assert never_check.stdout.strip() == "True 0 False"
        stale_source.write_text(
            'PWNED = "stale ordinary cache payload must not execute"\n',
            encoding="utf-8",
        )

    py_compile.compile(
        str(stale_source),
        cfile=str(ordinary_cache),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    stale_source.unlink()
    unchecked_import = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import tracked_module; "
            "print(bool(getattr(tracked_module, 'PWNED', 0)), "
            "getattr(tracked_module, 'VALUE', 0))",
        ],
        cwd=tmp_path,
        env=standard_import_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert unchecked_import.stdout.strip() == "True 0"
    assert verify_running_runtime_sha(sha, source_root=tmp_path) is False


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixture")
def test_runtime_sha_uses_trusted_git_and_scrubs_git_environment(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    sha, _ = _init_runtime_fixture(repo)
    hostile = tmp_path / "hostile-bin"
    hostile.mkdir()
    marker = tmp_path / "hostile-git-ran"
    fake_git = hostile / "git"
    fake_git.write_text(
        f"#!/bin/sh\nprintf ran > {marker}\nexit 0\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(hostile))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "forged.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "forged-tree"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "forged-config"))

    assert verify_running_runtime_sha(sha, source_root=repo) is True
    assert not marker.exists()


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_runtime_sha_attests_tracked_bytes_hidden_by_index_flags(tmp_path, flag):
    repo = tmp_path / "repo"
    sha, tracked = _init_runtime_fixture(repo)
    subprocess.run(
        ["git", "update-index", flag, "tracked.py"],
        cwd=repo,
        check=True,
    )
    tracked.write_text("PWNED = 1\n", encoding="utf-8")

    assert verify_running_runtime_sha(sha, source_root=repo) is False


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


def test_finalize_requires_the_exact_pending_claim_before_append(tmp_path):
    state_path = tmp_path / "private" / "state.json"
    receipt_path = tmp_path / "private" / "receipt.jsonl"
    claim, _ = claim_live_canary(
        runtime_sha="f" * 40,
        destination_alias="owner",
        message_id="message",
        update_id="update",
        state_path=state_path,
    )
    assert claim is not None
    state_path.write_text(
        json.dumps({"schema": CANARY_SCHEMA, "runs": {}}),
        encoding="utf-8",
    )
    state_path.chmod(0o600)
    payload = build_live_payload("f" * 40)

    with pytest.raises(ValueError, match="exact pending canary claim"):
        finalize_live_canary(
            claim,
            result=_passing_canary_result(payload),
            payload=payload,
            authentication={"source_authorized": True},
            duplicate_probe_suppressed=True,
            receipt_path=receipt_path,
            state_path=state_path,
        )

    assert not receipt_path.exists()


def test_finalize_is_idempotent_and_recovers_after_append_before_state(tmp_path, monkeypatch):
    state_path = tmp_path / "private" / "state.json"
    receipt_path = tmp_path / "private" / "receipt.jsonl"
    claim, _ = claim_live_canary(
        runtime_sha="1" * 40,
        destination_alias="owner",
        message_id="message",
        update_id="update",
        state_path=state_path,
        created_at="2026-07-15T12:00:00Z",
    )
    assert claim is not None
    payload = build_live_payload("1" * 40)
    result = _passing_canary_result(payload)
    original_write_state = telegram_canary._write_state
    crashed = False

    def crash_after_receipt(path, state):
        nonlocal crashed
        entry = state["runs"][claim.idempotency_key_sha256]
        if not crashed and entry.get("status") == "receipt_written":
            crashed = True
            raise OSError("synthetic crash after receipt append")
        original_write_state(path, state)

    monkeypatch.setattr(telegram_canary, "_write_state", crash_after_receipt)
    with pytest.raises(OSError, match="synthetic crash"):
        finalize_live_canary(
            claim,
            result=result,
            payload=payload,
            authentication={"source_authorized": True},
            duplicate_probe_suppressed=True,
            receipt_path=receipt_path,
            state_path=state_path,
        )
    assert len(receipt_path.read_text(encoding="utf-8").splitlines()) == 1

    monkeypatch.setattr(telegram_canary, "_write_state", original_write_state)
    recovered, recovered_sha = finalize_live_canary(
        claim,
        result=result,
        payload=payload,
        authentication={"source_authorized": True},
        duplicate_probe_suppressed=True,
        receipt_path=receipt_path,
        state_path=state_path,
    )
    repeated, repeated_sha = finalize_live_canary(
        claim,
        result=result,
        payload=payload,
        authentication={"source_authorized": True},
        duplicate_probe_suppressed=True,
        receipt_path=receipt_path,
        state_path=state_path,
    )

    assert recovered == repeated
    assert recovered_sha == repeated_sha
    assert len(receipt_path.read_text(encoding="utf-8").splitlines()) == 1


def test_finalize_rejects_changed_receipt_after_append_before_state(tmp_path, monkeypatch):
    state_path = tmp_path / "private" / "state.json"
    receipt_path = tmp_path / "private" / "receipt.jsonl"
    claim, _ = claim_live_canary(
        runtime_sha="2" * 40,
        destination_alias="owner",
        message_id="message",
        update_id="update",
        state_path=state_path,
        created_at="2026-07-15T12:00:00Z",
    )
    assert claim is not None
    payload = build_live_payload("2" * 40)
    result = _passing_canary_result(payload)
    original_write_state = telegram_canary._write_state
    crashed = False

    def crash_after_receipt(path, state):
        nonlocal crashed
        entry = state["runs"][claim.idempotency_key_sha256]
        if not crashed and entry.get("status") == "receipt_written":
            crashed = True
            raise OSError("synthetic crash after receipt append")
        original_write_state(path, state)

    monkeypatch.setattr(telegram_canary, "_write_state", crash_after_receipt)
    with pytest.raises(OSError, match="synthetic crash"):
        finalize_live_canary(
            claim,
            result=result,
            payload=payload,
            authentication={"source_authorized": True},
            duplicate_probe_suppressed=True,
            receipt_path=receipt_path,
            state_path=state_path,
        )

    forged = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged["result"] = "fail"
    receipt_path.write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)

    monkeypatch.setattr(telegram_canary, "_write_state", original_write_state)
    with pytest.raises(ValueError, match="conflicts with its pending claim"):
        finalize_live_canary(
            claim,
            result=result,
            payload=payload,
            authentication={"source_authorized": True},
            duplicate_probe_suppressed=True,
            receipt_path=receipt_path,
            state_path=state_path,
        )
