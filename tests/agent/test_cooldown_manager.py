"""Regression coverage for persistent provider cooldowns."""

from __future__ import annotations

import json
import multiprocessing
import time

from agent.cooldown_manager import CooldownManager, build_cooldown_key


def _record_concurrent_failure(storage_path: str, key: str, barrier) -> None:
    manager = CooldownManager(base_seconds=10.0, storage_path=storage_path)
    barrier.wait()
    manager.mark_failure(key, "rate_limit")


def test_rate_limit_cooldown_escalates_and_persists_without_raw_api_key(tmp_path):
    storage_path = tmp_path / "cooldowns.json"
    manager = CooldownManager(
        base_seconds=10.0,
        multiplier=2.0,
        max_seconds=100.0,
        storage_path=storage_path,
    )
    key = build_cooldown_key("OpenAI", "«redacted:sk-…»", "rate_limit")

    assert key == build_cooldown_key("openai", "«redacted:sk-…»", "rate_limit")
    assert manager.mark_failure(key, "rate_limit") == 10.0
    assert manager.mark_failure(key, "rate_limit") == 20.0
    assert manager.is_cooling(key)

    persisted = json.loads(storage_path.read_text(encoding="utf-8"))
    assert "«redacted:sk-…»" not in json.dumps(persisted)
    reloaded = CooldownManager(storage_path=storage_path)
    assert reloaded.is_cooling(key)


def test_stale_managers_preserve_and_escalate_persisted_cooldowns(tmp_path):
    storage_path = tmp_path / "cooldowns.json"
    first = CooldownManager(
        base_seconds=10.0,
        multiplier=2.0,
        storage_path=storage_path,
    )
    # Construct this manager before the first write to model another process.
    second = CooldownManager(
        base_seconds=10.0,
        multiplier=2.0,
        storage_path=storage_path,
    )
    primary_key = build_cooldown_key("openai", "first-secret", "rate_limit")
    other_key = build_cooldown_key("anthropic", "second-secret", "rate_limit")

    assert first.mark_failure(primary_key, "rate_limit") == 10.0
    assert second.mark_failure(primary_key, "rate_limit") == 20.0
    assert second.mark_failure(other_key, "rate_limit") == 10.0

    reloaded = CooldownManager(storage_path=storage_path)
    assert reloaded.get_all_states()[primary_key]["count"] == 2
    assert reloaded.is_cooling(other_key)


def test_separate_processes_preserve_each_others_cooldowns(tmp_path):
    storage_path = tmp_path / "cooldowns.json"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    keys = (
        build_cooldown_key("openai", "first-secret", "rate_limit"),
        build_cooldown_key("anthropic", "second-secret", "rate_limit"),
    )
    processes = [
        context.Process(target=_record_concurrent_failure, args=(str(storage_path), key, barrier))
        for key in keys
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    states = CooldownManager(storage_path=storage_path).get_all_states()
    assert set(states) == set(keys)


def test_running_manager_observes_external_write_and_clear(tmp_path):
    storage_path = tmp_path / "cooldowns.json"
    reader = CooldownManager(base_seconds=10.0, storage_path=storage_path)
    writer = CooldownManager(base_seconds=10.0, storage_path=storage_path)
    key = build_cooldown_key("openai", "secret", "rate_limit")

    writer.mark_failure(key, "rate_limit")
    assert reader.is_cooling(key)
    writer.clear(key)
    assert not reader.is_cooling(key)


def test_stale_manager_does_not_resurrect_externally_cleared_cooldown(tmp_path):
    storage_path = tmp_path / "cooldowns.json"
    first = CooldownManager(base_seconds=10.0, storage_path=storage_path)
    second = CooldownManager(base_seconds=10.0, storage_path=storage_path)
    cleared_key = build_cooldown_key("openai", "first-secret", "rate_limit")
    other_key = build_cooldown_key("anthropic", "second-secret", "rate_limit")

    first.mark_failure(cleared_key, "rate_limit")
    second.clear(cleared_key)
    first.mark_failure(other_key, "rate_limit")

    assert set(CooldownManager(storage_path=storage_path).get_all_states()) == {other_key}


def test_billing_key_is_provider_scoped_and_clear_resets_backoff(tmp_path):
    manager = CooldownManager(
        base_seconds=10.0,
        multiplier=2.0,
        billing_base_hours=1.0,
        storage_path=tmp_path / "cooldowns.json",
    )
    key = build_cooldown_key("anthropic", "sk-private", "billing")

    assert key == "anthropic"
    assert manager.mark_failure(key, "billing") == 3600.0
    manager.clear(key)
    assert manager.mark_failure(key, "rate_limit") == 10.0


def test_expired_persisted_cooldown_is_discarded(tmp_path):
    storage_path = tmp_path / "cooldowns.json"
    storage_path.write_text(
        json.dumps({"openai:0123456789abcdef": {
            "count": 1, "reason": "rate_limit", "until_wall": time.time() - 1,
        }}),
        encoding="utf-8",
    )

    manager = CooldownManager(storage_path=storage_path)
    assert manager.get_all_states() == {}


def test_callable_credential_uses_provider_scope_without_invoking_or_serializing_it(tmp_path):
    class CredentialSource:
        def __call__(self):
            raise AssertionError("cooldown tracking must not invoke credential sources")

        def __str__(self):
            raise AssertionError("cooldown tracking must not serialize credential sources")

    credential = CredentialSource()
    key = build_cooldown_key("OpenAI", credential, "rate_limit")

    assert key == "openai"
    manager = CooldownManager(storage_path=tmp_path / "cooldowns.json")
    manager.mark_failure(key, "rate_limit")
    assert "CredentialSource" not in (tmp_path / "cooldowns.json").read_text(encoding="utf-8")


def test_load_discards_unknown_colonless_keys_but_keeps_provider_and_fingerprint_keys(tmp_path):
    storage_path = tmp_path / "cooldowns.json"
    future = time.time() + 60
    storage_path.write_text(
        json.dumps({
            "raw-unknown-secret": {"count": 1, "reason": "rate_limit", "until_wall": future},
            "openai": {"count": 1, "reason": "billing", "until_wall": future},
            "openai:0123456789abcdef": {"count": 1, "reason": "rate_limit", "until_wall": future},
        }),
        encoding="utf-8",
    )

    manager = CooldownManager(storage_path=storage_path)

    assert "raw-unknown-secret" not in manager.get_all_states()
    assert manager.is_cooling("openai")
    assert manager.is_cooling("openai:0123456789abcdef")
    persisted = json.loads(storage_path.read_text(encoding="utf-8"))
    assert "raw-unknown-secret" not in persisted
