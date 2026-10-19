"""In-memory secret-link store: hashed tokens, one-time state, expiry, limits and cleanup."""

import hmac
import re

import pytest

from gateway import secret_link_store as store_mod
from gateway.secret_link_store import (
    LINK_TTL_SECONDS, MAX_FAILED_SUBMITS, MAX_OPEN_LINKS_PER_CHAT, SecretLinkStore)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


async def _noop(_text):
    return None


def _create(store, chat="telegram:1", on_done=None, suggested="KEY"):
    return store.create(chat_key=chat, owner_home=None, suggested_key=suggested, notify=_noop, on_done=on_done)


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def store(clock):
    return SecretLinkStore(clock=clock)


def test_token_is_a_43_char_urlsafe_value_and_only_its_hash_is_stored(store):
    token, record = _create(store)
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
    assert store.lookup(token) is record
    stored = repr(vars(record)) + repr(store._records)
    assert token not in stored
    assert repr(record) == "<SecretLinkRequest>"


def test_lookup_compares_with_compare_digest(store, monkeypatch):
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(store_mod.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    token, record = _create(store)
    assert store.lookup(token) is record
    assert calls


@pytest.mark.parametrize("bad", ["", "short", "x" * 42, "x" * 44, "!" * 43, "a/" * 21 + "a"])
def test_malformed_and_unknown_tokens_find_nothing(store, bad):
    _create(store)
    assert store.lookup(bad) is None
    assert store.lookup("A" * 43) is None


def test_expired_lookup_marks_expired_removes_and_reports_failure(store, clock):
    done = []
    token, record = _create(store, on_done=done.append)
    clock.now += LINK_TTL_SECONDS + 1
    assert store.lookup(token) is None
    assert record.state == "expired"
    assert record not in store._records
    assert done == [False]


def test_link_is_valid_until_its_ttl(store, clock):
    token, record = _create(store)
    clock.now += LINK_TTL_SECONDS - 1
    assert store.lookup(token) is record


def test_claim_is_a_one_time_compare_and_set(store):
    token, record = _create(store)
    assert store.claim(record) is True
    assert store.claim(record) is False
    assert store.lookup(token) is None


def test_release_reopens_a_claimed_record(store):
    token, record = _create(store)
    store.claim(record)
    store.release(record)
    assert store.lookup(token) is record


def test_finish_reports_the_outcome_once(store):
    done = []
    _token, record = _create(store, on_done=done.append)
    store.claim(record)
    store.finish(record, True)
    store.finish(record, False)
    assert done == [True]


def test_failed_submits_expire_the_record_at_the_limit(store):
    done = []
    token, record = _create(store, on_done=done.append)
    for _ in range(MAX_FAILED_SUBMITS - 1):
        assert store.record_failure(record) is False
    assert store.lookup(token) is record
    assert store.record_failure(record) is True
    assert store.lookup(token) is None
    assert done == [False]


def test_nonce_rotates_and_is_bound_to_the_record(store):
    _token, record = _create(store)
    first = store.rotate_nonce(record)
    assert store.nonce_matches(record, first)
    second = store.rotate_nonce(record)
    assert first != second
    assert not store.nonce_matches(record, first)
    assert store.nonce_matches(record, second)
    assert not store.nonce_matches(record, "")


def test_open_links_per_chat_are_capped_and_independent(store):
    tokens = [_create(store)[0] for _ in range(MAX_OPEN_LINKS_PER_CHAT)]
    assert _create(store) is None
    assert _create(store, chat="telegram:2") is not None
    assert all(store.lookup(t) is not None for t in tokens)


def test_sweep_removes_used_and_expired_records(store, clock):
    _t1, used = _create(store)
    store.claim(used)
    _t2, stale = _create(store)
    t3, live = _create(store)
    clock.now += 1
    stale.expires_at = clock.now - 1
    store.sweep()
    assert store._records == [live]
    assert store.lookup(t3) is live


def test_clear_drops_everything_and_reports_open_waiters(store):
    done = []
    _create(store, on_done=done.append)
    store.clear()
    assert store._records == []
    assert done == [False]


def test_client_rate_limit_is_ten_per_minute(store, clock):
    assert all(store.allow_client("1.2.3.4") for _ in range(10))
    assert store.allow_client("1.2.3.4") is False
    assert store.allow_client("5.6.7.8") is True
    clock.now += 61
    assert store.allow_client("1.2.3.4") is True
