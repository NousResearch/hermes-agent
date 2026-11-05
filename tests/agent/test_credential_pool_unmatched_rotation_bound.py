"""#70401: the unmatched-identity rotation branch in
``mark_exhausted_and_rotate()`` must be bounded and must not write cooldowns
onto innocent healthy keys.

With OAuth-token auth (provider ``nous``), the upstream 401's ``api_key_hint``
never matches any pool entry's ``runtime_api_key`` — the wrapper's runtime key
rotates. The no-match branch deliberately marks nothing exhausted (marking
would quarantine an innocent healthy key for the full cooldown TTL) and hands
back a fresh selection. But because nothing is ever marked, the pool can never
converge to the "no available entries" state: with the old code the caller
retried the same dead token forever (~6/sec), starving the event loop so chat
``/stop`` interrupts were never processed; only killing the gateway ended it.

The fix keeps the don't-mark-innocent-keys semantics (see the breaker/cooldown
design notes in ``mark_exhausted_and_rotate`` — the pool only trips on
confirmed-empty state, and no cooldown is invented here) but BOUNDS the
branch: after one full lap of the available entries with no recovery, the
rotation returns None so the caller surfaces the error / activates fallback.
Healthy keys carry no cooldown and are immediately available next turn — this
does not reintroduce hammering, it stops it.
"""
import json

import pytest


def _seed_pool(tmp_path, monkeypatch, entries, provider="openrouter"):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {provider: entries}})
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool

    return load_pool(provider)


def _entry(idx, key):
    return {
        "id": f"cred-{idx}",
        "label": f"key-{idx}",
        "auth_type": "api_key",
        "priority": idx,
        "source": "manual",
        "access_token": key,
    }


class TestUnmatchedHintRotationIsBounded:

    def test_multi_entry_pool_unmatched_hint_loop_terminates(
        self, tmp_path, monkeypatch
    ):
        """Multi-entry pool: consecutive unmatched-hint rotations must reach
        None within one lap of the pool instead of ping-ponging forever."""
        pool = _seed_pool(
            tmp_path, monkeypatch,
            [_entry(0, "key-a"), _entry(1, "key-b"), _entry(2, "key-c")],
        )
        assert pool.select() is not None

        results = []
        for _ in range(10):  # caller's retry loop
            nxt = pool.mark_exhausted_and_rotate(
                status_code=401,
                error_context={"reason": "unauthorized"},
                api_key_hint="oauth-runtime-token-that-matches-nothing",
            )
            results.append(nxt)
            if nxt is None:
                break
        else:
            pytest.fail(
                "unbounded 401 retry loop: 10 unmatched-hint rotations never "
                "returned None (#70401)"
            )

        # Bounded within one lap (3 available entries → at most 3 rotations
        # before the streak trips).
        assert len(results) <= 4
        assert results[-1] is None
        # The escape must NOT have invented cooldowns for healthy keys.
        statuses = {e.id: e.last_status for e in pool._entries}
        assert all(status != "exhausted" for status in statuses.values()), (
            f"innocent keys were quarantined: {statuses}"
        )



    def test_matched_hint_path_unaffected(self, tmp_path, monkeypatch):
        """Regression guard: the normal matched-hint path still marks the
        failing entry and rotates to the healthy one."""
        pool = _seed_pool(
            tmp_path, monkeypatch,
            [_entry(0, "key-healthy"), _entry(1, "key-failed")],
        )
        assert pool.select().access_token == "key-healthy"

        nxt = pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="key-failed",
        )

        statuses = {e.id: e.last_status for e in pool._entries}
        assert statuses["cred-1"] == "exhausted"
        assert statuses["cred-0"] != "exhausted"
        assert nxt is not None
        assert nxt.access_token == "key-healthy"


class TestUnmatchedRotationWithBenchedSiblings:
    """#127722: the single-entry decline in _rotate_unmatched must key on
    pool size, not availability.

    The old guard (len(available) == 1 → None) conflated a true
    single-entry pool with a multi-entry pool whose siblings are benched: the
    one healthy entry was never handed back, the streak was reset so the
    #70401 cap could never engage, and the caller fell back while a healthy
    credential sat idle.
    """

    def _seed_with_benched_siblings(self, tmp_path, monkeypatch):
        import time

        now = time.time()
        entries = [_entry(0, "key-a")]
        for idx, key in ((1, "key-b"), (2, "key-c")):
            benched = _entry(idx, key)
            benched.update(
                {
                    "last_status": "exhausted",
                    "last_status_at": now,
                    "last_error_code": 429,
                }
            )
            entries.append(benched)
        pool = _seed_pool(tmp_path, monkeypatch, entries)
        assert [e.id for e in pool._available_entries()[0]] == ["cred-0"]
        return pool

    def test_benched_multi_entry_pool_rotates_to_healthy_sibling(
        self, tmp_path, monkeypatch
    ):
        """One available entry out of three is still a rotation target: the
        failed identity matched nothing, so any pool entry differs from it."""
        pool = self._seed_with_benched_siblings(tmp_path, monkeypatch)

        nxt = pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="oauth-runtime-token-that-matches-nothing",
        )

        assert nxt is not None
        assert nxt.id == "cred-0"
        # The streak must survive so the #70401 cap can engage on repeats.
        assert pool._unmatched_rotation_streak == 1

    def test_benched_multi_entry_pool_unmatched_loop_still_bounded(
        self, tmp_path, monkeypatch
    ):
        """Repeated unmatched rotations on the benched pool must terminate
        via the streak cap without inventing cooldowns."""
        pool = self._seed_with_benched_siblings(tmp_path, monkeypatch)

        results = []
        for _ in range(10):
            nxt = pool.mark_exhausted_and_rotate(
                status_code=401,
                error_context={"reason": "unauthorized"},
                api_key_hint="oauth-runtime-token-that-matches-nothing",
            )
            results.append(nxt)
            if nxt is None:
                break
        else:
            import pytest

            pytest.fail("unmatched rotations never terminated on benched pool")

        assert results[-1] is None
        assert len(results) <= 3  # one available entry → cap trips fast
        assert pool._entries[0].last_status != "exhausted"

    def test_true_single_entry_pool_still_declines(self, tmp_path, monkeypatch):
        """A pool with exactly one entry keeps the old decline: handing it
        back would retry the same 401 forever."""
        pool = _seed_pool(tmp_path, monkeypatch, [_entry(0, "key-only")])

        nxt = pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="oauth-runtime-token-that-matches-nothing",
        )

        assert nxt is None
