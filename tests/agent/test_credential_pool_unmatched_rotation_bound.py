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
    def test_single_entry_pool_returns_none_immediately(
        self, tmp_path, monkeypatch
    ):
        """Single-entry OAuth pool + unmatched hint → None right away (a
        no-op rotation must not be reported as recovery)."""
        pool = _seed_pool(tmp_path, monkeypatch, [_entry(0, "pool-key")])
        assert pool.select() is not None

        result = pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="oauth-runtime-token-that-matches-nothing",
        )

        assert result is None
        # No innocent key was quarantined.
        statuses = {e.id: e.last_status for e in pool._entries}
        assert statuses["cred-0"] != "exhausted"

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

    def test_streak_resets_when_identity_matches(self, tmp_path, monkeypatch):
        """A rotation that identifies a real entry resets the streak — the
        bound only fires on CONSECUTIVE unmatched rotations."""
        pool = _seed_pool(
            tmp_path, monkeypatch,
            [_entry(0, "key-a"), _entry(1, "key-b"), _entry(2, "key-c")],
        )
        assert pool.select() is not None

        # Two unmatched rotations (streak = 2, below the 3-entry bound).
        for _ in range(2):
            assert pool.mark_exhausted_and_rotate(
                status_code=401,
                error_context={"reason": "unauthorized"},
                api_key_hint="no-match",
            ) is not None

        # A matched rotation (key-a) marks a real entry → streak resets.
        assert pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="key-a",
        ) is not None

        # A fresh unmatched episode still gets its full lap (2 available
        # entries remain, so two rotations before the bound trips).
        assert pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="no-match",
        ) is not None

    def test_streak_resets_on_normal_selection(self, tmp_path, monkeypatch):
        """select() (a fresh episode) clears any leftover streak so the next
        failure gets its full rotation budget."""
        pool = _seed_pool(
            tmp_path, monkeypatch,
            [_entry(0, "key-a"), _entry(1, "key-b")],
        )
        assert pool.select() is not None

        # Burn the streak up to (but not past) the bound.
        for _ in range(2):
            pool.mark_exhausted_and_rotate(
                status_code=401,
                error_context={"reason": "unauthorized"},
                api_key_hint="no-match",
            )

        # New turn: a successful normal selection resets the streak.
        assert pool.select() is not None
        assert pool._unmatched_rotation_streak == 0

        # The next unmatched rotation is attempt 1 of a new episode.
        assert pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="no-match",
        ) is not None

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
