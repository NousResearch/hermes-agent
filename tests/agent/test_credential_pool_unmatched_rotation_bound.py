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


    def test_select_between_rotations_does_not_defeat_the_bound(
        self, tmp_path, monkeypatch
    ):
        """#83447: the agent flow calls select() between consecutive 401
        rotations (rotate -> swap -> retry -> re-select). The old select()
        reset wiped the streak every cycle, so a pool whose entries all fail
        401 with an unmatched hint looped forever. The bound must survive
        select() calls within the retry window."""
        pool = _seed_pool(
            tmp_path, monkeypatch,
            [_entry(0, "key-a"), _entry(1, "key-b")],
        )
        results = []
        for _ in range(10):  # caller's retry loop, re-selecting each time
            pool.select()  # agent re-selects between retries
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
                "unbounded 401 retry loop: select() reset the streak every "
                "cycle (#83447)"
            )
        assert results[-1] is None
        # The escape must NOT have invented cooldowns for healthy keys.
        statuses = {e.id: e.last_status for e in pool._entries}
        assert all(status != "exhausted" for status in statuses.values())


    def test_streak_decays_after_window(self, tmp_path, monkeypatch):
        """Rotations separated by more than the window are not consecutive:
        a stale streak must not trip the bound for a genuinely new problem."""
        pool = _seed_pool(
            tmp_path, monkeypatch,
            [_entry(0, "key-a"), _entry(1, "key-b"), _entry(2, "key-c")],
        )
        # Two rotations within the window accumulate...
        for _ in range(2):
            pool.mark_exhausted_and_rotate(
                status_code=401,
                error_context={"reason": "unauthorized"},
                api_key_hint="no-match",
            )
        assert pool._unmatched_rotation_streak == 2
        # ...then a long gap: the next rotation starts a fresh streak.
        pool._unmatched_rotation_streak_last_ts -= (
            pool._UNMATCHED_ROTATION_WINDOW_SECONDS + 1
        )
        pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="no-match",
        )
        assert pool._unmatched_rotation_streak == 1
