"""Regression: draft ids must be unique across gateway incarnations
(PR 85796 review, B3; hardened per r2 follow-up).

The relay connector tombstones sealed streams by (channel, draft_id) and
those tombstones outlive the gateway process (relay gateways are
disposable / scale-to-zero by design). A counter that restarts at zero
replays already-sealed wire identities: the connector answers the new
turn's frames out of the old tombstone — zero Slack API calls, the OLD
message ts returned as the new turn's identity — and the new answer is
silently dropped while every layer above records success.

r2 hardening: the first fix seeded from epoch milliseconds, which still
collides on same-millisecond starts, forks, and clock steps. The seed is
now a random 49-bit process nonce — collision probability is negligible
and ids stay inside the connector's JS number range (2^53) with room for
realistic per-process turn counts.
"""

import subprocess
import sys

from gateway.stream_consumer import GatewayStreamConsumer

_JS_SAFE_MAX = 2**53


class TestDraftIdRestartUniqueness:
    def test_seed_is_random_nonce_in_wire_range(self):
        seed = GatewayStreamConsumer._draft_id_counter
        assert seed > 0, "seed 0 replays wire identities after restart"
        # Leave headroom for per-process increments under the JS bound
        # the connector's draft_id?: number field implies.
        assert seed < 2**49
        assert seed + 10_000_000 < _JS_SAFE_MAX

    def test_two_incarnations_use_different_seeds(self):
        """Spawn two real fresh interpreters — the class-level seed must
        differ across process starts (this is exactly the scale-to-zero
        restart). Also implicitly proves the seed isn't wall-clock-locked:
        both processes start within the same second."""
        code = (
            "from gateway.stream_consumer import GatewayStreamConsumer;"
            "print(GatewayStreamConsumer._draft_id_counter)"
        )
        seeds = set()
        for _ in range(2):
            out = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, check=True,
            )
            seeds.add(int(out.stdout.strip()))
        assert len(seeds) == 2, (
            f"two fresh incarnations minted the same draft-id seed: {seeds} — "
            "restart would replay identities the connector already sealed"
        )
