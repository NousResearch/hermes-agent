"""Measure the exact-value scrub cost on a 100 KB tool result.

Run: HERMES_HOME=<isolated> python scripts/bench_env_redaction.py  (from the repo root)
Prints the per-pass time for 0 / 16 / 64 / 128 registered values, with and without a hit.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="hermes-bench-home-"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.redact import (
    clear_env_redaction_values, redact_registered_vault_values, redact_sensitive_text,
    register_env_redaction_values,
)

TEXT = ("line of ordinary tool output with paths /usr/local/bin and ids 1234567890\n" * 1400)[:100_000]


def _best_of(fn, n=20) -> float:
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    print(f"text: {len(TEXT)} chars")
    base = _best_of(lambda: redact_sensitive_text(TEXT, force=True, code_file=True))
    print(f"redact_sensitive_text alone (no registry): {base * 1000:.2f} ms")
    for count in (0, 16, 64, 128):
        clear_env_redaction_values()
        if count:
            register_env_redaction_values({f"TOKEN_{i:04d}": f"value-{i:04d}-abcdefghijklmnop" for i in range(count)})
        miss = _best_of(lambda: redact_registered_vault_values(TEXT))
        hit_text = TEXT[:50_000] + "value-0007-abcdefghijklmnop" + TEXT[50_000:]
        hit = _best_of(lambda: redact_registered_vault_values(hit_text))
        print(f"{count:4d} values: no hit {miss * 1000:.3f} ms   one hit {hit * 1000:.3f} ms")


if __name__ == "__main__":
    main()
