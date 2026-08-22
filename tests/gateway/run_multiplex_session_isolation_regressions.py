#!/usr/bin/env python3
"""Dependency-light regressions for multiplex session profile isolation.

The defect: with ``multiplex_profiles`` on,
``SessionStore._recovered_row_allowed_for_active_profile`` short-circuited to
True for ANY cross-profile revival, letting the routing self-heal bind one
profile's key to another profile's live session (agent B then answers with
agent A's frozen persona and history in a shared channel).

Run: python tests/gateway/run_multiplex_session_isolation_regressions.py \
        --source-root <hermes tree>
Exit 0 = all pass; exit 1 = failures (expected on the vanilla pin).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.source_root).expanduser()))

    from gateway.session import SessionStore

    def store(multiplex: bool) -> SessionStore:
        s = SessionStore.__new__(SessionStore)
        s.config = SimpleNamespace(multiplex_profiles=multiplex)
        return s

    def allowed(s, requested, recovered_key):
        return s._recovered_row_allowed_for_active_profile(
            requested_session_key=requested,
            recovered={"session_key": recovered_key},
        )

    BUILDER = "agent:adrolab-builder:buzz:group:chan1:user1"
    QA = "agent:adrolab-qa:buzz:group:chan1:user1"
    QA_DM = "agent:adrolab-qa:buzz:dm:chan2:user1"
    LEGACY = "agent:main:buzz:group:chan1:user1"

    results = []

    def check(name, got, want):
        ok = got == want
        results.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  (got {got}, want {want})")

    mux = store(True)
    # The defect under test: cross-profile revival must be DENIED under multiplex.
    check("mux_cross_profile_denied", allowed(mux, QA, BUILDER), False)
    # Same profile, different chat: allowed (the self-heal's legitimate case).
    check("mux_same_profile_allowed", allowed(mux, QA, QA_DM), True)
    # Same key: allowed.
    check("mux_same_key_allowed", allowed(mux, QA, QA), True)
    # Legacy/main row for a profile-namespaced request: denied under multiplex
    # (the default profile owns agent:main).
    check("mux_legacy_row_for_profile_denied", allowed(mux, QA, LEGACY), False)
    # Unparseable recovered key: permissive legacy behavior retained.
    check("mux_unparseable_recovered_allowed", allowed(mux, QA, "weird-key"), True)
    # Missing recovered key: allowed (nothing to protect).
    check("mux_empty_recovered_allowed", allowed(mux, QA, ""), True)

    # Non-multiplex behavior unchanged: compares against the active profile name.
    plain = store(False)
    active = SessionStore._active_profile_name()
    same_active = f"agent:{active}:buzz:group:chan1:user1"
    check("plain_same_active_allowed", allowed(plain, QA, same_active), True)
    other = "agent:someone-else:buzz:group:chan1:user1"
    check("plain_other_profile_denied", allowed(plain, QA, other), False)

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)} PASS / {len(failed)} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
