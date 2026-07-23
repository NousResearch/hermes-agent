"""Parity guard: every hook in VALID_HOOKS has a synthetic test payload.

``hermes hooks test`` / ``hermes hooks doctor`` promise that the stdin a
script sees is identical in shape to a real runtime firing.  That promise
only holds if ``_DEFAULT_PAYLOADS`` covers every event in ``VALID_HOOKS`` —
uncovered events silently fall back to a generic ``{"session_id": ...}``
payload that looks nothing like production.  This test pins the two sets
together so adding a hook without a synthetic payload fails CI.
"""

from __future__ import annotations

from hermes_cli.hooks import _DEFAULT_PAYLOADS
from hermes_cli.plugins import VALID_HOOKS


def test_default_payloads_cover_all_valid_hooks():
    missing = VALID_HOOKS - set(_DEFAULT_PAYLOADS)
    stale = set(_DEFAULT_PAYLOADS) - VALID_HOOKS
    assert not missing, (
        f"Hooks without a synthetic test payload (add them to "
        f"hermes_cli.hooks._DEFAULT_PAYLOADS, mirroring the real "
        f"invoke_hook() call site): {sorted(missing)}"
    )
    assert not stale, (
        f"_DEFAULT_PAYLOADS entries for events not in VALID_HOOKS "
        f"(remove or rename them): {sorted(stale)}"
    )


def test_default_payloads_are_dicts():
    for event, payload in _DEFAULT_PAYLOADS.items():
        assert isinstance(payload, dict), (
            f"_DEFAULT_PAYLOADS[{event!r}] must be a kwargs dict, "
            f"got {type(payload).__name__}"
        )
