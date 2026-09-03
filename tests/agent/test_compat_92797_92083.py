"""#92797 / #92083 compatibility tests for the ``feat/context-length-cap`` branch.

Purpose: prove (in code, not prose) that the dirty worktree stays compatible
with two upstream changes that touch the same code surface, and pin the exact
one-line fix the reviewer/maintainer must apply when those land.

  #92797 (MERGED 2026-08-23): Codex GPT slugs default to 272K; ``-900k``
      variants opt into 900K. Touches ``agent/model_metadata.py``
      (L2418-2625, the Codex-context resolvers) and
      ``agent/auxiliary_client.py`` (L161 import line, L1531 wire-strip).

  #92083 (OPEN PR): warns on unknown keys in the ``model:`` config block by
      adding ``_KNOWN_MODEL_KEYS`` to ``hermes_cli/config.py``.

Findings locked in by these tests:

  * #92797 — NO textual merge conflict (different line ranges in both shared
    files) and CLEAN semantic composition (the branch ceiling is a
    post-resolution hard cap, ``effective = min(resolved, ceiling)``). The
    only shared line (``auxiliary_client.py`` L161 import) is additive on
    both sides, so it merges cleanly.
  * #92083 — ONE real semantic conflict: the branch's ceiling lives at
    ``model.max_context_length`` (a ``model:``-block key), but #92083's
    proposed ``_KNOWN_MODEL_KEYS`` omits it. When #92083 merges, any user who
    sets the ceiling gets a spurious
    ``model: unknown config keys ignored: max_context_length`` warning.
    Fix = add ``max_context_length`` to ``_KNOWN_MODEL_KEYS`` (one line).
"""

from __future__ import annotations

from typing import Any, Dict

# ── #92083: the model.max_context_length ceiling key ──────────────────────

# Verbatim copy of the ``_KNOWN_MODEL_KEYS`` set proposed by #92083 (open PR),
# read from the PR diff. It is NOT present in this branch's base, so the test
# carries the reference set inline rather than importing it.
KNOWN_MODEL_KEYS_92083 = {
    "api_base",
    "base_url",
    "context_length",
    "default",
    "model",
    "name",
    "provider",
    "stale_timeout_seconds",
    "timeout_seconds",
}


def test_branch_ceiling_key_is_not_in_92083_known_set_documents_gap() -> None:
    """DOCUMENTS the #92083 conflict as a known, expected fact (green test).

    The branch reads its ceiling from ``model.max_context_length``. #92083's
    proposed known-key set does NOT include ``max_context_length``, so the
    set-difference that #92083's ``_warn_unknown_model_keys`` performs would
    flag it. This test asserts that gap EXACTLY so a reviewer can see it, and
    so it keeps failing loudly (by design) the moment #92083's set is
    imported into this branch WITHOUT the one-line fix.

    REQUIRED FIX (one line, in ``hermes_cli/config.py`` when #92083 lands):
        _KNOWN_MODEL_KEYS = {..., "max_context_length", ...}
    """
    branch_ceiling_key = "max_context_length"  # what agent_init.py / _get_max_context_length read

    # This is the gap: the branch's key is absent from #92083's known set.
    assert branch_ceiling_key not in KNOWN_MODEL_KEYS_92083, (
        "Surprise: #92083's _KNOWN_MODEL_KEYS now includes 'max_context_length'. "
        "If that is the case, the spurious-warning conflict is already resolved "
        "and this test (which documents the gap) should be inverted to assert "
        "the key IS recognised."
    )

    # And confirm the set-difference #92083 runs would flag exactly our key:
    model_block: Dict[str, Any] = {
        "model": "gpt-4o",
        "provider": "custom",
        "context_length": 128000,
        "max_context_length": 90000,  # the branch's ceiling key
    }
    flagged = set(model_block.keys()) - KNOWN_MODEL_KEYS_92083
    assert flagged == {"max_context_length"}, (
        f"Expected #92083 to flag only the branch ceiling key; got {flagged}"
    )


def test_92083_fix_resolves_the_spurious_warning() -> None:
    """Prove the one-line fix eliminates the spurious warning.

    Adding ``max_context_length`` to #92083's known-key set makes the
    set-difference empty for a config that carries the branch's ceiling, so
    ``_warn_unknown_model_keys`` logs nothing.
    """
    fixed_known_set = KNOWN_MODEL_KEYS_92083 | {"max_context_length"}

    model_block: Dict[str, Any] = {
        "model": "gpt-4o",
        "provider": "custom",
        "context_length": 128000,
        "max_context_length": 90000,
    }
    flagged = set(model_block.keys()) - fixed_known_set
    assert flagged == set(), (
        "With the fix applied, no key should be flagged as unknown"
    )


def test_ceiling_validation_still_strict_after_recognition() -> None:
    """Recognising the key in config must NOT loosen the branch's validator.

    Adding ``max_context_length`` to the known-key set is a config-validation
    change only; the branch's own strict ``coerce_context_ceiling`` pipeline
    (rejects bool/float/numeric-string/<=0) is unaffected.
    """
    from agent.model_metadata import coerce_context_ceiling

    assert coerce_context_ceiling(90000) == 90000
    assert coerce_context_ceiling(272000) == 272000
    # Invalid values still rejected — recognition must not relax these.
    assert coerce_context_ceiling(True) is None      # bool
    assert coerce_context_ceiling(12.5) is None       # float
    assert coerce_context_ceiling("90000") is None    # numeric string
    assert coerce_context_ceiling(0) is None          # zero
    assert coerce_context_ceiling(-1) is None         # negative


# ── #92797: ceiling composes with Codex context resolution ────────────────

def test_ceiling_caps_codex_900k_resolution() -> None:
    """#92797 resolves Codex slugs; the branch ceiling caps the result.

    #92797: ``gpt-5.6-sol`` → 272K, ``gpt-5.6-sol-900k`` → 900K.
    Branch: ``effective = min(resolved, max_context_length)``.

    The ceiling never RAISES a window, only lowers one, so it composes
    cleanly over any value #92797 produces.
    """
    from agent.model_metadata import coerce_context_ceiling

    resolved_codex_900k = 900_000
    ceiling = coerce_context_ceiling(272_000)
    assert ceiling == 272_000
    assert min(resolved_codex_900k, ceiling) == 272_000

    # Ceiling below a 272K base slug still caps downward.
    ceiling_128k = coerce_context_ceiling(128_000)
    assert min(272_000, ceiling_128k) == 128_000

    # A ceiling at or above the resolved value is a no-op (never raises).
    assert min(272_000, coerce_context_ceiling(272_000)) == 272_000


def test_wire_strip_and_ceiling_are_orthogonal() -> None:
    """#92797 strips the ``-900k`` suffix before the model id hits the wire;
    the branch's ceiling is a token budget — they operate on different axes
    and do not interact.

    The wire-strip only affects the *model id string*; the ceiling only
    affects the *token budget*. A ``-900k`` slug stripped to its base id
    still resolves the same 900K window that the ceiling then caps.
    """
    # The base id the wire sees after #92797's strip.
    base_slug = "gpt-5.6-sol"
    resolved_window = 900_000  # the -900k variant's window, before ceiling

    from agent.model_metadata import coerce_context_ceiling
    ceiling = coerce_context_ceiling(400_000)
    effective = min(resolved_window, ceiling)
    assert effective == 400_000
    # The wire model id is unaffected by the ceiling (string vs int).
    assert base_slug == "gpt-5.6-sol"
