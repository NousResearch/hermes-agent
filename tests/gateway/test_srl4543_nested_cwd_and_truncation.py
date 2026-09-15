"""SRL-4543 upstream review (andrexibiza, PR #110487) — two blockers found in the nested
(same-task set/clear-pair) case that the top-level-only tests couldn't see:

1. Nested session restoration dropped the outer runtime cwd: clear_session_vars() called
   clear_session_cwd() unconditionally instead of restoring the outer scope's cwd via its
   token, so after `outer cwd -> inner bind -> clear inner`, the outer turn's cwd became ""
   instead of its real value.
2. The malformed-token fail-closed guarantee was false for NESTED tokens: validation ran
   AFTER applying every supplied token, so a truncated token vector from a nested bind (whose
   tokens carry real old_value — not Token.MISSING/_UNSET) restored real outer identity
   values before the ValueError was ever raised.
"""
from agent.runtime_cwd import resolve_context_cwd
from gateway.session_context import clear_session_vars, get_session_env, set_session_vars


def test_nested_clear_restores_outer_cwd_not_empty_string(tmp_path):
    """outer cwd -> inner bind -> clear inner -> outer cwd must survive intact."""
    outer_dir = tmp_path / "outer"
    inner_dir = tmp_path / "inner"
    outer_dir.mkdir()
    inner_dir.mkdir()

    outer_tokens = set_session_vars(session_key="outer-key", user_id="outer@example.invalid", cwd=str(outer_dir))
    try:
        assert resolve_context_cwd() == outer_dir

        inner_tokens = set_session_vars(session_key="outer-key", user_id="outer@example.invalid", cwd=str(inner_dir))
        clear_session_vars(inner_tokens)

        # BUG (pre-fix): this was "" (clear_session_cwd() stomped it), not the outer cwd —
        # a later cwd consumer would then fall through to TERMINAL_CWD/launch dir and could
        # operate in the wrong workspace while the outer turn believed it was still admitted.
        assert resolve_context_cwd() == outer_dir
    finally:
        clear_session_vars(outer_tokens)


def test_nested_truncated_tokens_never_restore_real_outer_identity_before_raising():
    """Reproduces the exact nested Alice/Bob scenario from the review: a truncated tokens
    vector from a NESTED bind must reset everything to baseline and raise WITHOUT ever
    restoring the outer turn's real user_id/session_key/browser principal along the way."""
    outer_tokens = set_session_vars(
        session_key="alice-key", user_id="alice@example.invalid",
        browser_control_principal="alice-digest", browser_control_transport_family="cloud-ticket-ws")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == "alice@example.invalid"

        # Nested bind on top of Alice's outer scope — these tokens' old_value is Alice's real
        # values, not Token.MISSING/_UNSET, which is exactly what made the pre-fix bug latent.
        inner_tokens = set_session_vars(
            session_key="bob-key", user_id="bob@example.invalid",
            browser_control_principal="bob-digest", browser_control_transport_family="cloud-ticket-ws")

        truncated_inner = inner_tokens[:5]
        try:
            clear_session_vars(truncated_inner)
            raise AssertionError("expected ValueError for malformed nested tokens")
        except ValueError:
            pass

        # BUG (pre-fix): partial application would have restored Alice's real user_id/
        # session_key/browser principal from the tokens that WERE present in truncated_inner,
        # silently granting that stale authority before the exception ever surfaced — and
        # tui_gateway's _clear_session_context swallows cleanup exceptions, so nothing
        # downstream would ever see it. The fix-round contract: baseline EVERYTHING, never a
        # partial real restore, regardless of nesting depth.
        assert get_session_env("HERMES_SESSION_USER_ID") == ""
        assert get_session_env("HERMES_SESSION_KEY") == ""
        assert get_session_env("HERMES_BROWSER_CONTROL_PRINCIPAL") == ""
    finally:
        # outer_tokens' own reset already ran as part of the baseline-everything path above;
        # a second clear is a documented no-op-safe falsy-tokens call, not an outer restore.
        pass
