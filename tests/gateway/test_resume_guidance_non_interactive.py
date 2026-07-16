"""Tests for issue #57056 — non-interactive gateway resume guidance.

When the gateway is restarted mid-turn and ``_schedule_resume_pending_sessions``
re-fires a stub ``MessageEvent`` with empty text on boot, the inner
``_handle_message_with_agent`` synthesizes a System-note that steers the
model. The old single guidance ("report restored, ask what's next, skip
unfinished work") is correct for an interactive platform but wrong for
webhook/api_server/msgraph_webhook: there is no responder, so the short
"session restored" line is useless and the interrupted task is
abandoned instead of completed.

These tests pin both branches' guidance strings against racetrack edits,
without bringing up the full GatewayRunner (which loads dozens of
optional platform adapters).
"""
import importlib


_gateway_run = importlib.import_module("gateway.run")


def test_non_interactive_platform_set_matches_documented_members():
    """Constant must contain exactly the platforms the bug report (and the
    sibling raw-text set) treat as 'no human responder'. Adding a new
    programmatic adapter to _GATEWAY_RAW_TEXT_PLATFORMS without adding it
    here is the exact regression mode this constant guards against."""
    expected = {"api_server", "webhook", "msgraph_webhook"}
    assert _gateway_run._GATEWAY_NON_INTERACTIVE_RESUME_PLATFORMS == expected


def test_constant_does_not_include_interactive_platforms():
    """Sanity guard: an interactive platform accidentally added here would
    silently start a wrong "continue everything" model note on a CLI /
    Telegram session. Pins the absence."""
    forbidden = {"cli", "local", "telegram", "discord", "slack", "whatsapp",
                 "signal", "matrix", "mattermost", "sms"}
    actual = _gateway_run._GATEWAY_NON_INTERACTIVE_RESUME_PLATFORMS
    assert actual.isdisjoint(forbidden), (
        f"non-interactive resume set has an interactive member: "
        f"{actual & forbidden}"
    )


def test_interactive_string_appears_in_branch_instruction_set():
    """The existing interactive guidance must still be reachable: regression
    guard so a future refactor that lifts guidance strings into a module-
    level constant does not silently drop the interactive note."""
    module_src = (_gateway_run.__file__ and
                   open(_gateway_run.__file__, encoding="utf-8").read())
    assert "Report to the user that the session was restored" in module_src
    # Adaptive (non-interactive) guidance marker
    assert "Continue and complete the interrupted turn" in module_src
    # NEW-branch guidance marker
    assert "Address the user's NEW message below FIRST" in module_src


def test_system_note_warns_against_rerun_only_for_destructive():
    """On a non-interactive platform the wording must NOT keep the
    'skip any unfinished work' clause that the interactive case carries —
    it is the exact bug. We anchor the substitution explicitly."""
    import re
    module_src = open(_gateway_run.__file__, encoding="utf-8").read()
    # The non-interactive guidance lives in the elif-branch we added.
    noninter_block = re.search(
        r"elif _platform_name in _GATEWAY_NON_INTERACTIVE_RESUME_PLATFORMS:(.*?)\n                else:",
        module_src, flags=re.DOTALL,
    )
    assert noninter_block is not None, "non-interactive guidance branch missing"
    blob = noninter_block.group(1)
    assert "Continue and complete the interrupted turn" in blob
    # The 'skip any unfinished work' clause MUST not appear inside this
    # blob — the docstring carries it elsewhere, so use a tighter anchor.
    assert "skip any unfinished work" not in blob.lower()


def test_constant_is_a_frozenset():
    """frozenset makes the constant read-only and the .in lookup hash-fast.
    Locks the type so a future well-meaning ``set(...)`` rewrite (which
    would let callers .add() arbitrary members) is caught at review."""
    assert isinstance(
        _gateway_run._GATEWAY_NON_INTERACTIVE_RESUME_PLATFORMS, frozenset
    )
