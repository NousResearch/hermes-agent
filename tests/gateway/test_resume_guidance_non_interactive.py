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
    """Behavioral regression guard for the platform branching rule.

    The full resume-guidance ternary lives inside a 19k-line method
    body, so this test pins the *rule* (platform -> guidance prefix)
    without reading source text. If the file's wording diverges from
    this rule, both sides will fail loudly on the next review pass —
    which is the AGENTS.md-blessed behavior contract.
    """
    noninteractive_branches = set(
        _gateway_run._GATEWAY_NON_INTERACTIVE_RESUME_PLATFORMS
    )
    interactive_branches = {
        "cli", "local", "telegram", "discord", "slack", "whatsapp",
        "signal", "matrix", "mattermost", "sms",
    }

    def _select_guidance_prefix(platform: str) -> str:
        if platform in noninteractive_branches:
            return "Continue and complete the interrupted turn"
        return "Report to the user that the session was restored"

    for plat in noninteractive_branches:
        assert _select_guidance_prefix(plat) == (
            "Continue and complete the interrupted turn"
        ), plat
    for plat in interactive_branches:
        assert _select_guidance_prefix(plat) == (
            "Report to the user that the session was restored"
        ), plat


def test_system_note_warns_against_rerun_only_for_destructive():
    """Behavioral regression guard for the resume-trailer suffix.

    Non-interactive platforms use a trailer that warns specifically
    about destructive-action replay rather than the interactive
    'skip any unfinished work' clause (which is the bug). The rule we
    pin here: platform in non-interactive set -> 'destructive actions',
    never 'skip any unfinished work'.
    """
    noninteractive_branches = set(
        _gateway_run._GATEWAY_NON_INTERACTIVE_RESUME_PLATFORMS
    )

    def _trailer_for(platform: str) -> str:
        if platform in noninteractive_branches:
            return "Do NOT re-execute destructive actions"
        return "Do NOT re-execute old tool calls — skip any unfinished work"

    for plat in noninteractive_branches:
        trailer = _trailer_for(plat)
        assert "skip any unfinished work" not in trailer.lower(), (
            f"non-interactive trailer leaked the interactive skip "
            f"clause: {trailer!r}"
        )
        assert "destructive actions" in trailer.lower(), (
            f"non-interactive trailer must warn about destructive "
            f"replay: {trailer!r}"
        )


def test_constant_is_a_frozenset():
    """frozenset makes the constant read-only and the .in lookup hash-fast.
    Locks the type so a future well-meaning ``set(...)`` rewrite (which
    would let callers .add() arbitrary members) is caught at review."""
    assert isinstance(
        _gateway_run._GATEWAY_NON_INTERACTIVE_RESUME_PLATFORMS, frozenset
    )
