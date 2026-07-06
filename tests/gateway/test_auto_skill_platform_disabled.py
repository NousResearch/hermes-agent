"""
Regression test for PR #59478 fix: auto-skill channel bindings must respect
platform-disabled skill gates.

The `_handle_message_with_agent()` auto-skill block (Telegram DM Topics,
Discord `channel_skill_bindings`) loads bound skills via
`_load_skill_payload()` with a raw identifier, bypassing
`get_skill_commands()`'s scan-time disabled filter. Result: a skill an
operator disables for a platform (or globally via `skills.disabled`) still
gets its full content injected into every new session bound to that
channel/topic.

This test mirrors the approach used in
`test_10710_auto_reset_evicts_cached_agent.py` — `_handle_message_with_agent`
requires a large mocked harness to invoke directly, so we assert the fix
indirectly by verifying the source code of the relevant block in
`gateway/run.py` contains the expected disabled-skill check.
"""

import pathlib


def test_auto_skill_block_checks_platform_disabled_gate():
    """
    Verify that the auto-skill loading block in
    `_handle_message_with_agent()` checks `get_disabled_skill_names(platform=...)`
    before calling `_load_skill_payload()` for each auto-skill.
    """
    run_py = pathlib.Path(__file__).parents[2] / "gateway" / "run.py"
    source = run_py.read_text(encoding="utf-8")

    # Verify the key elements are present in the auto-skill block
    # 1. Import of get_disabled_skill_names (with alias _get_plat_disabled)
    assert "from agent.skill_utils import get_disabled_skill_names as _get_plat_disabled" in source, (
        "Missing import of get_disabled_skill_names as _get_plat_disabled"
    )
    # 2. Call to get_disabled_skill_names with platform argument (via alias)
    assert "_get_plat_disabled(platform=" in source or "_get_plat_disabled(platform =" in source, (
        "Missing call to _get_plat_disabled(platform=...)"
    )
    # 3. Check for the skip condition
    assert "_sname in _plat_disabled" in source, (
        "Missing check to skip disabled skills (_sname in _plat_disabled)"
    )
    # 4. Continue statement to skip disabled skill
    assert "continue" in source and "Skipping disabled auto-skill" in source, (
        "Missing continue/logic for skipping disabled skills"
    )


def test_auto_skill_block_logs_skipped_disabled_skills():
    """
    Verify that the auto-skill block logs when skipping a disabled skill.
    """
    run_py = pathlib.Path(__file__).parents[2] / "gateway" / "run.py"
    source = run_py.read_text(encoding="utf-8")

    # Search for the log message pattern
    assert "Skipping disabled auto-skill" in source, (
        "Expected log message for skipped disabled auto-skill not found"
    )


if __name__ == "__main__":
    test_auto_skill_block_checks_platform_disabled_gate()
    test_auto_skill_block_logs_skipped_disabled_skills()
    print("All tests passed!")