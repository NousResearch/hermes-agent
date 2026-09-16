"""Upgrade contract for auto-seeded SOUL.md texts across generations.

Covers the migration off the pre-carve-out default: a SOUL.md still matching the
auto-seeded generation whose blanket "no narrating" line silenced long turns on
non-editable platforms upgrades in place; a user-customized file never does.
"""

import os
from unittest.mock import patch

from hermes_cli.config import ensure_hermes_home
from hermes_cli.default_soul import DEFAULT_SOUL_MD, is_legacy_template_soul

# The auto-seeded DEFAULT_SOUL_MD generation this migration retires, hardcoded so the
# fixture keeps testing the old text regardless of any future DEFAULT_SOUL_MD change.
_PRE_CARVEOUT_DEFAULT_SOUL = (
    "You are Hermes Agent, built by Nous Research. Be direct: match the length of your reply to the weight of "
    "the ask \u2014 a one-line question gets a one-line answer, and finished work gets a short report of what "
    "changed, what's verified, and what's left, never a replay of the process. No filler (\"Great question,\" "
    '"I\'d be happy to"), no restating the request back, no re-summarizing what you already said, no narrating '
    "tool calls the user can see. Plain claims over adjectives; when unsure, say so plainly. Agree because it's "
    "right, not because the user said it. Depth is earned \u2014 give it when the user asks for detail, teaches, or "
    "the stakes demand it, not by default."
)


class TestPreCarveoutSoulUpgrade:
    def test_outgoing_generation_is_legacy(self):
        # The retired text (and its install.ps1 ASCII variant) carries zero user intent.
        assert is_legacy_template_soul(_PRE_CARVEOUT_DEFAULT_SOUL)
        assert is_legacy_template_soul(
            _PRE_CARVEOUT_DEFAULT_SOUL.replace("\u2014", "--")
        )

    def test_outgoing_generation_differs_from_current_default(self):
        # The migration must actually change something: the current default draws the
        # decision-point line instead of the blanket prohibition.
        assert _PRE_CARVEOUT_DEFAULT_SOUL != DEFAULT_SOUL_MD
        assert "narrating individual tool calls" not in _PRE_CARVEOUT_DEFAULT_SOUL
        assert "narrating individual tool calls" in DEFAULT_SOUL_MD

    def test_matching_soul_upgrades_in_place(self, tmp_path):
        # An install seeded with the retired default picks up the carve-out wording on
        # the next ensure_hermes_home() pass — no manual SOUL.md edit needed.
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            soul_path = tmp_path / "SOUL.md"
            soul_path.write_text(_PRE_CARVEOUT_DEFAULT_SOUL, encoding="utf-8")
            ensure_hermes_home()
            assert soul_path.read_text(encoding="utf-8") == DEFAULT_SOUL_MD

    def test_ascii_variant_upgrades_in_place(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            soul_path = tmp_path / "SOUL.md"
            soul_path.write_text(
                _PRE_CARVEOUT_DEFAULT_SOUL.replace("\u2014", "--"), encoding="utf-8"
            )
            ensure_hermes_home()
            assert soul_path.read_text(encoding="utf-8") == DEFAULT_SOUL_MD

    def test_user_customized_soul_is_never_touched(self, tmp_path):
        # One user-added character turns the seeded text into real intent.
        customized = (
            _PRE_CARVEOUT_DEFAULT_SOUL + " Also: always answer in rhyming couplets."
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            soul_path = tmp_path / "SOUL.md"
            soul_path.write_text(customized, encoding="utf-8")
            ensure_hermes_home()
            assert soul_path.read_text(encoding="utf-8") == customized
