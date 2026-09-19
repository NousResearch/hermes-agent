"""Behavioral tests for hermes_cli.default_soul — SOUL.md template matching.

Covers the legacy-template detection used when upgrading an old comment-only
SOUL.md (seeded by pre-DEFAULT_SOUL_MD installers) to the default persona,
without touching user-customized content.
"""

import pytest

from hermes_cli.default_soul import (
    DEFAULT_SOUL_MD,
    _LEGACY_TEMPLATE_SOULS,
    is_legacy_template_soul,
)


def test_default_soul_is_nonempty_persona():
    # The default must actually carry a persona, not be empty scaffolding.
    assert DEFAULT_SOUL_MD.strip()
    assert "Hermes Agent" in DEFAULT_SOUL_MD


def test_legacy_template_matches_verbatim():
    # A file whose content equals a known legacy template is safe to upgrade.
    assert is_legacy_template_soul(_LEGACY_TEMPLATE_SOULS[0]) is True
    assert is_legacy_template_soul(_LEGACY_TEMPLATE_SOULS[1]) is True


def test_legacy_template_matches_with_trailing_newline():
    # Installers may leave a trailing newline; normalization must still match.
    assert is_legacy_template_soul(_LEGACY_TEMPLATE_SOULS[0] + "\n") is True


def test_legacy_template_matches_with_crlf():
    # Windows installers may write CRLF line endings; normalization unifies them.
    crlf = _LEGACY_TEMPLATE_SOULS[0].replace("\n", "\r\n")
    assert is_legacy_template_soul(crlf) is True


def test_legacy_template_matches_with_bom():
    # A leading UTF-8 BOM must be stripped before comparison.
    assert is_legacy_template_soul("\ufeff" + _LEGACY_TEMPLATE_SOULS[0]) is True


def test_user_persona_is_not_matched():
    # Any user-authored content (even a single line added) must NOT match —
    # the whole safety guarantee is that legacy templates carry zero user intent.
    persona = _LEGACY_TEMPLATE_SOULS[0] + "\nYou are a sarcastic pirate."
    assert is_legacy_template_soul(persona) is False


def test_default_soul_is_not_legacy_template():
    # The current default persona is not one of the old comment-only scaffolds.
    assert is_legacy_template_soul(DEFAULT_SOUL_MD) is False


def test_empty_and_whitespace_input():
    # Degenerate input — no user persona, but also not a known legacy template.
    assert is_legacy_template_soul("") is False
    assert is_legacy_template_soul("   \n  ") is False
