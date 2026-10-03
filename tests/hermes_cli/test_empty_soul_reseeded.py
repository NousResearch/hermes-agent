"""A zero-byte or whitespace-only SOUL.md must be re-seeded with the default identity.

``_ensure_default_soul_md()`` only overwrites a file it recognises as an
auto-seeded legacy template. A truncated ``SOUL.md`` (failed write, interrupted
sync, cloud-sync conflict) matched no template, so the seed was skipped forever:
the file loaded as an *empty* context source and the profile ran with no persona
at all -- not the default one, not a custom one. ``_normalize_soul`` strips
whitespace, so empty and whitespace-only files are the same case.
"""

import pytest

from hermes_cli.config import DEFAULT_SOUL_MD, _ensure_default_soul_md
from hermes_cli.config_home import initialize_home

_SUBDIRS = ("cron", "sessions", "logs", "memories")


@pytest.mark.parametrize(
    "content",
    ["", "   ", "\n\n", "\r\n \t\r\n"],
    ids=["empty", "spaces", "newlines", "crlf-mixed"],
)
def test_initialize_home_reseeds_empty_soul(tmp_path, content):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "SOUL.md").write_text(content, encoding="utf-8")

    initialize_home(home, _SUBDIRS, set())

    assert (home / "SOUL.md").read_text(encoding="utf-8") == DEFAULT_SOUL_MD


def test_empty_soul_is_recognised_as_legacy():
    """The empty file must be classified as auto-seeded, not user-authored."""
    from hermes_cli.default_soul import is_legacy_template_soul

    assert is_legacy_template_soul("") is True
    assert is_legacy_template_soul("  \n ") is True


def test_real_persona_is_still_never_touched(tmp_path):
    """The safety guarantee must survive: a persona the user typed stays put."""
    home = tmp_path / ".hermes"
    home.mkdir()
    soul = home / "SOUL.md"
    persona = "You are a concise technical expert. No fluff, just facts."
    soul.write_text(persona, encoding="utf-8")

    initialize_home(home, _SUBDIRS, set())

    assert soul.read_text(encoding="utf-8") == persona


def test_seeded_default_soul_is_not_rewritten(tmp_path):
    """Re-running init on a healthy install must be a no-op on SOUL.md."""
    home = tmp_path / ".hermes"
    home.mkdir()
    soul = home / "SOUL.md"
    soul.write_text(DEFAULT_SOUL_MD, encoding="utf-8")

    _ensure_default_soul_md(home)

    assert soul.read_text(encoding="utf-8") == DEFAULT_SOUL_MD
