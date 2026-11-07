"""Home initialization must survive a SOUL.md symlink the seed cannot write through.

A cyclic ``SOUL.md -> SOUL.md`` link (ELOOP) or a link dangling into a missing directory
(ENOENT) made the first-run seed raise OSError, which ``initialize_home`` turned into
``HomeInitializationError`` on every gateway spawn (launchd exit-75 relaunch storm). The seed
now replaces such a link with the default identity file; a link that resolves is operator
wiring and stays a link.
"""

import os
import stat

import pytest

from hermes_cli.config import DEFAULT_SOUL_MD, _ensure_default_soul_md
from hermes_cli.config_home import initialize_home

_SUBDIRS = ("cron", "sessions", "logs", "memories")


@pytest.mark.parametrize("target", ["SOUL.md", "missing-dir/SOUL.md"], ids=["cyclic", "dangling"])
def test_initialize_home_replaces_unwritable_soul_symlink(tmp_path, target):
    home = tmp_path / ".hermes"
    home.mkdir()
    soul = home / "SOUL.md"
    soul.symlink_to(home / target)

    initialize_home(home, _SUBDIRS, set())

    assert not soul.is_symlink()
    assert soul.read_text(encoding="utf-8") == DEFAULT_SOUL_MD
    assert not [p for p in home.iterdir() if p.name.startswith(".SOUL.md.")]


def test_soul_symlink_to_customized_file_is_left_alone(tmp_path):
    """Control: a resolving link survives and its content is untouched."""
    home = tmp_path / ".hermes"
    home.mkdir()
    target = tmp_path / "shared-identity.md"
    target.write_text("custom identity\n", encoding="utf-8")
    soul = home / "SOUL.md"
    soul.symlink_to(target)

    _ensure_default_soul_md(home)
    initialize_home(home, _SUBDIRS, set())

    assert soul.is_symlink()
    assert soul.read_text(encoding="utf-8") == "custom identity\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through 0444")
def test_initialize_home_keeps_read_only_legacy_scaffold(tmp_path):
    """A read-only legacy scaffold (0444 from early hosted provisioning) cannot be upgraded.

    The upgrade is optional, so home init must leave it in place instead of failing every
    gateway start (v0.23.0 Webapp listener crash loop).
    """
    from hermes_cli.default_soul import _LEGACY_TEMPLATE_SOULS

    home = tmp_path / ".hermes"
    home.mkdir()
    soul = home / "SOUL.md"
    soul.write_text(_LEGACY_TEMPLATE_SOULS[0], encoding="utf-8")
    soul.chmod(0o444)

    initialize_home(home, _SUBDIRS, set())

    assert soul.read_text(encoding="utf-8") == _LEGACY_TEMPLATE_SOULS[0]
    assert stat.S_IMODE(soul.stat().st_mode) == 0o444
