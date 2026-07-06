"""Managed-runtime SOUL.md seeding behavior (hermes_cli.config._ensure_default_soul_md).

Regression tests for the v2026.7.1 stage-deploy failure: upstream's
"upgrade legacy SOUL.md in place" logic crashed managed agent creation with
PermissionError because managed state dirs carry a read-only SOUL.md and the
managed prompt ignores SOUL.md entirely.
"""

from __future__ import annotations

import os
import stat

import pytest

from hermes_cli.config import DEFAULT_SOUL_MD, _ensure_default_soul_md


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv("KARINAI_MANAGED_RUNTIME", raising=False)
    return tmp_path


def test_seeds_default_soul_when_absent(home):
    _ensure_default_soul_md(home)
    assert (home / "SOUL.md").read_text(encoding="utf-8") == DEFAULT_SOUL_MD


def test_managed_runtime_skips_seeding_entirely(home, monkeypatch):
    monkeypatch.setenv("KARINAI_MANAGED_RUNTIME", "true")
    _ensure_default_soul_md(home)
    assert not (home / "SOUL.md").exists()


def test_managed_runtime_never_touches_existing_readonly_soul(home, monkeypatch):
    monkeypatch.setenv("KARINAI_MANAGED_RUNTIME", "true")
    soul = home / "SOUL.md"
    soul.write_text("# managed product soul\n", encoding="utf-8")
    soul.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    _ensure_default_soul_md(home)  # must not raise
    assert soul.read_text(encoding="utf-8") == "# managed product soul\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permission bits")
def test_unwritable_legacy_soul_is_nonfatal_outside_managed_mode(home):
    """A read-only legacy-template SOUL.md must degrade to a warning, not crash.

    Uses a real legacy scaffold (from _LEGACY_TEMPLATE_SOULS) so the in-place
    upgrade path actually attempts the write; the file is read-only so the
    write raises OSError, which must be swallowed. This is the exact shape of
    the v2026.7.1 stage failure (stage2-hook-seeded read-only docker SOUL.md).
    """
    from hermes_cli.default_soul import _LEGACY_TEMPLATE_SOULS, is_legacy_template_soul

    legacy = _LEGACY_TEMPLATE_SOULS[0]
    assert is_legacy_template_soul(legacy)
    soul = home / "SOUL.md"
    soul.write_text(legacy, encoding="utf-8")
    soul.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    _ensure_default_soul_md(home)  # must not raise
    assert soul.read_text(encoding="utf-8") == legacy
