"""Regression: symlink-farm overlay HERMES_HOME resolves its profile (#93862).

Multi-agent orchestrators isolate sessions/logs in a per-task directory while
sharing a named profile's content via symlinks (SOUL.md, plugins/, cron/ ->
``<root>/profiles/<name>/...``). Such a home is neither the root nor
``<root>/profiles/<name>``, so both resolvers reported "default".
"""

from __future__ import annotations

import os
from pathlib import Path


def _make_farm(tmp_path: Path, monkeypatch, name: str = "evaluator"):
    """Build native root + named profile + overlay farm; bind HERMES_HOME."""
    home_parent = tmp_path / "home"
    native = home_parent / ".hermes"
    profile = native / "profiles" / name
    (profile / "plugins").mkdir(parents=True)
    (profile / "cron").mkdir(parents=True)
    (profile / "SOUL.md").write_text("# soul\n", encoding="utf-8")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    for member in ("SOUL.md", "plugins", "cron"):
        (overlay / member).symlink_to(
            profile / member, target_is_directory=(member != "SOUL.md")
        )
    monkeypatch.setattr(Path, "home", lambda: home_parent)
    monkeypatch.setenv("HERMES_HOME", str(overlay))
    # hermes_constants memoizes the root on (native, env) — distinct tmp
    # paths per test keep the memo key fresh without manual resets.
    return native, profile, overlay


def test_profile_name_for_home_overlay(tmp_path, monkeypatch):
    from agent import system_prompt

    _, _, overlay = _make_farm(tmp_path, monkeypatch)
    assert system_prompt._profile_name_for_home(overlay) == "evaluator"


def test_resolve_active_profile_name_overlay(tmp_path, monkeypatch):
    from agent.file_safety import _resolve_active_profile_name

    _make_farm(tmp_path, monkeypatch)
    assert _resolve_active_profile_name() == "evaluator"


def test_overlay_mixed_farm_falls_back_to_default(tmp_path, monkeypatch):
    from agent import system_prompt
    from agent.file_safety import (
        _resolve_active_profile_name,
        _profile_name_from_overlay_links,
    )
    from hermes_constants import get_default_hermes_root

    native, _, overlay = _make_farm(tmp_path, monkeypatch, name="one")
    other = native / "profiles" / "two"
    (other / "plugins").mkdir(parents=True)
    (overlay / "plugins").unlink()
    (overlay / "plugins").symlink_to(other / "plugins", target_is_directory=True)
    root = get_default_hermes_root()
    assert _profile_name_from_overlay_links(overlay, root) is None
    assert system_prompt._profile_name_for_home(overlay) == "default"
    assert _resolve_active_profile_name() == "default"


def test_plain_dir_still_default(tmp_path, monkeypatch):
    from agent import system_prompt
    from agent.file_safety import _resolve_active_profile_name

    home_parent = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home_parent)
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(plain))
    assert system_prompt._profile_name_for_home(plain) == "default"
    assert _resolve_active_profile_name() == "default"


def test_real_profile_dir_unchanged(tmp_path, monkeypatch):
    from agent import system_prompt
    from agent.file_safety import _resolve_active_profile_name

    native, profile, _ = _make_farm(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    assert system_prompt._profile_name_for_home(profile) == "evaluator"
    assert _resolve_active_profile_name() == "evaluator"
    assert os.environ["HERMES_HOME"] == str(profile)
