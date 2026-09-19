"""Passive checks use the same release selection as explicit source updates."""

import json
from unittest.mock import Mock

import pytest

from hermes_cli import banner, source_check
from hermes_cli.source_releases import SourceTarget
from hermes_cli.update_channel import install_id


@pytest.mark.parametrize("channel", ["stable", "canary", "preview-from-r2"])
def test_release_channel_never_compares_main_or_reuses_main_cache(tmp_path, monkeypatch, channel):
    from hermes_constants import get_hermes_home

    root = tmp_path / "source"
    root.mkdir()
    (root / ".git").mkdir()
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: root)
    head = "a" * 40
    target = "b" * 40
    monkeypatch.setattr(source_check, "_git_stdout", lambda args, **kw: head if args == ["rev-parse", "HEAD"] else "https://github.com/example/fork.git")
    (get_hermes_home() / "config.yaml").write_text(json.dumps({
        "update": {"installs": {install_id(root): {"path": str(root), "channel": channel}}}
    }))
    (get_hermes_home() / ".update_check").write_text(json.dumps({
        "rev": None, "ver": banner.VERSION, "head": head, "behind": 99, "ts": 10**12,
    }))
    resolve = Mock(return_value=SourceTarget(channel, channel, "example/fork", commit=target, version="1.2.3"))
    monkeypatch.setattr(source_check, "resolve_source_target", resolve)
    main = Mock(side_effect=AssertionError("release must not check a branch"))
    monkeypatch.setattr(source_check, "_branch_tip", main)
    status = source_check.check_for_updates(passive=True)
    assert status["behind"] == source_check.UPDATE_AVAILABLE_NO_COUNT
    resolve.assert_called_once_with(channel, ["git"], root, repository="example/fork")
    main.assert_not_called()
    assert status["channel"] == channel
    assert status["targetSha"] == target
    # The selected release may be an ancestor (a requested canary → stable switch).
    # That is still a different release, not "already current" or "behind main".
    label = banner.format_banner_version_label()
    assert channel in label
    assert "upstream" not in label
