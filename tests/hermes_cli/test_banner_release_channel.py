"""Passive checks use the same release selection as explicit source updates."""

import json
from unittest.mock import Mock

import pytest

from hermes_cli import banner
from hermes_cli.update_channel import install_id


@pytest.mark.parametrize("channel", ["stable", "canary"])
def test_release_channel_never_compares_main_or_reuses_main_cache(tmp_path, monkeypatch, channel):
    from hermes_constants import get_hermes_home

    root = tmp_path / "source"
    root.mkdir()
    (root / ".git").mkdir()
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: root)
    monkeypatch.setattr(banner, "_resolve_repo_dir", lambda: root)
    head = "a" * 40
    target = "b" * 40
    monkeypatch.setattr(banner, "_git_stdout", lambda args, **kw: head if args == ["rev-parse", "HEAD"] else "https://github.com/NousResearch/hermes-agent.git")
    (get_hermes_home() / "config.yaml").write_text(json.dumps({
        "update": {"installs": {install_id(root): {"path": str(root), "channel": channel}}}
    }))
    (get_hermes_home() / ".update_check").write_text(json.dumps({
        "rev": None, "ver": banner.VERSION, "head": head, "behind": 99, "ts": 10**12,
    }))
    resolve = Mock(return_value=("v1.2.3" if channel == "stable" else "v1.2.4-canary.20260911", target))
    monkeypatch.setattr("hermes_cli.source_releases.resolve_source_release", resolve)
    repository = Mock(return_value="example/fork")
    monkeypatch.setattr("hermes_cli.source_releases.source_repository", repository)
    main = Mock(return_value="c" * 40)
    monkeypatch.setattr(banner, "_github_branch_tip", main)
    compare = Mock(return_value=2)
    monkeypatch.setattr(banner, "_tips_behind", compare)

    assert banner.check_for_updates(passive=True) == banner.UPDATE_AVAILABLE_NO_COUNT
    repository.assert_called_once_with(["git"], root)
    resolve.assert_called_once_with(channel, repository="example/fork")
    compare.assert_not_called()
    main.assert_not_called()
    cache = json.loads((get_hermes_home() / ".update_check").read_text())
    assert cache["channel"] == channel
    assert cache["target"] == target
    # The selected release may be an ancestor (a requested canary → stable switch).
    # That is still a different release, not "already current" or "behind main".
    label = banner.format_banner_version_label()
    assert channel in label
    assert "upstream" not in label
