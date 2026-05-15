import json
import time
from pathlib import Path

from hermes_cli import banner


def test_update_check_cache_invalidates_when_local_head_changes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    install = tmp_path / "install" / "hermes-agent"
    repo = install
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()
    cache_file = home / ".update_check"
    cache_file.write_text(json.dumps({"ts": time.time(), "behind": 9, "rev": "old-head"}))

    calls = []

    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr(banner, "get_hermes_home", lambda: home)
    monkeypatch.setattr(banner, "__file__", str(install / "hermes_cli" / "banner.py"))
    monkeypatch.setattr(banner, "_local_git_rev", lambda repo_dir: "new-head")

    def fake_check(repo_dir: Path):
        calls.append(repo_dir)
        return 0

    monkeypatch.setattr(banner, "_check_via_local_git", fake_check)

    assert banner.check_for_updates() == 0
    assert calls == [repo]
    cached = json.loads(cache_file.read_text())
    assert cached["behind"] == 0
    assert cached["rev"] == "new-head"


def test_update_check_uses_cache_when_local_head_matches(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    install = tmp_path / "install" / "hermes-agent"
    repo = install
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()
    cache_file = home / ".update_check"
    cache_file.write_text(json.dumps({"ts": time.time(), "behind": 3, "rev": "same-head"}))

    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr(banner, "get_hermes_home", lambda: home)
    monkeypatch.setattr(banner, "__file__", str(install / "hermes_cli" / "banner.py"))
    monkeypatch.setattr(banner, "_local_git_rev", lambda repo_dir: "same-head")

    def fail_if_called(repo_dir: Path):
        raise AssertionError("fresh update check should not run on cache hit")

    monkeypatch.setattr(banner, "_check_via_local_git", fail_if_called)

    assert banner.check_for_updates() == 3


def test_update_check_seeds_local_head_rev_when_cache_missing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    install = tmp_path / "install" / "hermes-agent"
    repo = install
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()
    cache_file = home / ".update_check"

    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr(banner, "get_hermes_home", lambda: home)
    monkeypatch.setattr(banner, "__file__", str(install / "hermes_cli" / "banner.py"))
    monkeypatch.setattr(banner, "_local_git_rev", lambda repo_dir: "fresh-head")
    monkeypatch.setattr(banner, "_check_via_local_git", lambda repo_dir: 0)

    assert banner.check_for_updates() == 0

    cached = json.loads(cache_file.read_text())
    assert cached["behind"] == 0
    assert cached["rev"] == "fresh-head"
