"""Native chat serializes local startup I/O before entering the CLI."""

from argparse import Namespace
import sys
import threading
import types

import pytest

from hermes_cli import banner, main
from tools import skills_sync, skills_tool


@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("failed_scan", [False, True])
@pytest.mark.parametrize("termux, prefetch_updates", [(False, True), (True, False), (True, True)])
def test_native_chat_syncs_before_warming_banner(
    monkeypatch, tmp_path, seeded, failed_scan, termux, prefetch_updates,
):
    home = tmp_path / "profile"
    bundled = tmp_path / "bundled"
    skill = bundled / "general" / "sample" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: sample\ndescription: A synthetic skill\n---\nExample\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    if termux:
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    else:
        monkeypatch.delenv("PREFIX", raising=False)
    if prefetch_updates:
        monkeypatch.setenv("HERMES_TERMUX_PREFETCH_UPDATES", "1")
    else:
        monkeypatch.delenv("HERMES_TERMUX_PREFETCH_UPDATES", raising=False)
    monkeypatch.setattr(main, "PROJECT_ROOT", bundled.parent)
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: bundled)
    monkeypatch.setattr(skills_sync, "_get_optional_dir", lambda: tmp_path / "optional")
    if seeded:
        skills_sync.sync_skills(quiet=True)

    events = []
    scheduled = []
    sync = main._sync_bundled_skills_for_startup

    def sync_on_main():
        assert threading.current_thread() is threading.main_thread()
        events.append("sync")
        return sync()

    scan = skills_tool._find_all_skills
    attempts = []

    def scan_skills():
        events.append("skills")
        attempts.append(None)
        if failed_scan and len(attempts) == 1:
            raise OSError("transient scan failure")
        return scan()

    def git_state():
        events.append("git")
        return {"upstream": "abc", "local": "abc", "ahead": 0}

    def release_tag(*_args, **_kwargs):
        events.append("release")
        return "v1.0.0"

    class DeferredThread:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name")

        def start(self):
            scheduled.append(self.name)

    monkeypatch.setattr(main.threading, "Thread", DeferredThread)
    monkeypatch.setattr(main, "_sync_bundled_skills_for_startup", sync_on_main)
    monkeypatch.setattr(skills_tool, "_find_all_skills", scan_skills)
    monkeypatch.setattr(banner, "_compute_git_banner_state", git_state)
    monkeypatch.setattr(banner, "_resolve_repo_dir", lambda: tmp_path)
    monkeypatch.setattr(banner.source_check, "_git_stdout", release_tag)
    monkeypatch.setattr(banner, "prefetch_update_check", lambda: events.append("update"))
    for cache in ("_available_skills_cache", "_git_banner_state_cache", "_latest_release_cache"):
        monkeypatch.setattr(banner, cache, None)
    monkeypatch.setattr(main, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(main, "_resolve_chat_session_args", lambda *_args: None)
    monkeypatch.setattr(main, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr(main, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.run_bootstrap", lambda **_kw: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_consent.offer_consent_before_chat", lambda _args: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_process.begin_process", lambda *_args: None)
    monkeypatch.setitem(sys.modules, "cli", types.SimpleNamespace(main=lambda **_kw: events.append("cli")))

    main.cmd_chat(Namespace(command="chat", model=None, toolsets=None, query=None))

    assert scheduled == [], "Local scans must not overlap the foreground CLI imports"
    assert events == ["sync", "skills", "git", "release", *(["update"] if prefetch_updates else []), "cli"]
    assert banner.get_available_skills() == {"general": ["sample"]}
    assert len(attempts) == (2 if failed_scan else 1), "Failed scans remain retryable"
    assert banner.get_git_banner_state() == {"upstream": "abc", "local": "abc", "ahead": 0}
    assert banner.get_latest_release_tag()[0] == "v1.0.0"
