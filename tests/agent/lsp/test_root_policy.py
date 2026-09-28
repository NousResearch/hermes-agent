"""Per-root policy knobs of :class:`LSPService` (#116446): ``lsp.exclude_roots`` gates a workspace off
without disabling the server fleet-wide, and ``lsp.warmup_timeout`` gives a cold root one long wait
instead of the steady-state budget.  Both are driven through ``create_from_config`` and the production
pre-write / post-write entry points, never through the helpers.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.lsp import eventlog
from agent.lsp.manager import LSPService, _client_key
from agent.lsp.servers import find_server_for_file
from agent.lsp.workspace import clear_cache


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_cache()
    eventlog.reset_announce_caches()
    yield
    clear_cache()


def _git_repo(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    (repo / "x.py").write_text("", encoding="utf-8")
    return repo


def _service_from(lsp_cfg: dict) -> LSPService:
    with patch("hermes_cli.config.load_config_readonly", return_value={"lsp": lsp_cfg}):
        svc = LSPService.create_from_config()
    assert svc is not None
    return svc


def test_exclude_roots_skips_matching_root_only(tmp_path, monkeypatch, caplog):
    """A root matching ``lsp.exclude_roots`` never reaches the server — the pre-write snapshot does not
    spawn or wait — while a sibling workspace keeps its diagnostics.  A non-list value fails closed
    (every root skipped) with a WARNING naming the expected shape instead of silently excluding nothing."""
    big = _git_repo(tmp_path, "big")
    small = _git_repo(tmp_path, "small")
    monkeypatch.chdir(str(tmp_path))
    svc = _service_from({"wait_timeout": 1.0, "install_strategy": "manual",
                         "exclude_roots": [str(tmp_path / "bi*")]})
    try:
        async def must_not_run(*_a, **_k):
            raise AssertionError("excluded root reached the server")

        with patch.object(svc, "_snapshot_async", must_not_run), caplog.at_level("INFO", logger=eventlog.event_log.name):
            svc.snapshot_baseline(str(big / "x.py"))
        assert svc.enabled_for(str(big / "x.py")) is False
        assert svc.enabled_for(str(small / "x.py")) is True
        assert any("exclude_roots" in r.getMessage() and str(big) in r.getMessage() for r in caplog.records)
    finally:
        svc.shutdown()

    with caplog.at_level("WARNING", logger=eventlog.event_log.name):
        bad = _service_from({"install_strategy": "manual", "exclude_roots": str(big)})
    try:
        assert bad.enabled_for(str(small / "x.py")) is False
        warned = [r for r in caplog.records if r.levelno >= 30 and "exclude_roots" in r.getMessage()]
        assert warned and "list" in warned[0].getMessage() and "str" in warned[0].getMessage()
    finally:
        bad.shutdown()


def test_warmup_timeout_applies_to_cold_root_only(tmp_path, monkeypatch):
    """The first request against a root with no running client waits up to ``lsp.warmup_timeout``;
    once the client is up the steady-state ``wait_timeout`` applies again (#116446: a 55 s cold build
    was being cut off at the 5 s steady-state budget, then the root was poisoned)."""
    repo = _git_repo(tmp_path, "repo")
    src = repo / "x.py"
    monkeypatch.chdir(str(repo))
    svc = _service_from({"wait_timeout": 1.0, "warmup_timeout": 30.0, "install_strategy": "manual"})
    budgets: list = []

    class FakeClient:
        server_id = "pyright"
        workspace_root = str(repo)
        workspace_folders = (str(repo),)
        state = "running"
        is_running = True

        async def open_file(self, *_a, **_k):
            return 1

        async def save_file(self, *_a, **_k):
            return None

        async def wait_for_diagnostics(self, *_a, timeout=None, **_k):
            budgets.append(timeout)
            return True

        def diagnostics_for(self, *_a, **_k):
            return []

        async def shutdown(self):
            return None

    fake = FakeClient()

    async def spawn(_path):
        return fake

    try:
        with patch.object(svc, "_get_or_spawn", spawn):
            svc.snapshot_baseline(str(src))  # cold: no client registered for the root yet
            key = _client_key(find_server_for_file(str(src)), str(repo))
            with svc._state_lock:
                svc._clients[key] = fake
            assert svc.get_diagnostics_sync(str(src)) == []  # warm
    finally:
        svc.shutdown()
    assert budgets == [30.0, 1.0]
