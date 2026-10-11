"""Detached cron/kanban sessions are not implicitly exec children of a chat picker."""
from __future__ import annotations

import json

import pytest

from hermes_constants import get_hermes_home

PROVIDERS = {"ecc-router": {"base_url": "http://192.168.10.13:4000/v1", "api_key": "sk-x",
                            "opusplan": {"plan": "GLM-5.3-Flash-850K", "exec": "Qwen3.8FlashNext"}}}


def _config(default="opusplan", **model_extra):
    (get_hermes_home() / "config.yaml").write_text(json.dumps(
        {"model": {"default": default, "provider": "ecc-router", **model_extra}, "providers": PROVIDERS}))


def _job_model(job):
    import cron.scheduler as sched
    return sched._load_cron_job_config({"id": "j", "name": "j", "prompt": "x", **job}, "j", "j").model


class TestCronModel:
    def test_unpinned_job_under_opusplan_runs_the_plan_model(self):
        _config()
        assert _job_model({}) == "GLM-5.3-Flash-850K"

    def test_job_pinned_to_opusplan_runs_the_plan_model(self):
        _config("some-other-model")
        assert _job_model({"model": "opusplan", "provider": "ecc-router"}) == "GLM-5.3-Flash-850K"

    def test_cron_model_override_of_opusplan_is_honoured(self):
        _config()
        cfg = json.loads((get_hermes_home() / "config.yaml").read_text())
        cfg["cron"] = {"model": "opusplan", "model_provider": "ecc-router"}
        (get_hermes_home() / "config.yaml").write_text(json.dumps(cfg))
        assert _job_model({}) == "GLM-5.3-Flash-850K"

    def test_explicit_job_model_is_untouched(self):
        _config()
        assert _job_model({"model": "pinned"}) == "pinned"

    def test_no_pair_for_the_provider_fails_the_job_naming_it(self):
        (get_hermes_home() / "config.yaml").write_text(json.dumps(
            {"model": {"default": "opusplan", "provider": "acme-gw"}, "providers": {"acme-gw": {"base_url": "http://a/v1"}}}))
        from unittest.mock import patch
        with patch("providers.get_provider_profile", return_value=None), \
             pytest.raises(RuntimeError, match="'acme-gw'.*opusplan"):
            _job_model({})


class TestKanbanWorker:
    @staticmethod
    def _task(**kw):
        from hermes_cli.kanban_db import Task
        base = dict(id="t_op", title="t", body="b", assignee="worker", status="running", priority=0, created_by=None,
                    created_at=1, started_at=None, completed_at=None, workspace_kind="scratch", workspace_path=None,
                    claim_lock=None, claim_expires=None, tenant=None)
        base.update(kw)
        return Task(**base)

    @pytest.fixture(autouse=True)
    def _argv(self, monkeypatch):
        from hermes_cli import kanban_db_dispatch as dispatch
        monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
        monkeypatch.setattr(dispatch, "_resolve_worker_cli_toolsets", lambda home: None)
        self.dispatch = dispatch

    def test_worker_has_no_implicit_exec_override(self):
        _config()
        argv = self.dispatch._worker_argv(self._task(), "worker", str(get_hermes_home()))
        assert "-m" not in argv
        assert "--provider" not in argv

    def test_task_model_override_still_wins(self):
        _config()
        argv = self.dispatch._worker_argv(
            self._task(model_override="pinned", provider_override="openrouter"), "worker", str(get_hermes_home()))
        assert argv.count("-m") == 1 and argv[argv.index("-m") + 1] == "pinned"

    def test_ordinary_profile_gets_no_model_flag(self):
        _config("qwen3")
        assert "-m" not in self.dispatch._worker_argv(self._task(), "worker", str(get_hermes_home()))
