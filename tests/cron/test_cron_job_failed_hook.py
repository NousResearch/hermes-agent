"""Tests for the ``cron_job_failed`` lifecycle hook.

The hook fires from the cron scheduler's failure path
(``cron/scheduler.py::_save_compose_deliver``, reached via ``run_one_job``)
when a job finishes with ``success=False``, so reactive consumers (shell
hooks, outbound webhooks, plugins) get a programmatic failure signal without
polling ``jobs.json`` or wrapping per-job delivery paths.  It must:

* fire exactly once per failed run, with the full job payload
* NOT fire on success
* fire BEFORE the failure notice is delivered, so a broken/blocking hook can
  never suppress or delay it
* never raise into the scheduler (a broken hook cannot crash the job loop)
"""

import pytest

import cron.scheduler as s


@pytest.fixture
def run_env(monkeypatch, tmp_path):
    """Drive ``run_one_job`` with the real delivery path down to a fake sender.

    Same shape as ``tests/cron/test_cron_failure_deliver.py``: bookkeeping
    primitives are stubbed, but ``_deliver_result`` and the real
    ``_fire_cron_job_failed_hook`` run unless a test patches them.  Returns an
    ordered ``events`` list; a delivery appends ``("deliver", chat_id, text)``.
    """
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    events = []

    async def fake_sender(pconfig, chat_id, message, *, thread_id=None,
                          media_files=None, force_document=False, caption=None):
        events.append(("deliver", chat_id, message))
        return {"success": True, "chat_id": chat_id, "message_id": "1.2"}

    import gateway.platform_registry as reg
    import hermes_cli.plugins as hp

    entry = reg.platform_registry.get("slack")
    if entry is None:
        hp.discover_plugins()
        entry = reg.platform_registry.get("slack")
    if entry is None:
        pytest.skip("slack platform entry not registered")
    monkeypatch.setattr(entry, "standalone_sender_fn", fake_sender)
    monkeypatch.setattr(hp, "discover_plugins", lambda *a, **k: None)

    monkeypatch.setattr(s, "create_execution", lambda *_a, **_kw: {"id": "exec-t"})
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: {})
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **kw: True)
    monkeypatch.setattr(s, "finish_execution", lambda *a, **kw: None)
    # No durable incident store in play: never acked, no id.
    monkeypatch.setattr(s, "_upsert_incident_for_failure", lambda *_a, **_kw: (False, None))
    monkeypatch.setattr(s, "load_config", lambda: {})
    return events


def _failing_run_job(error="provider exploded"):
    def _fake(job, **_kw):
        return (False, "raw output", "", error)
    return _fake


def _succeeding_run_job(final="all good, here is the brief"):
    def _fake(job, **_kw):
        return (True, "raw output", final, None)
    return _fake


class TestFireSite:
    def test_failed_job_fires_hook_once_with_job_and_error(self, run_env, monkeypatch):
        monkeypatch.setattr(s, "run_job", _failing_run_job("boom"))
        fired = []
        monkeypatch.setattr(
            s, "_fire_cron_job_failed_hook", lambda job, err: fired.append((job, err))
        )

        job = {"id": "jfail", "name": "nightly", "profile": "work", "deliver": "slack:D0MAIN"}
        s.run_one_job(job)

        assert len(fired) == 1
        assert fired[0][0] is job
        assert fired[0][1] == "boom"

    def test_successful_job_does_not_fire_hook(self, run_env, monkeypatch):
        monkeypatch.setattr(s, "run_job", _succeeding_run_job())
        fired = []
        monkeypatch.setattr(
            s, "_fire_cron_job_failed_hook", lambda job, err: fired.append((job, err))
        )

        ok = s.run_one_job({"id": "jok", "name": "ok-job", "deliver": "slack:D0MAIN"})

        assert ok is True
        assert fired == []

    def test_hook_fires_before_delivery(self, run_env, monkeypatch):
        """A blocking hook must not be able to delay/suppress the failure notice."""
        monkeypatch.setattr(s, "run_job", _failing_run_job("boom"))
        monkeypatch.setattr(
            s, "_fire_cron_job_failed_hook",
            lambda job, err: run_env.append(("hook", job["id"])),
        )

        s.run_one_job({"id": "jorder", "name": "ordered", "deliver": "slack:D0MAIN"})

        assert run_env, "expected the failure notice to be delivered"
        assert run_env[0][0] == "hook"
        assert any(e[0] == "deliver" for e in run_env)

    def test_real_hook_swallows_plugin_exception_end_to_end(self, run_env, monkeypatch):
        """A raising plugin dispatch is swallowed: run_one_job still completes."""
        monkeypatch.setattr(s, "run_job", _failing_run_job("boom"))
        import hermes_cli.plugins as plugins

        def raising_invoke_hook(name, **kwargs):
            raise RuntimeError("plugin dispatch failed")

        monkeypatch.setattr(plugins, "invoke_hook", raising_invoke_hook)

        # Must not raise even though the real hook dispatch raises internally.
        s.run_one_job({"id": "jboom", "name": "boom-job", "deliver": "slack:D0MAIN"})

    def test_escaped_exception_fires_hook(self, run_env, monkeypatch):
        """A crash out of run_job is a failure too and must fire the hook."""
        monkeypatch.setattr(
            s, "run_job",
            lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("cannot import name X")),
        )
        fired = []
        monkeypatch.setattr(
            s, "_fire_cron_job_failed_hook", lambda job, err: fired.append((job, err))
        )

        s.run_one_job({"id": "jcrashed", "name": "crasher", "deliver": "slack:D0MAIN"})

        assert len(fired) == 1
        assert fired[0][0]["id"] == "jcrashed"
        assert "cannot import name X" in fired[0][1]


class TestRealHookFunction:
    def test_dispatches_through_plugins_invoke_hook(self, monkeypatch):
        import hermes_cli.plugins as plugins

        calls = []

        def fake_invoke_hook(name, **kwargs):
            calls.append((name, kwargs))

        monkeypatch.setattr(plugins, "invoke_hook", fake_invoke_hook)

        job = {
            "id": "jreal", "name": "real-job", "profile": "ops",
            "last_run_at": "2026-08-10T00:00:00Z",
        }
        s._fire_cron_job_failed_hook(job, "kaput")

        assert len(calls) == 1
        name, kwargs = calls[0]
        assert name == "cron_job_failed"
        assert kwargs["job_id"] == "jreal"
        assert kwargs["job_name"] == "real-job"
        assert kwargs["profile"] == "ops"
        assert kwargs["error"] == "kaput"
        assert kwargs["last_run_at"] == "2026-08-10T00:00:00Z"
        assert kwargs["job"] is job

    def test_job_name_falls_back_to_id_and_empty_defaults(self, monkeypatch):
        import hermes_cli.plugins as plugins

        calls = []
        monkeypatch.setattr(
            plugins, "invoke_hook", lambda name, **kwargs: calls.append(kwargs)
        )

        s._fire_cron_job_failed_hook({"id": "jonly"}, None)

        assert calls[0]["job_name"] == "jonly"
        assert calls[0]["error"] == ""
        assert calls[0]["profile"] == ""
        assert calls[0]["last_run_at"] == ""

    def test_swallows_exception(self, monkeypatch):
        import hermes_cli.plugins as plugins

        def raising_invoke_hook(name, **kwargs):
            raise RuntimeError("plugin dispatch failed")

        monkeypatch.setattr(plugins, "invoke_hook", raising_invoke_hook)

        # Must not raise.
        s._fire_cron_job_failed_hook({"id": "jx", "name": "x"}, "err")
