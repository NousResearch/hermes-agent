"""Tests for cron scheduler skip_memory configuration.

Regression for issue #9763: skip_memory was hardcoded to True in
cron/scheduler.py, making external memory providers (e.g. mem0) unusable
in cron jobs. Resolution is centralized in ``_resolve_cron_skip_memory``.

Cron opt-in is *provider-only*: ``run_job`` always passes
``skip_local_memory=True`` so MEMORY.md/USER.md injection stays off even
when ``cron.skip_memory: false`` (#52897 teknium review / 005e0ec).
"""

from cron.scheduler import _resolve_cron_skip_memory


class TestResolveCronSkipMemory:
    def test_defaults_to_true_when_no_cron_section(self):
        assert _resolve_cron_skip_memory({}) is True

    def test_defaults_to_true_when_skip_memory_missing(self):
        assert _resolve_cron_skip_memory({"cron": {}}) is True

    def test_false_when_configured(self):
        assert _resolve_cron_skip_memory({"cron": {"skip_memory": False}}) is False

    def test_true_when_explicitly_set(self):
        assert _resolve_cron_skip_memory({"cron": {"skip_memory": True}}) is True

    def test_other_cron_settings_do_not_affect_default(self):
        assert _resolve_cron_skip_memory({"cron": {"timeout": 300}}) is True

    def test_tolerates_non_dict_cron_section(self):
        assert _resolve_cron_skip_memory({"cron": None}) is True
        assert _resolve_cron_skip_memory(None) is True


class TestRunJobSkipMemoryWiring:
    """Prove run_job forwards resolved skip_memory + always skip_local_memory.

    Uses the FakeAgent constructor-kwargs capture seam from
    tests/cron/test_cron_workdir.py (teknium #52897 review).

    run_job loads cron config from HERMES_HOME/config.yaml (not load_config),
    so tests stage a temp hermes home with the desired yaml.
    """

    @staticmethod
    def _install_stubs(monkeypatch, observed: dict, tmp_path, cron_cfg: dict):
        import sys
        from pathlib import Path
        import cron.scheduler as sched

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["skip_memory"] = kwargs.get("skip_memory")
                observed["skip_local_memory"] = kwargs.get("skip_local_memory")

            def run_conversation(self, *_a, **_kw):
                return {"final_response": "done", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        from hermes_cli import runtime_provider as _rtp

        monkeypatch.setattr(
            _rtp,
            "resolve_runtime_provider",
            lambda **_kw: {
                "provider": "test",
                "api_key": "k",
                "base_url": "http://test.local",
                "api_mode": "chat_completions",
            },
        )
        monkeypatch.setattr(
            sched, "_build_job_prompt", lambda job, prerun_script=None: "hi"
        )
        monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(
            sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None
        )
        monkeypatch.setattr(
            sched, "_resolve_cron_disabled_toolsets", lambda cfg: None
        )
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

        home = Path(tmp_path)
        home.mkdir(parents=True, exist_ok=True)
        cfg_path = home / "config.yaml"
        # Minimal yaml; avoid importing pyyaml if string write is enough
        lines = ["model: test-model", "cron:"]
        if not cron_cfg:
            lines.append("  {}")
        else:
            for k, v in cron_cfg.items():
                rendered = (
                    "true"
                    if v is True
                    else "false"
                    if v is False
                    else repr(v)
                )
                lines.append(f"  {k}: {rendered}")
        cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(sched, "_get_hermes_home", lambda: home)

    def test_run_job_default_skip_memory_true_and_skip_local(
        self, tmp_path, monkeypatch
    ):
        import cron.scheduler as sched

        observed = {}
        self._install_stubs(monkeypatch, observed, tmp_path, {})

        ok, *_ = sched.run_job(
            {
                "id": "skipmem-default",
                "name": "skip memory default",
                "prompt": "hi",
                "model": "test-model",
            }
        )
        assert ok is True
        assert observed["skip_memory"] is True
        assert observed["skip_local_memory"] is True

    def test_run_job_provider_only_opt_in(self, tmp_path, monkeypatch):
        import cron.scheduler as sched

        observed = {}
        self._install_stubs(
            monkeypatch, observed, tmp_path, {"skip_memory": False}
        )

        ok, *_ = sched.run_job(
            {
                "id": "skipmem-opt-in",
                "name": "provider only",
                "prompt": "hi",
                "model": "test-model",
            }
        )
        assert ok is True
        # Provider-only: external memory providers may run…
        assert observed["skip_memory"] is False
        # …but local MEMORY.md/USER.md injection must stay off.
        assert observed["skip_local_memory"] is True
