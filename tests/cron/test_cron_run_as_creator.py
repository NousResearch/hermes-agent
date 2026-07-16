"""Cron ``run_as_creator``: opt-in execution under the scheduling user's identity.

By default a cron job runs the agent with no sender identity — ``run_job`` seeds
``HERMES_SESSION_USER_ID = ""`` — so sender-scoped tools (per-user credentials,
access control, rate limits, personalization) fall back to a service/anon path.
That silently *widens* what a scheduled job can reach beyond what its creator
could reach interactively.

``origin`` already captures the creator's ``user_id`` at create time
(``_origin_from_env``); it is used for delivery/mirror routing but NOT for the
agent run. These tests pin the opt-in that threads it into the agent-run session
(``user_id`` only — ``platform``/``chat_id`` stay cron-neutral by design) and,
critically, that the default stays OFF so existing deployments are unchanged.

Coverage is end-to-end, not just the helpers:

- ``TestRunJobSeedsCreatorIdentity`` runs the real ``run_job`` (agent stubbed)
  and observes the ContextVars the agent's tools would actually see.
- ``TestCronjobToolWiring`` drives the REGISTERED ``cronjob`` handler — the
  same ``args``-dict path a model call takes — because a parameter that exists
  on the Python function but is dropped by the registry lambda is unreachable
  in production.
- ``TestDefaultConfig`` pins ``cron.run_as_creator: False`` in DEFAULT_CONFIG.
"""
import json

import cron.scheduler as scheduler


class TestRunAsCreatorEnabled:
    def test_default_off(self):
        # No per-job field, no config -> off (byte-for-byte legacy behaviour).
        assert scheduler._cron_run_as_creator_enabled({}, cfg={}) is False

    def test_global_flag_on(self):
        cfg = {"cron": {"run_as_creator": True}}
        assert scheduler._cron_run_as_creator_enabled({}, cfg=cfg) is True

    def test_global_flag_off_explicit(self):
        cfg = {"cron": {"run_as_creator": False}}
        assert scheduler._cron_run_as_creator_enabled({}, cfg=cfg) is False

    def test_per_job_true_overrides_global_off(self):
        cfg = {"cron": {"run_as_creator": False}}
        job = {"run_as_creator": True}
        assert scheduler._cron_run_as_creator_enabled(job, cfg=cfg) is True

    def test_per_job_false_overrides_global_on(self):
        cfg = {"cron": {"run_as_creator": True}}
        job = {"run_as_creator": False}
        assert scheduler._cron_run_as_creator_enabled(job, cfg=cfg) is False

    def test_non_bool_per_job_falls_through_to_global(self):
        # Garbage per-job value must not be treated as a decisive override.
        cfg = {"cron": {"run_as_creator": True}}
        job = {"run_as_creator": "yes"}
        assert scheduler._cron_run_as_creator_enabled(job, cfg=cfg) is True

    def test_malformed_cfg_defaults_off(self):
        assert scheduler._cron_run_as_creator_enabled({}, cfg={"cron": None}) is False


class TestCreatorUserId:
    _CFG_ON = {"cron": {"run_as_creator": True}}

    def test_disabled_returns_empty_even_with_origin(self):
        # Feature off -> anonymous cron context regardless of a captured creator.
        origin = {"platform": "telegram", "chat_id": "42", "user_id": "8223344881"}
        assert scheduler._cron_creator_user_id({}, origin, cfg={}) == ""

    def test_enabled_seeds_origin_user_id(self):
        origin = {"platform": "telegram", "chat_id": "42", "user_id": "8223344881"}
        assert (
            scheduler._cron_creator_user_id({}, origin, cfg=self._CFG_ON)
            == "8223344881"
        )

    def test_enabled_but_no_origin_returns_empty(self):
        # API/script-created jobs have no origin -> unchanged anonymous behaviour.
        assert scheduler._cron_creator_user_id({}, None, cfg=self._CFG_ON) == ""

    def test_enabled_but_origin_without_user_id_returns_empty(self):
        origin = {"platform": "telegram", "chat_id": "42"}
        assert scheduler._cron_creator_user_id({}, origin, cfg=self._CFG_ON) == ""

    def test_per_job_opt_in_seeds_without_global_flag(self):
        origin = {"platform": "discord", "chat_id": "7", "user_id": "1271274566"}
        job = {"run_as_creator": True}
        assert (
            scheduler._cron_creator_user_id(job, origin, cfg={})
            == "1271274566"
        )

    def test_user_id_coerced_to_str(self):
        # Some stores round-trip ids as ints; the session var must be a str.
        origin = {"platform": "telegram", "chat_id": "42", "user_id": 8223344881}
        out = scheduler._cron_creator_user_id({}, origin, cfg=self._CFG_ON)
        assert out == "8223344881"
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# run_job execution path: the ContextVars the agent's tools actually see
# ---------------------------------------------------------------------------

class TestRunJobSeedsCreatorIdentity:
    """Run the real ``run_job`` (agent stubbed, same harness as the workdir
    tests) and observe the session ContextVars DURING agent execution via
    ``get_session_env`` — the exact accessor sender-scoped tools use."""

    @staticmethod
    def _install_stubs(monkeypatch, observed: dict, config: dict):
        """Patch run_job's heavy deps; capture session vars inside the run."""
        import sys
        import cron.scheduler as sched

        from gateway.session_context import get_session_env

        class FakeAgent:
            def __init__(self, **kwargs):
                pass

            def run_conversation(self, *_a, **_kw):
                # Snapshot what a sender-scoped tool would resolve mid-run.
                observed["user_id"] = get_session_env("HERMES_SESSION_USER_ID")
                observed["platform"] = get_session_env("HERMES_SESSION_PLATFORM")
                observed["chat_id"] = get_session_env("HERMES_SESSION_CHAT_ID")
                return {"final_response": "done", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        # Bypass the real provider resolver — it reads ~/.hermes and credentials.
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

        # Stub scheduler helpers that would otherwise hit the filesystem /
        # config. NOTE ``_resolve_origin`` (the delivery-routing filter) is
        # stubbed to None on purpose: creator identity must read the job's RAW
        # stored origin, so it has to survive an unresolvable delivery origin.
        monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None: "hi")
        monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
        # Hermetic config: a dev box's real ~/.hermes/config.yaml must not
        # decide these tests' global-flag state.
        monkeypatch.setattr(sched, "load_config", lambda *a, **k: config)
        # Unlimited inactivity so the poll loop returns immediately.
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

    _ORIGIN = {"platform": "telegram", "chat_id": "42", "user_id": "8223344881"}

    def _run(self, monkeypatch, job_fields: dict, config: dict) -> dict:
        import cron.scheduler as sched

        observed: dict = {}
        self._install_stubs(monkeypatch, observed, config)
        job = {"id": "rac1", "name": "rac-job", "schedule_display": "manual"}
        job.update(job_fields)
        success, _output, response, error = sched.run_job(job)
        assert success is True, f"run_job failed: error={error!r} response={response!r}"
        assert observed, "FakeAgent.run_conversation never ran"
        return observed

    def test_default_off_runs_anonymous(self, monkeypatch):
        # No flag anywhere: byte-for-byte legacy — empty identity during the run.
        observed = self._run(monkeypatch, {"origin": dict(self._ORIGIN)}, config={})
        assert observed["user_id"] == ""
        assert observed["platform"] == ""
        assert observed["chat_id"] == ""

    def test_per_job_flag_seeds_user_id_only(self, monkeypatch):
        observed = self._run(
            monkeypatch,
            {"origin": dict(self._ORIGIN), "run_as_creator": True},
            config={},
        )
        assert observed["user_id"] == "8223344881"
        # platform/chat_id stay cron-neutral — the origin-chat leakage the
        # empty seeding exists to prevent must not come back with the flag.
        assert observed["platform"] == ""
        assert observed["chat_id"] == ""

    def test_global_flag_seeds_user_id(self, monkeypatch):
        observed = self._run(
            monkeypatch,
            {"origin": dict(self._ORIGIN)},
            config={"cron": {"run_as_creator": True}},
        )
        assert observed["user_id"] == "8223344881"
        assert observed["platform"] == ""
        assert observed["chat_id"] == ""

    def test_per_job_false_vetoes_global_on(self, monkeypatch):
        observed = self._run(
            monkeypatch,
            {"origin": dict(self._ORIGIN), "run_as_creator": False},
            config={"cron": {"run_as_creator": True}},
        )
        assert observed["user_id"] == ""

    def test_enabled_without_origin_stays_anonymous(self, monkeypatch):
        # API/script-created job: no origin captured -> nothing to seed.
        observed = self._run(
            monkeypatch,
            {"run_as_creator": True},
            config={},
        )
        assert observed["user_id"] == ""

    def test_identity_cleared_after_run(self, monkeypatch):
        from gateway.session_context import get_session_env

        self._run(
            monkeypatch,
            {"origin": dict(self._ORIGIN), "run_as_creator": True},
            config={},
        )
        # run_job's finally must not leak the creator identity into whatever
        # this thread/context does next.
        assert get_session_env("HERMES_SESSION_USER_ID") == ""


# ---------------------------------------------------------------------------
# cronjob tool wiring: the REGISTERED handler end-to-end (schema -> store)
# ---------------------------------------------------------------------------

class TestCronjobToolWiring:
    """Drive the registered ``cronjob`` handler with a raw ``args`` dict — the
    path a model call takes. A parameter accepted by the ``cronjob()`` Python
    function but dropped by the registry lambda is unreachable in production,
    so these tests go through the lambda, not the function."""

    @staticmethod
    def _handler():
        import tools.cronjob_tools  # noqa: F401  (ensures registration ran)
        from tools.registry import registry

        entry = registry.get_entry("cronjob")
        assert entry is not None, "cronjob tool is not registered"
        return entry.handler

    @staticmethod
    def _create(handler, tmp_path, extra: dict) -> dict:
        from cron.jobs import get_job, use_cron_store

        args = {"action": "create", "schedule": "every 1h", "prompt": "say hi"}
        args.update(extra)
        with use_cron_store(tmp_path):
            out = json.loads(handler(args))
            assert out.get("success") is True, f"create failed: {out}"
            stored = get_job(out["job_id"])
        assert stored is not None
        return stored

    def test_schema_exposes_run_as_creator(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA

        params = CRONJOB_SCHEMA["parameters"]
        assert isinstance(params, dict)
        assert "run_as_creator" in params["properties"]

    def test_create_persists_true(self, tmp_path, monkeypatch):
        stored = self._create(self._handler(), tmp_path, {"run_as_creator": True})
        assert stored.get("run_as_creator") is True

    def test_create_persists_explicit_false(self, tmp_path, monkeypatch):
        # False is a per-job veto of a global True — it must be STORED, not
        # collapsed into "absent".
        stored = self._create(self._handler(), tmp_path, {"run_as_creator": False})
        assert stored.get("run_as_creator") is False
        assert "run_as_creator" in stored

    def test_create_default_leaves_key_absent(self, tmp_path, monkeypatch):
        # Absent key => the global cron.run_as_creator config decides at run
        # time; existing jobs.json stays byte-identical for non-users.
        stored = self._create(self._handler(), tmp_path, {})
        assert "run_as_creator" not in stored

    def test_update_flips_flag(self, tmp_path, monkeypatch):
        from cron.jobs import get_job, use_cron_store

        handler = self._handler()
        stored = self._create(handler, tmp_path, {"run_as_creator": True})
        with use_cron_store(tmp_path):
            out = json.loads(handler({
                "action": "update",
                "job_id": stored["id"],
                "run_as_creator": False,
            }))
            assert out.get("success") is True, f"update failed: {out}"
            updated = get_job(stored["id"])
        assert updated is not None
        assert updated.get("run_as_creator") is False

    def test_update_without_field_leaves_stored_value(self, tmp_path, monkeypatch):
        # Flip-only semantics (documented in the schema and cron.md): omitting
        # the field on update must NOT clear or change the stored override.
        from cron.jobs import get_job, use_cron_store

        handler = self._handler()
        stored = self._create(handler, tmp_path, {"run_as_creator": False})
        with use_cron_store(tmp_path):
            out = json.loads(handler({
                "action": "update",
                "job_id": stored["id"],
                "name": "renamed",
            }))
            assert out.get("success") is True, f"update failed: {out}"
            updated = get_job(stored["id"])
        assert updated is not None
        assert updated.get("run_as_creator") is False

    def test_list_surfaces_flag(self, tmp_path, monkeypatch):
        from tools.cronjob_tools import _format_job

        stored = self._create(self._handler(), tmp_path, {"run_as_creator": True})
        assert _format_job(stored).get("run_as_creator") is True

    def test_list_surfaces_explicit_false_veto(self, tmp_path, monkeypatch):
        # The audit surface must show a stored False too — it is a per-job veto
        # of a global True, not "unset". Pins the presence-check in _format_job
        # against a truthiness-style regression (`if job.get(...)`).
        from tools.cronjob_tools import _format_job

        stored = self._create(self._handler(), tmp_path, {"run_as_creator": False})
        assert _format_job(stored).get("run_as_creator") is False

    def test_create_captures_session_user_id_into_origin(self, tmp_path, monkeypatch):
        # The load-bearing seam of the whole feature: _origin_from_env must
        # capture the scheduling user's HERMES_SESSION_USER_ID into the stored
        # origin at create time — run_as_creator seeds identity from THAT.
        # Without this pin, dropping the capture turns the feature into a
        # silent no-op for every chat-created job and no test notices.
        from gateway.session_context import clear_session_vars, set_session_vars

        handler = self._handler()
        tokens = set_session_vars(
            platform="telegram",
            chat_id="42",
            user_id="8223344881",
        )
        try:
            stored = self._create(handler, tmp_path, {"run_as_creator": True})
        finally:
            clear_session_vars(tokens)
        origin = stored.get("origin")
        assert isinstance(origin, dict)
        assert origin.get("user_id") == "8223344881"

    def test_full_chain_create_then_run_seeds_creator(self, tmp_path, monkeypatch):
        # End-to-end: tool create (origin captured from session vars) -> the
        # STORED job -> run_job -> agent sees the creator's user_id.
        from gateway.session_context import clear_session_vars, set_session_vars

        handler = self._handler()
        tokens = set_session_vars(
            platform="telegram",
            chat_id="42",
            user_id="8223344881",
        )
        try:
            stored = self._create(handler, tmp_path, {"run_as_creator": True})
        finally:
            clear_session_vars(tokens)

        import cron.scheduler as sched

        observed: dict = {}
        TestRunJobSeedsCreatorIdentity._install_stubs(monkeypatch, observed, config={})
        success, _output, response, error = sched.run_job(stored)
        assert success is True, f"run_job failed: error={error!r} response={response!r}"
        assert observed["user_id"] == "8223344881"
        assert observed["platform"] == ""
        assert observed["chat_id"] == ""


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG: the global switch ships, discoverable and off
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_default_config_ships_run_as_creator_off(self):
        from hermes_cli.config import DEFAULT_CONFIG

        cron_cfg = DEFAULT_CONFIG["cron"]
        assert isinstance(cron_cfg, dict)
        assert cron_cfg["run_as_creator"] is False
