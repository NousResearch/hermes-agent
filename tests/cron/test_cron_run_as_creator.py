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
"""
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
