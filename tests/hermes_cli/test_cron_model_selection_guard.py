"""`hermes cron create/edit --model` must run the selection-guard registry.

`hermes_cli.model_selection_guards` is the single evaluation point for
selection-time guards, and its module docstring enumerates the surfaces that
call it: CLI picker, TUI, dashboard, gateway `/model`, Telegram/Discord
pickers, TUI-gateway RPC. Cron is on none of them, even though `--model` pins
the model an unattended job runs on every tick.

There is no second chance later: `cron/scheduler.py` reads `job["model"]` and
constructs the `AIAgent` in-process inside the gateway, so it never re-enters
`hermes_cli/main.py` and `_confirm_startup_expensive_model_override` cannot
fire for a scheduled run. Create/edit time is the only reachable confirm point.

`muse-spark-1.2-contributor` is the repo's own data-training-tier id (see
`hermes_cli/model_data_policy_guard.py`), so these tests drive the real
registry end to end instead of stubbing a warning payload. The provider is
left as "custom" / unset throughout, which keeps the cost guard's pricing
lookups offline.
"""

import io
import sys
from argparse import Namespace

import pytest

from hermes_cli import cron as cron_cli

CONTRIBUTOR_MODEL = "muse-spark-1.2-contributor"
ORDINARY_MODEL = "some/ordinary-model"


class _FakeTtyStdin(io.StringIO):
    """stdin that reports itself as a terminal, so the confirm branch runs."""

    def isatty(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def stub_config(monkeypatch):
    """Keep the guard off the developer's real on-disk config.yaml."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"default": ORDINARY_MODEL}},
    )


@pytest.fixture()
def api_calls(monkeypatch):
    """Record `_cron_api` calls instead of writing to the real job store."""
    calls = []

    def _fake_api(**kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "job_id": "job-1",
            "name": "Nightly digest",
            "schedule": "*/10 * * * *",
            "next_run_at": "2026-01-01T00:00:00Z",
            "job": {
                "job_id": "job-1",
                "name": "Nightly digest",
                "schedule": "*/10 * * * *",
            },
        }

    monkeypatch.setattr(cron_cli, "_cron_api", _fake_api)
    monkeypatch.setattr(cron_cli, "_warn_if_gateway_not_running", lambda: None)
    return calls


@pytest.fixture()
def stored_job(monkeypatch):
    """A job already on a schedule, returned by `cron edit`'s ref lookup."""
    job = {
        "id": "job-1",
        "name": "Nightly digest",
        "schedule": "*/10 * * * *",
        "model": None,
        "provider": None,
        "skills": [],
    }
    monkeypatch.setattr("cron.jobs.resolve_job_ref", lambda ref: job)
    return job


def _answer(monkeypatch, reply):
    """Make the confirm prompt interactive and answer it with `reply`."""
    monkeypatch.setattr(sys, "stdin", _FakeTtyStdin(reply + "\n"))
    monkeypatch.setattr("builtins.input", lambda prompt="": reply)


def _non_interactive(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))


def _never_prompts(monkeypatch):
    def _boom(prompt=""):
        raise AssertionError("the guard prompted when no warning should fire")

    monkeypatch.setattr("builtins.input", _boom)


def _create_args(**overrides):
    args = Namespace(
        cron_command="create",
        schedule="*/10 * * * *",
        prompt="Summarize my inbox",
        name="Nightly digest",
        deliver=None,
        repeat=None,
        skill=None,
        skills=None,
        script=None,
        workdir=None,
        model=None,
        model_provider=None,
        no_agent=False,
        monitor_script=None,
        monitor_url=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _edit_args(**overrides):
    args = Namespace(
        cron_command="edit",
        job_id="job-1",
        schedule=None,
        prompt=None,
        name=None,
        deliver=None,
        repeat=None,
        skill=None,
        skills=None,
        clear_skills=False,
        add_skills=None,
        remove_skills=None,
        script=None,
        workdir=None,
        model=None,
        model_provider=None,
        no_agent=None,
        monitor_script=None,
        monitor_url=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestCronCreatePinGuard:

    def test_declined_pin_never_reaches_the_job_store(
        self, monkeypatch, api_calls, capsys
    ):
        _answer(monkeypatch, "n")

        rc = cron_cli.cron_create(_create_args(model=CONTRIBUTOR_MODEL))

        assert rc == 1
        assert api_calls == [], "a declined pin must not create the job"
        err = capsys.readouterr().err
        assert "TRAINS ON YOUR DATA" in err
        assert "Model pin cancelled." in err

    def test_confirmed_pin_is_stored_after_the_warning_is_shown(
        self, monkeypatch, api_calls, capsys
    ):
        _answer(monkeypatch, "y")

        rc = cron_cli.cron_create(_create_args(model=CONTRIBUTOR_MODEL))

        assert rc == 0
        assert len(api_calls) == 1
        assert api_calls[0]["model"] == CONTRIBUTOR_MODEL
        # The block has to actually reach the user — a confirm nobody was
        # shown is the bug, not the fix.
        assert "TRAINS ON YOUR DATA" in capsys.readouterr().err

    def test_non_interactive_pin_is_refused(self, monkeypatch, api_calls, capsys):
        _non_interactive(monkeypatch)

        rc = cron_cli.cron_create(_create_args(model=CONTRIBUTOR_MODEL))

        assert rc == 1
        assert api_calls == []
        err = capsys.readouterr().err
        assert "TRAINS ON YOUR DATA" in err
        assert "non-interactive mode" in err

    def test_provider_only_pin_is_judged_against_the_configured_default(
        self, monkeypatch, api_calls
    ):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"model": {"default": CONTRIBUTOR_MODEL}},
        )
        _answer(monkeypatch, "n")

        rc = cron_cli.cron_create(_create_args(model_provider="custom"))

        assert rc == 1
        assert api_calls == []

    def test_pin_that_fires_no_guard_is_not_blocked(self, monkeypatch, api_calls):
        _never_prompts(monkeypatch)

        rc = cron_cli.cron_create(_create_args(model=ORDINARY_MODEL))

        assert rc == 0
        assert len(api_calls) == 1

    def test_no_pin_never_evaluates_the_guard(self, monkeypatch, api_calls):
        evaluated = []
        monkeypatch.setattr(
            "hermes_cli.model_selection_guards.combined_selection_warning",
            lambda *args, **kwargs: evaluated.append(args),
        )
        _never_prompts(monkeypatch)

        rc = cron_cli.cron_create(_create_args())

        assert rc == 0
        assert evaluated == [], "the common path must not touch the registry"
        assert len(api_calls) == 1

    def test_guard_failure_never_blocks_job_creation(self, monkeypatch, api_calls):
        def _boom(*args, **kwargs):
            raise RuntimeError("bad guard")

        monkeypatch.setattr(
            "hermes_cli.model_selection_guards.combined_selection_warning", _boom
        )
        _never_prompts(monkeypatch)

        rc = cron_cli.cron_create(_create_args(model=CONTRIBUTOR_MODEL))

        assert rc == 0
        assert len(api_calls) == 1


class TestCronEditPinGuard:

    def test_declined_pin_never_updates_the_job(
        self, monkeypatch, api_calls, stored_job, capsys
    ):
        _answer(monkeypatch, "n")

        rc = cron_cli.cron_edit(_edit_args(model=CONTRIBUTOR_MODEL))

        assert rc == 1
        assert api_calls == [], "a declined pin must not update the job"
        err = capsys.readouterr().err
        assert "TRAINS ON YOUR DATA" in err
        assert "Model pin cancelled." in err

    def test_provider_only_pin_is_judged_against_the_jobs_current_model(
        self, monkeypatch, api_calls, stored_job
    ):
        # Re-routing an existing pin to another provider must be judged against
        # the model the job is actually pinned to, not the config default.
        stored_job["model"] = CONTRIBUTOR_MODEL
        _answer(monkeypatch, "n")

        rc = cron_cli.cron_edit(_edit_args(model_provider="custom"))

        assert rc == 1
        assert api_calls == []

    def test_confirmed_pin_is_stored_after_the_warning_is_shown(
        self, monkeypatch, api_calls, stored_job, capsys
    ):
        _answer(monkeypatch, "y")

        rc = cron_cli.cron_edit(_edit_args(model=CONTRIBUTOR_MODEL))

        assert rc == 0
        assert len(api_calls) == 1
        assert api_calls[0]["model"] == CONTRIBUTOR_MODEL
        assert "TRAINS ON YOUR DATA" in capsys.readouterr().err

    def test_clearing_the_model_while_setting_a_provider_drops_the_old_pin(
        self, monkeypatch, api_calls, stored_job
    ):
        # `--model ""` removes the pin in the same command that sets a
        # provider, so the guard must judge what the job will run on AFTER the
        # edit (the fleet default), not the model being taken away.
        stored_job["model"] = CONTRIBUTOR_MODEL
        _never_prompts(monkeypatch)

        rc = cron_cli.cron_edit(_edit_args(model="", model_provider="custom"))

        assert rc == 0
        assert len(api_calls) == 1

    def test_clearing_the_model_falls_through_to_the_cron_fleet_default(
        self, monkeypatch, api_calls, stored_job
    ):
        # The other half of the same rule: once the job's own pin is dropped,
        # resolution continues down the scheduler's chain, so a guarded
        # cron.model must still be caught.
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {
                "cron": {"model": CONTRIBUTOR_MODEL},
                "model": {"default": ORDINARY_MODEL},
            },
        )
        stored_job["model"] = ORDINARY_MODEL
        _answer(monkeypatch, "n")

        rc = cron_cli.cron_edit(_edit_args(model="", model_provider="custom"))

        assert rc == 1
        assert api_calls == []

    def test_clearing_a_pin_is_never_guarded(
        self, monkeypatch, api_calls, stored_job
    ):
        # `--model ""` is the documented way to drop back to cron.model /
        # model.default. De-escalation must not prompt, even on a job whose
        # current pin would fire every guard in the registry.
        stored_job["model"] = CONTRIBUTOR_MODEL
        _never_prompts(monkeypatch)

        rc = cron_cli.cron_edit(_edit_args(model="", model_provider=""))

        assert rc == 0
        assert len(api_calls) == 1
        assert api_calls[0]["model"] == ""
