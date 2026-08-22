"""The drift guard's remediation must name the fleet-wide remedy, not only the pin.

#44585's guard fails an unpinned cron job closed when the global provider/model
moves under it. Its remediation has always named exactly one fix: pin this job.
That is the right advice for one job and the wrong advice for a fleet — #59031
was 34 jobs breaking on a single global switch, i.e. 34 edits for one change.

The fleet-wide answer already exists and is already documented
(``website/docs/user-guide/features/cron.md``, since #73532): ``cron.model`` /
``cron.model_provider`` give unpinned cron jobs their own default, and
``_cron_fleet_default_covers_axis`` makes the guard skip any axis they cover.
Three merged PRs built that path deliberately. None of them taught the runtime
messages to mention it, so an operator hitting the guard is told to solve an
O(1) problem O(n) times and has no way to discover the alternative.

These tests pin the mention into all three surfaces an operator can hit — the
log/alert blob, the chat delivery line, and the ``hermes config set`` warning —
and, more importantly, pin the *command in the message to the key the guard
actually reads*, so advice and mechanism cannot drift apart.

They also pin the exemption itself. #89242 proposes removing snapshots from
inherit-mode jobs, which would take the guard off exactly the population these
three PRs kept it on. The exemption test below is what that change would have
to break.
"""

from __future__ import annotations

import re

import pytest

from cron.scheduler import _summarize_cron_failure_for_delivery
from hermes_cli.config import (
    _cron_fleet_default_covers_axis,
    warn_unpinned_cron_jobs_after_model_config_change,
)

# Reuse the guard's own harness: these messages are emitted from inside
# ``run_job``, and a message test that stubs the guard instead of running it
# would keep passing after the guard stopped firing.
from tests.cron.test_cron_provider_pin import (
    _base_job,
    _run_with_current_provider_and_model,
)


# The exact config keys the guard consults. Written out rather than imported so
# a rename has to be made twice, deliberately.
FLEET_MODEL_KEY = "cron.model"
FLEET_PROVIDER_KEY = "cron.model_provider"


def _drift_error(tmp_path, **kwargs):
    """Run an unpinned job into the guard and return its error blob."""
    job = kwargs.pop("job", None) or _base_job(
        provider_snapshot="openrouter", model_snapshot="old-model"
    )
    success, output, _final, error, agent_constructed = (
        _run_with_current_provider_and_model(
            job, kwargs.pop("provider", "openrouter"),
            kwargs.pop("model", "new-model"), tmp_path, **kwargs
        )
    )
    return success, f"{error}\n{output}", error, agent_constructed


class TestTheSkipMessageNamesBothRemedies:

    def test_recurring_job_learns_about_the_cron_fleet_default(self, tmp_path):
        success, blob, _error, agent_constructed = _drift_error(tmp_path)

        assert success is False
        assert agent_constructed is False, "the guard must still fail closed"
        assert FLEET_MODEL_KEY in blob, (
            "the operator is told to pin one job at a time with no hint that a "
            "fleet-wide default exists — that is #59031's 34 edits"
        )
        assert "hermes config set" in blob

    def test_the_per_job_pin_advice_is_preserved(self, tmp_path):
        """Additive, not a replacement.

        Pinning is still the right answer for a single job that must not move,
        and #72056's lifecycle wording is still the right answer for a spent
        one-shot. This change adds a second remedy; it removes none.
        """
        _success, blob, _error, _constructed = _drift_error(tmp_path)

        assert "hermes cron edit pin-test --provider <provider> --model <model>" in blob

    def test_finite_oneshot_learns_about_it_too(self, tmp_path):
        """The path where the pin advice is useless is the one that needs it most.

        #72056 established that a consumed one-shot cannot be fixed by editing
        it. Until now the only alternative offered was "create a new one-shot
        with an explicit provider and model" — still per job.
        """
        job = _base_job(
            provider_snapshot="openrouter",
            model_snapshot="old-model",
            schedule={"kind": "once", "run_at": "2030-01-01T00:00:00Z"},
            repeat={"times": 1, "completed": 1},
        )
        _success, blob, _error, _constructed = _drift_error(tmp_path, job=job)

        assert "create a new one-shot job" in blob.lower()
        assert FLEET_MODEL_KEY in blob

    def test_the_provider_axis_is_named_as_well(self, tmp_path):
        """Two axes drift, and the fleet default has a separate key per axis.

        Naming only ``cron.model`` would send an operator whose *provider*
        moved to a setting that cannot help them.
        """
        _success, blob, _error, _constructed = _drift_error(
            tmp_path, provider="nous"
        )

        assert FLEET_PROVIDER_KEY in blob

    def test_the_chat_delivery_line_names_it(self, tmp_path):
        """The surface an operator actually reads.

        The log and the alert blob are where they look *after* something sent
        them there; the delivery line is what arrives unprompted.
        """
        _success, _blob, error, _constructed = _drift_error(tmp_path)

        delivered = _summarize_cron_failure_for_delivery(_base_job(), error)
        assert FLEET_MODEL_KEY in delivered
        assert "hermes config set" in delivered

    def test_the_delivery_line_stays_a_single_line(self, tmp_path):
        """``_summarize_cron_failure_for_delivery`` exists to be compact.

        Its docstring puts the full detail in the log and the output directory
        on purpose. A remedy sentence that turns the chat notification into a
        paragraph would defeat the function.
        """
        _success, _blob, error, _constructed = _drift_error(tmp_path)

        delivered = _summarize_cron_failure_for_delivery(_base_job(), error)
        assert "\n" not in delivered
        assert len(delivered) < 500


class TestTheAdviceMatchesTheMechanism:
    """A remedy in a message is a promise; these tests make it a checked one."""

    @pytest.mark.parametrize("axis,expected_key", [
        ("model", FLEET_MODEL_KEY),
        ("provider", FLEET_PROVIDER_KEY),
    ])
    def test_every_key_the_messages_name_is_one_the_guard_honors(
        self, axis, expected_key
    ):
        """Parse the key back out of the advice and feed it to the guard.

        This is the test that would have caught the message being stale in the
        first place, and the one that keeps it from going stale again: if
        ``_cron_fleet_default_covers_axis`` is ever taught a different key, the
        sentence telling operators to set the old one fails here rather than in
        somebody's cron log.
        """
        section = expected_key.split(".", 1)[1]
        assert _cron_fleet_default_covers_axis(
            axis, {"cron": {section: "some/value"}}
        ) is True
        assert _cron_fleet_default_covers_axis(axis, {"cron": {}}) is False

    def test_the_command_in_the_skip_message_parses_to_a_real_key(self, tmp_path):
        """No hand-written key strings: read them out of the emitted text."""
        _success, blob, _error, _constructed = _drift_error(tmp_path)

        keys = set(re.findall(r"hermes config set (cron\.[a-z_]+)", blob))
        assert keys, "the skip message names no `hermes config set` command"
        for dotted in keys:
            section = dotted.split(".", 1)[1]
            axis = "model" if section == "model" else "provider"
            assert _cron_fleet_default_covers_axis(
                axis, {"cron": {section: "some/value"}}
            ) is True, f"{dotted} is advertised but the guard ignores it"


class TestTheExemptionTheAdviceRelies_On:
    """The advice is only true because the guard already exempts covered axes.

    #89242 asks for inherit-mode jobs to stop recording snapshots at all, which
    would disarm the guard for the entire unpinned fleet rather than for the
    axes an operator deliberately routed. That is the outcome #73323 and #73532
    were merged to avoid and #61468 was closed over. These two tests are what
    such a change would have to break.
    """

    def test_a_covered_axis_does_not_drift(self, tmp_path):
        """Setting the fleet default really does stop the skip."""
        success, _blob, _error, agent_constructed = _drift_error(
            tmp_path, cron_model="fleet/model"
        )

        assert success is True
        assert agent_constructed is True

    def test_an_uncovered_axis_still_fails_closed(self, tmp_path):
        """And it stops it only for the axis it covers.

        A fleet default on the model must not buy an unpinned job a free ride
        onto a paid *provider* — that is the $7.73 half of #44585.
        """
        success, blob, _error, agent_constructed = _drift_error(
            tmp_path, provider="nous", cron_model="fleet/model"
        )

        assert success is False
        assert agent_constructed is False
        assert "provider" in blob.lower()


def _warn(key, value, jobs, config=None, monkeypatch=None):
    monkeypatch.setattr(
        "hermes_cli.config._load_cron_jobs_for_config_warning", lambda: jobs
    )
    warn_unpinned_cron_jobs_after_model_config_change(key, value, config or {})


def _unpinned_job(**overrides):
    job = {
        "id": "job-1",
        "name": "Morning summary",
        "enabled": True,
        "no_agent": False,
        "provider_snapshot": "openrouter",
        "model_snapshot": "old/model",
    }
    job.update(overrides)
    return job


class TestTheConfigSetWarningNamesBothRemedies:
    """The earlier of the two moments — before any job has failed yet.

    ``hermes config set model.default X`` is where an operator can still act
    before N jobs skip. Telling them only about per-job pinning here is the
    most expensive place to omit the fleet default.
    """

    def test_model_change_names_the_fleet_model_key(self, monkeypatch, capsys):
        _warn("model.default", "new/model", [_unpinned_job()], monkeypatch=monkeypatch)

        out = capsys.readouterr().out
        assert FLEET_MODEL_KEY in out
        assert "hermes cron edit" in out, "the per-job remedy is still offered"

    def test_provider_change_names_the_fleet_provider_key(self, monkeypatch, capsys):
        """Not ``cron.provider``.

        The fleet default's provider key is ``cron.model_provider``; the axis is
        called ``provider``. Interpolating the axis name into the command is the
        obvious way to write this line and produces a key that does not exist.
        """
        _warn("model.provider", "nous", [_unpinned_job()], monkeypatch=monkeypatch)

        out = capsys.readouterr().out
        assert FLEET_PROVIDER_KEY in out
        assert "cron.provider " not in out
        assert "cron.provider>" not in out

    def test_a_covered_axis_produces_no_warning_at_all(self, monkeypatch, capsys):
        """The advice is not offered to someone who already took it.

        An operator who has set ``cron.model`` is not affected by a chat-model
        change on that axis, so warning them would be a false alarm — and would
        advertise a remedy they are already using.
        """
        _warn(
            "model.default",
            "new/model",
            [_unpinned_job()],
            config={"cron": {"model": "fleet/model"}},
            monkeypatch=monkeypatch,
        )

        assert capsys.readouterr().out == ""

    def test_the_warning_still_says_what_will_happen(self, monkeypatch, capsys):
        """Behaviour preservation: the diagnosis is unchanged, only the cure grew."""
        _warn("model.default", "new/model", [_unpinned_job()], monkeypatch=monkeypatch)

        out = capsys.readouterr().out
        assert "fail closed" in out
        assert "model_snapshot" in out
        assert "hermes cron list" in out

    def test_the_key_the_warning_names_is_one_the_guard_honors(
        self, monkeypatch, capsys
    ):
        """Same round-trip as the skip message, at the earlier surface."""
        _warn("model.provider", "nous", [_unpinned_job()], monkeypatch=monkeypatch)

        out = capsys.readouterr().out
        keys = set(re.findall(r"hermes config set (cron\.[a-z_]+)", out))
        assert keys
        for dotted in keys:
            section = dotted.split(".", 1)[1]
            axis = "model" if section == "model" else "provider"
            assert _cron_fleet_default_covers_axis(
                axis, {"cron": {section: "some/value"}}
            ) is True, f"{dotted} is advertised but the guard ignores it"
