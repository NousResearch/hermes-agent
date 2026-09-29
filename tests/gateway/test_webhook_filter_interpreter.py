"""Webhook filter scripts run under the Git-Bash-safe interpreter and log silent failures."""
import logging
import stat

import pytest

from gateway.platforms.webhook_filters import ScriptOutcome, WebhookRouteProcessor
from tools.environments import local


def _filter_script(body: str):
    from hermes_constants import get_hermes_home
    scripts = get_hermes_home() / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    filt = scripts / "filter.sh"
    filt.write_text(body, encoding="utf-8")
    return filt


def _fake_bash(tmp_path, body: str):
    script = tmp_path / "fake-bash"
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


@pytest.mark.platforms("linux")  # POSIX shebang fixture; the Windows half is the wine2e receipt
def test_shell_filter_runs_under_find_bash_interpreter(tmp_path, monkeypatch):
    """The .sh filter must be spawned through ``_find_bash()``, never a PATH/``which`` lookup (#116818)."""
    marker = tmp_path / "ran"
    fake = _fake_bash(tmp_path, f'touch "{marker}"\nprintf \'{{"ok": true}}\'\n')
    monkeypatch.setattr(local, "_find_bash", lambda: str(fake))
    filt = _filter_script("exit 99\n")  # would veto if run by a real bash

    result = WebhookRouteProcessor().run_route_script(str(filt), {"a": 1})

    assert marker.exists()
    assert result.outcome is ScriptOutcome.ACCEPTED and result.payload == {"ok": True}


@pytest.mark.platforms("linux")  # POSIX shebang fixture; the Windows half is the wine2e receipt
def test_silent_nonzero_exit_is_logged_as_warning(tmp_path, monkeypatch, caplog):
    """rc!=0 with no stdout AND no stderr is the interpreter-never-ran signature: WARNING, not INFO."""
    fake = _fake_bash(tmp_path, "exit 1\n")
    monkeypatch.setattr(local, "_find_bash", lambda: str(fake))
    filt = _filter_script("")

    with caplog.at_level(logging.INFO, logger="gateway.platforms.webhook_filters"):
        result = WebhookRouteProcessor().run_route_script(str(filt), {})

    assert result.outcome is ScriptOutcome.FAILED
    silent = [r for r in caplog.records if "script ignored webhook path=filter.sh" in r.getMessage()]
    assert silent and silent[0].levelno == logging.WARNING


def _no_bash():
    raise RuntimeError("bash not found")


@pytest.mark.parametrize("find_bash", [
    pytest.param(_no_bash, id="interpreter-missing"),
    pytest.param(lambda: "no-such-dir/bash", id="launch-error"),
])
def test_unlaunchable_filter_is_a_failure_not_a_veto(find_bash, monkeypatch):
    """A script that never started has not decided anything: FAILED (a retry_on_script_failure route answers
    503), never the IGNORED veto a script signals by exiting 0 silently."""
    monkeypatch.setattr(local, "_find_bash", find_bash)
    filt = _filter_script("exit 0\n")

    assert WebhookRouteProcessor().run_route_script(str(filt), {}).outcome is ScriptOutcome.FAILED
