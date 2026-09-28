"""Webhook filter scripts run under the Git-Bash-safe interpreter and log silent failures."""
import logging
import stat

import pytest

from gateway.platforms.webhook_filters import WebhookRouteProcessor
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


@pytest.mark.linux_only  # POSIX shebang fixture; the Windows half is the wine2e receipt
def test_shell_filter_runs_under_find_bash_interpreter(tmp_path, monkeypatch):
    """The .sh filter must be spawned through ``_find_bash()``, never a PATH/``which`` lookup (#116818)."""
    marker = tmp_path / "ran"
    fake = _fake_bash(tmp_path, f'touch "{marker}"\nprintf \'{{"ok": true}}\'\n')
    monkeypatch.setattr(local, "_find_bash", lambda: str(fake))
    filt = _filter_script("exit 99\n")  # would veto if run by a real bash

    accepted, transformed = WebhookRouteProcessor().run_route_script(str(filt), {"a": 1})

    assert marker.exists()
    assert accepted is True and transformed == {"ok": True}


@pytest.mark.linux_only  # POSIX shebang fixture; the Windows half is the wine2e receipt
def test_silent_nonzero_exit_is_logged_as_warning(tmp_path, monkeypatch, caplog):
    """rc!=0 with no stdout AND no stderr is the interpreter-never-ran signature: WARNING, not INFO."""
    fake = _fake_bash(tmp_path, "exit 1\n")
    monkeypatch.setattr(local, "_find_bash", lambda: str(fake))
    filt = _filter_script("")

    with caplog.at_level(logging.INFO, logger="gateway.platforms.webhook_filters"):
        accepted, _ = WebhookRouteProcessor().run_route_script(str(filt), {})

    assert accepted is False
    silent = [r for r in caplog.records if "script ignored webhook path=filter.sh" in r.getMessage()]
    assert silent and silent[0].levelno == logging.WARNING
