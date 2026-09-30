"""Webhook filter scripts run under the Git-Bash-safe interpreter and log silent failures."""
import logging
import stat
import subprocess
import types

import pytest

import gateway.platforms.webhook_filters as webhook_filters
from gateway.platforms.webhook_filters import WebhookRouteProcessor
from tools.environments import local


def _filter_script(body: str, name: str = "filter.sh"):
    from hermes_constants import get_hermes_home
    scripts = get_hermes_home() / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    filt = scripts / name
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

    accepted, transformed = WebhookRouteProcessor().run_route_script(str(filt), {"a": 1})

    assert marker.exists()
    assert accepted is True and transformed == {"ok": True}


@pytest.mark.platforms("linux")  # POSIX shebang fixture; the Windows half is the wine2e receipt
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


@pytest.mark.platforms("posix")  # POSIX store layout (bin/python3); the Windows half is the wine2e receipt
def test_python_filter_spawns_through_cron_resolver_argv(tmp_path, monkeypatch):
    """On a store install a .py route script must spawn via cron's ``_script_argv`` (#129100):
    the dependency venv's interpreter plus the repo bootstrap — never the bare store
    ``sys.executable``, whose PYTHONPATH ``build_subprocess_env`` strips."""
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)  # conftest seeds it session-wide
    store = tmp_path / "store" / "bin" / "python3"
    venv_python = tmp_path / "venv" / "bin" / "python3"
    for target in (store, venv_python):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")  # never executed; the spawn is captured below
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda _root: store)
    monkeypatch.setattr("pm.environments.project_python", lambda _root: venv_python)

    captured = {}

    def spy_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["env"] = kwargs["env"]

        class _Result:
            returncode = 0
            stdout = '{"ok": true}'
            stderr = ""

        return _Result()

    monkeypatch.setattr(
        webhook_filters, "subprocess",
        types.SimpleNamespace(run=spy_run, TimeoutExpired=subprocess.TimeoutExpired))

    filt = _filter_script("pass", name="filter.py")
    accepted, transformed = WebhookRouteProcessor().run_route_script(str(filt), {"a": 1})

    assert accepted is True and transformed == {"ok": True}
    assert captured["argv"][0] == str(venv_python)
    assert "-c" in captured["argv"]  # the repo bootstrap from _posix_cron_script_argv
    assert captured["env"].get("HERMES_DISABLE_LAZY_INSTALLS") == "1"


@pytest.mark.platforms("posix")  # POSIX interpreter spawn; the Windows half is the wine2e receipt
def test_python_filter_runs_with_lazy_installs_disabled(tmp_path, monkeypatch):
    """End to end on the non-store path: a plain .py route script still runs, reads its payload
    from stdin, and gets HERMES_DISABLE_LAZY_INSTALLS in its environment on every OS (#129100)."""
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)  # conftest seeds it session-wide
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda _root: None)
    body = (
        "import json, os, sys\n"
        "payload = json.load(sys.stdin)\n"
        'print(json.dumps({"lazy": os.environ.get("HERMES_DISABLE_LAZY_INSTALLS"), "a": payload["a"]}))\n'
    )
    filt = _filter_script(body, name="filter.py")

    accepted, transformed = WebhookRouteProcessor().run_route_script(str(filt), {"a": 1})

    assert accepted is True
    assert transformed == {"lazy": "1", "a": 1}


@pytest.mark.platforms("posix")  # POSIX store layout; the Windows half is the wine2e receipt
def test_python_filter_vetoes_when_dependency_env_missing(tmp_path, monkeypatch, caplog):
    """Store install whose dependency venv disappeared: cron's resolver raises and the route is
    vetoed with a warning instead of silently running the bare store Python (#129100)."""
    store = tmp_path / "store" / "bin" / "python3"
    store.parent.mkdir(parents=True)
    store.write_text("", encoding="utf-8")
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda _root: store)
    missing = tmp_path / "venv" / "bin" / "python3"  # deliberately left absent
    monkeypatch.setattr("pm.environments.project_python", lambda _root: missing)

    filt = _filter_script("print('not json')", name="filter.py")
    with caplog.at_level(logging.WARNING, logger="gateway.platforms.webhook_filters"):
        accepted, transformed = WebhookRouteProcessor().run_route_script(str(filt), {})

    assert accepted is False and transformed is None
    assert any("script ignored webhook" in r.getMessage() for r in caplog.records)
