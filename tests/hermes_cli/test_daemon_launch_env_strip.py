"""One launch's hook consent and Kanban/terminal pins never reach the auto-started shared daemon."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.waiting import wait_until

# Every launch-scoped value a client process may hold when it auto-starts the shared daemon
# (SWEEP_daemon-launch-env.md): consent, Kanban worker pins, the launching turn's session identity,
# human-presence and one-shot markers, a bot-to-bot turn author, a terminal's systemd markers, and the
# launchd/external-supervisor markers every process below a service-managed gateway inherits.
SUPERVISOR_LEAKS = {'XPC_SERVICE_NAME': 'ai.hermes.gateway', 'LAUNCHD_SOCKET': '/tmp/launchd-x/Listeners',
                    'HERMES_LAUNCHD_LABEL': 'ai.hermes.gateway', 'HERMES_GATEWAY_EXTERNAL_SUPERVISOR': '1',
                    'HERMES_SUPERVISED_CHILD': '1'}
LEAKS = {**SUPERVISOR_LEAKS, 'HERMES_ACCEPT_HOOKS': '1', 'HERMES_KANBAN_TASK': 't_1', 'HERMES_KANBAN_BOARD': 'other',
         'HERMES_KANBAN_DB': '/x/kanban.db', 'HERMES_SESSION_SOURCE': 'kanban', 'TERMINAL_CWD': '/x',
         'HERMES_TENANT': 'acme', 'HERMES_SESSION_SOURCE_EXPLICIT': '1', 'HERMES_TURN_AUTHOR': '{"handle":"x"}',
         'HERMES_INTERACTIVE': '1', 'HERMES_GATEWAY_SESSION': '1', 'HERMES_SINGLE_QUERY_SESSION': '1',
         'HERMES_SESSION_ID': 'launching-turn', 'HERMES_SESSION_KEY': 'agent:main:tui:x',
         'HERMES_SESSION_PLATFORM': 'tui', 'HERMES_SESSION_CHAT_ID': 'c1', 'HERMES_CRON_SESSION': '1',
         # A terminal running under a systemd service: not the daemon's own supervisor.
         'INVOCATION_ID': 'terminal-scope', 'JOURNAL_STREAM': '8:1',
         # A process below a Type=notify gateway: the unit's notify socket and watchdog are its main PID's.
         'NOTIFY_SOCKET': '/run/user/1000/systemd/notify', 'WATCHDOG_USEC': '30000000', 'WATCHDOG_PID': '4242'}


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike  # stub interpreter records env then exits; reaped below
def test_daemon_does_not_inherit_hook_consent_or_worker_pins(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_start as start
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    witness = home / 'env.json'
    executable = tmp_path / 'owned-interpreter'
    executable.write_text(
        '#!' + sys.executable + '\nimport json, os\n'
        + f'open({str(witness)!r}, "w").write(json.dumps('
        + '{k: os.environ.get(k) for k in ' + repr([*LEAKS, 'HERMES_INFERENCE_MODEL']) + '}))\n',
        encoding='utf-8')
    executable.chmod(0o700)
    monkeypatch.setattr(sys, 'executable', str(executable))
    for key, value in {**LEAKS, 'HERMES_INFERENCE_MODEL': 'fallback'}.items():
        monkeypatch.setenv(key, value)
    child = start.spawn_unmanaged_gateway(home, deadline=time.monotonic() + 5)
    try:
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    seen = json.loads(Path(witness).read_text())
    assert seen == {**dict.fromkeys(LEAKS), 'HERMES_INFERENCE_MODEL': 'fallback'}
    assert os.environ['HERMES_ACCEPT_HOOKS'] == '1'  # the launching process keeps its own policy


@pytest.mark.platforms("macos")
@pytest.mark.spawns_gateway_lookalike  # stub interpreter records its supervisor then exits; reaped below
def test_daemon_started_below_a_launchd_gateway_is_unsupervised(tmp_path, monkeypatch):
    # An agent of a launchd gateway auto-starts a daemon for another home: that daemon has no supervisor,
    # so it must declare "manual" (which arms --idle-exit) instead of claiming its launcher's launchd job.
    from hermes_cli import gateway_runtime_start as start
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    witness = home / 'supervisor.json'
    repo = Path(__file__).resolve().parents[2]
    executable = tmp_path / 'owned-interpreter'
    executable.write_text(
        '#!' + sys.executable + '\nimport json, sys\n'
        + f'sys.path.insert(0, {str(repo)!r})\n'
        + 'from gateway.control_socket import _detect_supervisor\n'
        + 'from gateway.restart import is_supervised_gateway_launch\n'
        + f'open({str(witness)!r}, "w").write(json.dumps('
        + '[_detect_supervisor(), is_supervised_gateway_launch()]))\n',
        encoding='utf-8')
    executable.chmod(0o700)
    monkeypatch.setattr(sys, 'executable', str(executable))
    for key, value in SUPERVISOR_LEAKS.items():
        monkeypatch.setenv(key, value)
    for key in ('INVOCATION_ID', 'HERMES_DESKTOP_MANAGED', 'HERMES_S6_SUPERVISED_CHILD'):
        monkeypatch.delenv(key, raising=False)
    child = start.spawn_unmanaged_gateway(home, deadline=time.monotonic() + 5, idle_exit=True)
    try:
        # With a GUI session launchd takes the job (a GatewayJob: poll()/pid only), else a plain child.
        wait_until(lambda: witness.exists() and witness.read_text(), timeout=20, what='supervisor witness')
    finally:
        if isinstance(child, subprocess.Popen):
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        else:
            subprocess.run(['launchctl', 'remove', child.target.rsplit('/', 1)[-1]], check=False,
                           capture_output=True, stdin=subprocess.DEVNULL)
    assert json.loads(Path(witness).read_text()) == ['manual', False]
