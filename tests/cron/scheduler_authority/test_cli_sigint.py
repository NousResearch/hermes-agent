"""A manual ``hermes cron run`` must not free a gateway-admitted run on Ctrl-C.

Once a separate gateway accepted the run, a SIGINT in the CLI raised KeyboardInterrupt out of
``run_canonical_job``: the CLI booked the execution failed and cleared the fire claim while the gateway
kept running the job, so a second ``hermes cron run`` could start the same job again. Ctrl-C is now an
operator cancel: the claim stays until the gateway reports the run terminal (``cancelled``).
"""
import json
import os
import re
import signal
import subprocess
import threading

import pytest

from tests.e2e.core.parity._helpers import ParityHome, hermes_argv
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
from tests.waiting import wait_until


def _job(hh, job_id):
    return next(j for j in json.loads((hh / 'cron/jobs.json').read_text())['jobs'] if j['id'] == job_id)


def _stopped(pid):
    with open(f'/proc/{pid}/status') as status:
        return status.read().split('State:')[1].lstrip().startswith('T')


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('action', ['interrupt', 'cancel'])
def test_manual_gateway_fire_keeps_claim_until_cancelled(tmp_path, action):
    home = tmp_path / 'home'
    hh = home / '.hermes'
    hh.mkdir(parents=True)
    entered = threading.Event()
    release = threading.Event()

    def respond(_record):
        entered.set()
        release.wait(90)
        return Text('finished')

    with FakeLLMServer(respond) as server:
        write_hermes_home(hh, server.base_url)
        ph = ParityHome(tmp_path, home, hh, home, tmp_path, tmp_path / 'pids', 'cron-interrupt')
        env = ph.env({'HERMES_GATEWAY_LOCK_DIR': str(tmp_path / 'locks'),
                      'XDG_STATE_HOME': str(tmp_path / 'state')})

        def cli(*args):
            return subprocess.run(hermes_argv('cron', *args), cwd=home, env=env,
                                  text=True, capture_output=True, timeout=90, check=False)
        children = []
        stopped_gateway = None
        try:
            created = cli('create', '1d', 'test', '--name', 'signal-test', '--deliver', 'local')
            match = re.search(r'Created job: (\S+)', created.stdout)
            assert created.returncode == 0 and match, (created.stdout, created.stderr)
            job_id = match.group(1)
            child = subprocess.Popen(hermes_argv('cron', 'run', job_id), cwd=home, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            children.append(child)
            assert entered.wait(100), 'gateway never reached the provider'
            if action == 'interrupt':
                # Freeze only this fixture's gateway so it cannot finish during the second CLI:
                # the overlap check then does not depend on how fast the cancel reaches it.
                state = json.loads((hh / 'gateway_state.json').read_text())
                assert state['hermes_home'] == str(hh)
                pid = int(state['pid'])
                with open(f'/proc/{pid}/environ', 'rb') as environ:
                    assert f'PARITY_TREE_TAG={ph.tag}'.encode() in environ.read().split(b'\0')
                os.kill(pid, signal.SIGSTOP)
                stopped_gateway = pid
                wait_until(lambda: _stopped(pid), timeout=10, what='fixture gateway to stop')
                child.send_signal(signal.SIGINT)
                second = subprocess.Popen(hermes_argv('cron', 'run', job_id), cwd=home, env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                children.append(second)
                wait_until(lambda: second.poll() is not None or child.poll() is not None,
                           timeout=30, what='second run refusal or premature CLI exit')
                assert child.poll() is None, 'CLI abandoned the still-admitted gateway run'
                assert second.poll() is not None, 'second CLI started another gateway run'
                out, err = second.communicate()
                assert len(server.main_requests()) == 1, (out, err)
                assert _job(hh, job_id).get('fire_claim'), 'admitted run lost its claim'
                os.kill(pid, signal.SIGCONT)
                stopped_gateway = None
            else:
                cancelled = cli('cancel', job_id)
                assert 'Cancel requested' in cancelled.stdout, cancelled.stdout
            out, err = child.communicate(timeout=45)
            assert 'Ran now: cancelled' in out, (out, err)
            job = _job(hh, job_id)
            assert job['last_status'] == 'cancelled'
            assert not job.get('fire_claim')
            assert len(server.main_requests()) == 1
        finally:
            if stopped_gateway is not None:
                os.kill(stopped_gateway, signal.SIGCONT)
            release.set()
            for proc in children:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.communicate(timeout=5)
            subprocess.run(hermes_argv('gateway', 'stop'), cwd=home, env=env,
                           capture_output=True, timeout=45, check=False)
