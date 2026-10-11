"""Windows handoff regression; real PowerShell, harmless fixture launcher only."""
import json
import os
from pathlib import Path
import subprocess

import pytest
import psutil

from tests.installation_launcher_fixture import publish_fixture_launcher
from tests.scripts.desktop_update.test_desktop_update_windows_python_handoff import CLI, ROOT


def run_handoff(tmp_path, *, warning='', no_gateway=False, state_sha='a' * 40,
                heartbeat_offset=0, correlation_matches=True, state_missing=False,
                recovery=False, sibling=False, identity='matching', gateway_state='running',
                culture='pt-BR', pending_pid=False, served=None, named=False, excluded=None,
                restart_profiles=(), newer_archive=None, planned_failure=False):
    source = CLI.replace("    print('Desktop build failed')", """
    if sys.argv[1] == 'update' or (sys.argv[1:3] == ['gateway', 'start'] and os.environ['FIXTURE_RECOVERY'] == '1'):
        from datetime import datetime, timezone, timedelta
        home = Path(os.environ['HERMES_HOME'])
        state = {'pid': int(os.environ['FIXTURE_PID']), 'gateway_state': os.environ['FIXTURE_GATEWAY_STATE'],
                 'kind': 'hermes-gateway', 'hermes_home': str(home) if os.environ['FIXTURE_IDENTITY'] != 'home' else str(home / 'other'),
                 'start_time': int(os.environ['FIXTURE_START_TIME']) + (10000 if os.environ['FIXTURE_IDENTITY'] == 'start' else 0),
                 'code_sha': os.environ['FIXTURE_STATE_SHA'],
                 'updated_at': (datetime.now(timezone.utc) + timedelta(seconds=int(os.environ['FIXTURE_HEARTBEAT_OFFSET']))).isoformat(),
                 'served_profiles': json.loads(os.environ['FIXTURE_SERVED'])}
        if os.environ['FIXTURE_MISSING'] != '1' or sys.argv[1] != 'update':
            (home / 'gateway_state.json').write_text(json.dumps(state))
        receipts = Path(os.environ['FIXTURE_ROOT']) / 'logs/update_receipts'; receipts.mkdir(parents=True, exist_ok=True)
        (receipts / 'update_fixture.json').write_text(json.dumps({'outcome': 'success',
            'post_update': {'sha': 'a' * 40}, 'correlation_id': (os.environ.get('HERMES_UPDATE_CORRELATION_ID')
                if os.environ['FIXTURE_CORRELATION'] == '1' else 'different-run'),
            'plan': {'runtimes': [{'kind': 'gateway', 'profile': 'excluded'}] if os.environ['FIXTURE_PLANNED_FAILURE'] == '1' else []},
            'runtime_outcomes': [{'kind': 'gateway', 'profile': 'excluded', 'outcome': 'unaccounted'}] if os.environ['FIXTURE_PLANNED_FAILURE'] == '1' else [],
            'gateway_restart': {'incomplete': os.environ['FIXTURE_PLANNED_FAILURE'] == '1',
                                'fresh_recovery': {'requested': json.loads(os.environ['FIXTURE_REQUESTED'])}},
            'finished_at': datetime.now(timezone.utc).isoformat()}))
        if os.environ['FIXTURE_NEWER']:
            newer = receipts / 'update_newer.json'
            newer.write_text('{' if os.environ['FIXTURE_NEWER'] == 'malformed' else json.dumps({
                'outcome': 'success', 'correlation_id': 'other-run',
                'finished_at': datetime.now(timezone.utc).isoformat(), 'post_update': {'sha': 'b' * 40}}))
            newer_time = (receipts / 'update_fixture.json').stat().st_mtime + 10
            os.utime(newer, (newer_time, newer_time))
        print(os.environ['FIXTURE_WARNING'])
""")
    install = tmp_path / 'checkout with spaces'
    publish_fixture_launcher(install, source)
    # Import the unchanged production helper/topology modules normally, while
    # keeping only the side-effecting CLI as a harmless stand-in.
    (install / 'hermes_cli/__init__.py').write_text(
        f'__path__.append({str(ROOT / "hermes_cli")!r})\n', encoding='utf-8')
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n', encoding='utf-8')
    root_home = tmp_path / 'home'; root_home.mkdir()
    (root_home / 'config.yaml').write_text('{}\n', encoding='utf-8')
    home = root_home / 'profiles/launch' if named else root_home
    if named:
        home.mkdir(parents=True)
        (home / 'config.yaml').write_text('gateway:\n  standalone: true\n', encoding='utf-8')
    if pending_pid:
        (home / 'gateway.pid').write_text(json.dumps({'pid': os.getpid()}), encoding='utf-8')
    if sibling:
        sibling_home = home / 'profiles/sibling'; sibling_home.mkdir(parents=True)
        (sibling_home / 'config.yaml').write_text('{}\n', encoding='utf-8')
    if excluded:
        excluded_home = root_home / 'profiles/excluded'; excluded_home.mkdir(parents=True)
        (excluded_home / 'config.yaml').write_text(
            'gateway:\n  standalone: true\n' if excluded == 'standalone' else '{}\n', encoding='utf-8')
        if excluded == 'parked':
            (excluded_home / 'gateway.parked').touch()
    calls = tmp_path / 'calls.jsonl'
    script = str(ROOT / 'scripts/desktop-update/windows.ps1').replace("'", "''")
    install_arg = str(install).replace("'", "''")
    command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command',
        f"[Threading.Thread]::CurrentThread.CurrentCulture = [cultureinfo]::GetCultureInfo('{culture}'); "
        f"& '{script}' -InstallRoot '{install_arg}' -NoUi" + (' -NoGateway' if no_gateway else '')]
    result = subprocess.run(command, cwd=tmp_path, env={**os.environ,
        'TEMP': str(tmp_path), 'TMP': str(tmp_path), 'HERMES_HOME': str(home),
        'PYTHONPATH': str(ROOT), 'FIXTURE_ROOT': str(root_home),
        'FIXTURE_PLANNED_FAILURE': str(int(planned_failure)),
        'FIXTURE_REQUESTED': json.dumps(list(restart_profiles)), 'FIXTURE_NEWER': newer_archive or '',
        'HERMES_RUNTIME_DIR': str(tmp_path / 'runtime'), 'HANDOFF_CALLS': str(calls),
        'HANDOFF_EXIT': '0', 'GATEWAY_EXIT': '0' if recovery else '1', 'FIXTURE_PID': str(os.getpid()),
        'FIXTURE_STATE_SHA': state_sha, 'FIXTURE_WARNING': warning,
        'FIXTURE_CORRELATION': str(int(correlation_matches)),
        'FIXTURE_MISSING': str(int(state_missing)), 'FIXTURE_RECOVERY': str(int(recovery)),
        'FIXTURE_HEARTBEAT_OFFSET': str(heartbeat_offset), 'FIXTURE_IDENTITY': identity,
        'FIXTURE_GATEWAY_STATE': gateway_state,
        'FIXTURE_SERVED': json.dumps(served if served is not None else (['default', 'sibling'] if sibling == 'covered' else ['default'])),
        'FIXTURE_START_TIME': str(round(psutil.Process().create_time() * 100))},
        capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return ([json.loads(line)['argv'] for line in calls.read_text().splitlines()],
            json.loads((home / '.hermes-update-result.json').read_text(encoding='utf-8-sig')))


@pytest.mark.platforms('windows')
def test_planned_failed_sibling_keeps_gateway_warning(tmp_path):
    calls, receipt = run_handoff(tmp_path, excluded='standalone', planned_failure=True,
        warning="Update follow-up 'gateway_restart' did not finish: sibling failed")
    assert [call[0] for call in calls] == ['update']
    assert receipt['manual']
    assert 'gateway_restart' in receipt['message']


@pytest.mark.platforms('windows')
def test_updated_live_fleet_is_not_started_twice(tmp_path):
    calls, receipt = run_handoff(tmp_path)
    assert [call[0] for call in calls] == ['update']
    assert not receipt.get('manual', False)


@pytest.mark.platforms('windows')
def test_current_fleet_clears_only_gateway_warning(tmp_path):
    calls, receipt = run_handoff(tmp_path, warning="Update follow-up 'gateway_restart' did not finish: early probe")
    assert [call[0] for call in calls] == ['update']
    assert not receipt.get('manual', False)


@pytest.mark.platforms('windows')
def test_unrelated_receipt_does_not_clear_gateway_warning(tmp_path):
    calls, receipt = run_handoff(tmp_path, correlation_matches=False,
        warning="Update follow-up 'gateway_restart' did not finish: early probe")
    assert receipt.get('manual', False)
    # Inconclusive evidence never authorizes a broad process sweep.
    assert [call[0] for call in calls] == ['update']


@pytest.mark.platforms('windows')
def test_missing_gateway_uses_scoped_start_and_rechecks_warning(tmp_path):
    calls, receipt = run_handoff(tmp_path, state_missing=True, recovery=True,
        warning="Update follow-up 'gateway_restart' did not finish: early probe")
    assert calls[1] == ['gateway', 'start']
    assert not receipt.get('manual', False)


@pytest.mark.platforms('windows')
def test_future_heartbeat_is_not_readiness(tmp_path):
    calls, receipt = run_handoff(tmp_path, heartbeat_offset=3600)
    assert receipt.get('manual', False)
    assert [call[0] for call in calls] == ['update']


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('identity', ['home', 'start'])
def test_foreign_home_or_reused_pid_does_not_discharge_warning(tmp_path, identity):
    calls, receipt = run_handoff(tmp_path, identity=identity)
    assert receipt.get('manual', False)
    assert [call[0] for call in calls] == ['update']


@pytest.mark.platforms('windows')
def test_gateway_reconciliation_preserves_unrelated_followup(tmp_path):
    calls, receipt = run_handoff(tmp_path, warning="Update follow-up 'gateway_restart' did not finish: early probe\nUpdate follow-up 'skills_sync' did not finish: real error")
    assert [call[0] for call in calls] == ['update']
    assert receipt['manual']
    assert 'skills_sync' in receipt['message']
    assert 'gateway_restart' not in receipt['message']


@pytest.mark.platforms('windows')
def test_remote_handoff_is_passive_and_keeps_unverified_warning(tmp_path):
    calls, receipt = run_handoff(tmp_path, no_gateway=True,
        warning="Update follow-up 'gateway_restart' did not finish: early probe")
    assert [call[0] for call in calls] == ['update']
    assert '--gateway' not in calls[0]
    assert receipt['manual']


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('kwargs', [
    {'state_sha': 'b' * 40}, {'state_sha': ''}, {'heartbeat_offset': -3600},
    {'gateway_state': 'startup_failed'}, {'gateway_state': 'unknown'},
])
def test_unready_existing_gateway_fails_closed_without_start(tmp_path, kwargs):
    calls, receipt = run_handoff(tmp_path, **kwargs)
    assert [call[0] for call in calls] == ['update']
    assert receipt['manual']


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('sibling', ['covered', 'missing'])
def test_sibling_fleet_is_never_swept_for_incomplete_coverage(tmp_path, sibling):
    calls, receipt = run_handoff(tmp_path, sibling=sibling)
    assert [call[0] for call in calls] == ['update']
    assert receipt['manual'] == (sibling == 'missing')


@pytest.mark.platforms('windows')
def test_failed_scoped_recovery_keeps_gateway_warning(tmp_path):
    calls, receipt = run_handoff(tmp_path, state_missing=True,
        warning="Update follow-up 'gateway_restart' did not finish: early probe")
    assert calls[1] == ['gateway', 'start']
    assert receipt['manual']
    assert 'gateway_restart' in receipt['message']


@pytest.mark.platforms('windows')
def test_missing_state_with_pid_owner_does_not_start_competing_gateway(tmp_path):
    calls, receipt = run_handoff(tmp_path, state_missing=True, pending_pid=True)
    assert [call[0] for call in calls] == ['update']
    assert receipt['manual']


@pytest.mark.platforms('windows')
def test_single_profile_standalone_empty_served_record_is_current(tmp_path):
    calls, receipt = run_handoff(tmp_path, served=[], heartbeat_offset=-75)
    assert [call[0] for call in calls] == ['update']
    assert not receipt.get('manual', False)


@pytest.mark.platforms('windows')
def test_named_standalone_reads_update_archive_from_root(tmp_path):
    calls, receipt = run_handoff(tmp_path, named=True, served=[],
        warning="Update follow-up 'gateway_restart' did not finish: early probe")
    assert [call[0] for call in calls] == ['update']
    assert not receipt.get('manual', False)


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('excluded', ['parked', 'standalone'])
def test_host_coverage_excludes_profiles_not_served_by_multiplexer(tmp_path, excluded):
    calls, receipt = run_handoff(tmp_path, excluded=excluded)
    assert [call[0] for call in calls] == ['update']
    assert not receipt.get('manual', False)


@pytest.mark.platforms('windows')
def test_host_does_not_discharge_update_owned_standalone_sibling(tmp_path):
    calls, receipt = run_handoff(tmp_path, excluded='standalone', restart_profiles=['excluded'],
        warning="Update follow-up 'gateway_restart' did not finish: sibling owed")
    assert [call[0] for call in calls] == ['update']
    assert receipt['manual']
    assert 'gateway_restart' in receipt['message']


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('newer_archive', ['unrelated', 'malformed'])
def test_newer_archive_cannot_shadow_matching_finalized_root_receipt(tmp_path, newer_archive):
    calls, receipt = run_handoff(tmp_path, named=True, served=[], newer_archive=newer_archive,
        warning="Update follow-up 'gateway_restart' did not finish: early probe")
    assert [call[0] for call in calls] == ['update']
    assert not receipt.get('manual', False)

