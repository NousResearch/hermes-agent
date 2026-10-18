"""Normal-module read-only probes on isolated homes using real topology helpers."""
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def probe(tmp_path, *, named=False, served=(), requested=(), standalone_sibling=False,
          planned=(), runtime_outcomes=(), incomplete=False):
    root = tmp_path / 'home'
    root.mkdir()
    (root / 'config.yaml').write_text('{}\n', encoding='utf-8')
    home = root / 'profiles/launch' if named else root
    if named:
        home.mkdir(parents=True)
        (home / 'config.yaml').write_text('gateway:\n  standalone: true\n', encoding='utf-8')
    if standalone_sibling:
        sibling = root / 'profiles/excluded'
        sibling.mkdir(parents=True)
        (sibling / 'config.yaml').write_text('gateway:\n  standalone: true\n', encoding='utf-8')
    (home / 'gateway_state.json').write_text(json.dumps({'served_profiles': list(served)}), encoding='utf-8')
    archives = root / 'logs/update_receipts'
    archives.mkdir(parents=True)
    now = datetime.now(UTC)
    (archives / 'update_current.json').write_text(json.dumps({
        'correlation_id': 'current-run', 'finished_at': now.isoformat(),
        'outcome': 'partial', 'post_update': {'sha': 'a' * 40},
        'plan': {'runtimes': list(planned)}, 'runtime_outcomes': list(runtime_outcomes),
        'gateway_restart': {'incomplete': incomplete,
                            'fresh_recovery': {'requested': list(requested)}}}), encoding='utf-8')
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = subprocess.run([sys.executable, '-m', 'hermes_cli.desktop_gateway_reconcile',
        '--correlation', 'current-run', '--started-at', str(now.timestamp())],
        cwd=ROOT, env={**os.environ, 'HERMES_HOME': str(home),
            'HERMES_RUNTIME_DIR': str(tmp_path / 'runtime'), 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    return json.loads(result.stdout)


@pytest.mark.parametrize('named', [False, True])
def test_normal_module_reads_finalized_root_receipt_without_writes(tmp_path, named):
    result = probe(tmp_path, named=named)
    assert result == {'expected_sha': 'a' * 40, 'missing_profiles': []}


def test_host_cannot_claim_standalone_sibling_via_served_names(tmp_path):
    result = probe(tmp_path, served=['default', 'excluded'], requested=['excluded'], standalone_sibling=True)
    assert result['missing_profiles'] == ['excluded']


def test_standalone_launch_cannot_claim_host_served_record(tmp_path):
    result = probe(tmp_path, named=True, served=['default'], requested=['default'])
    assert result['missing_profiles']


@pytest.mark.parametrize('outcome', ['aborted', 'unaccounted'])
def test_failed_planned_sibling_survives_empty_restart_inventory(tmp_path, outcome):
    runtime = {'kind': 'gateway', 'profile': 'excluded', 'pid': 12345}
    result = probe(tmp_path, served=['default'], standalone_sibling=True,
                   planned=[runtime], incomplete=True,
                   runtime_outcomes=[{**runtime, 'outcome': outcome}])
    assert 'excluded' in result['missing_profiles']


def test_planned_host_can_be_verified_after_incomplete_probe(tmp_path):
    runtime = {'kind': 'gateway', 'profile': 'default', 'pid': 12345}
    result = probe(tmp_path, planned=[runtime], incomplete=True,
                   runtime_outcomes=[{**runtime, 'outcome': 'unaccounted'}])
    assert result['missing_profiles'] == []

