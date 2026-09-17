"""Acceptance reports must cover the required manifest, not only successes."""
import importlib.util
from pathlib import Path

import pytest


def verifier():
    path = Path(__file__).resolve().parents[2] / 'evals/shared_gateway/verify_receipt.py'
    assert path.is_file(), 'shared gateway acceptance verifier missing'
    spec = importlib.util.spec_from_file_location('shared_gateway_verifier', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify


def fixture(tmp_path):
    evidence = tmp_path / 'observed.log'
    evidence.write_text('fixture evidence; not a production acceptance receipt')
    manifest = {'schema': 1, 'cases': [{'name': 'one'}, {'name': 'two'}]}
    report = {'schema': 1, 'source_sha': 'a' * 40, 'native_os': 'linux',
              'fixture_home_isolated': True, 'production_dispatch': True,
              'model_boundary': 'loopback', 'remaining_owned_processes': [],
              'cases': [{'name': name, 'status': 'passed', 'assertions': 1,
                         'evidence_paths': ['observed.log']} for name in ('one', 'two')]}
    return manifest, report


def test_complete_receipt_requires_exact_manifest_and_existing_evidence(tmp_path):
    verify = verifier()
    manifest, report = fixture(tmp_path)
    verify(report, manifest, evidence_root=tmp_path, expected_sha='a' * 40, expected_os='linux')
    (tmp_path / 'observed.log').unlink()
    with pytest.raises(ValueError, match='evidence'):
        verify(report, manifest, evidence_root=tmp_path, expected_sha='a' * 40, expected_os='linux')


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'extra', 'skipped', 'zero',
                                  'stale', 'wrong-os', 'leaked', 'no-dispatch', 'no-evidence'])
def test_incomplete_receipts_cannot_pass(tmp_path, fault):
    verify = verifier()
    manifest, report = fixture(tmp_path)
    mutations = {
        'missing': lambda: report['cases'].pop(),
        'duplicate': lambda: report['cases'].append(report['cases'][0].copy()),
        'extra': lambda: report['cases'][0].update(name='unrequired'),
        'skipped': lambda: report['cases'][0].update(status='skipped'),
        'zero': lambda: report['cases'][0].update(assertions=0),
        'stale': lambda: report.update(source_sha='b' * 40),
        'wrong-os': lambda: report.update(native_os='windows'),
        'leaked': lambda: report.update(remaining_owned_processes=[123]),
        'no-dispatch': lambda: report.update(production_dispatch=False),
        'no-evidence': lambda: report['cases'][0].update(evidence_paths=[]),
    }
    mutations[fault]()
    with pytest.raises(ValueError):
        verify(report, manifest, evidence_root=tmp_path, expected_sha='a' * 40, expected_os='linux')
