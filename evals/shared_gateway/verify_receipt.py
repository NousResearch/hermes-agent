"""Check measured shared-gateway receipts against the full required case set.

This validates coverage and evidence availability, not the truth of observations.
Native topology and trace review remain separate acceptance requirements.
"""
import argparse
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(receipt, manifest, *, evidence_root, expected_sha, expected_os):
    require(receipt.get('schema') == 1 and manifest.get('schema') == 1, 'unsupported schema')
    require(receipt.get('source_sha') == expected_sha, 'stale source SHA')
    require(receipt.get('native_os') == expected_os, 'wrong native OS')
    require(receipt.get('fixture_home_isolated') is True, 'fixture home not isolated')
    require(receipt.get('production_dispatch') is True, 'production dispatch not exercised')
    require(receipt.get('model_boundary') in ('loopback', 'live-provider'), 'unknown model boundary')
    require(receipt.get('remaining_owned_processes') == [], 'fixture leaked processes')
    required = [case['name'] for case in manifest['cases']]
    observed = receipt.get('cases', [])
    names = [case['name'] for case in observed]
    require(required and len(required) == len(set(required)), 'empty or duplicate manifest')
    require(len(names) == len(set(names)), 'duplicate scenario receipts')
    require(set(names) == set(required),
            f'case mismatch: missing={sorted(set(required) - set(names))}, extra={sorted(set(names) - set(required))}')
    root = Path(evidence_root).resolve()
    for case in observed:
        name = case['name']
        require(case.get('status') == 'passed', f'{name}: not passed')
        assertions = case.get('assertions')
        require(type(assertions) is int and assertions > 0, f'{name}: no assertions')
        paths = case.get('evidence_paths')
        require(isinstance(paths, list) and paths, f'{name}: no evidence')
        for relative in paths:
            path = (root / relative).resolve()
            require(path.is_relative_to(root) and path.is_file() and path.stat().st_size > 0,
                    f'{name}: missing, empty or external evidence: {relative}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('cases.json'))
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--expected-os', choices=('linux', 'macos', 'windows'), required=True)
    args = parser.parse_args()
    try:
        verify(json.loads(args.receipt.read_text(encoding='utf-8')),
               json.loads(args.manifest.read_text(encoding='utf-8')),
               evidence_root=args.receipt.parent, expected_sha=args.expected_sha,
               expected_os=args.expected_os)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'FAIL: {exc}\n')
    print('PASS: shared gateway receipt coverage (native topology/trace review still required)')


if __name__ == '__main__':
    main()
