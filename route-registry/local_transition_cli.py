#!/usr/bin/env python3
"""Governed, local-only migration; no service lifecycle actions."""
import argparse
import json
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent / 'route_registry'))
from local_transition import (LocalTransitionError, build_local_plan,
                              apply_local_plan, rollback_local_plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'apply', 'rollback'))
    parser.add_argument('--target-root', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--backup-root')
    parser.add_argument('--registry', default=str(Path(__file__).parent / 'registry/route-slots.yaml'))
    args = parser.parse_args(argv)
    try:
        root = Path(args.target_root).resolve()
        spec = (yaml.safe_load(Path(args.registry).read_text(encoding='utf-8')) or {})['local_transition']
        if args.mode == 'plan':
            plan = build_local_plan(root, spec)
            with Path(args.plan).open('x', encoding='utf-8') as handle:
                handle.write(json.dumps(plan, indent=2) + '\n')
            return 0
        if not args.backup_root:
            raise LocalTransitionError('--backup-root is required')
        plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
        if Path(plan['target_root']).resolve() != root:
            raise LocalTransitionError('plan root does not match --target-root')
        backup = Path(args.backup_root).absolute()
        if args.mode == 'apply':
            apply_local_plan(plan, spec, backup)
        else:
            rollback_local_plan(plan, backup)
        return 0
    except (LocalTransitionError, KeyError, OSError, TypeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
