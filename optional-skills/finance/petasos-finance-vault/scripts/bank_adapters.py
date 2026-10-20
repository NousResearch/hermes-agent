#!/usr/bin/env python3
"""Petasos bank adapters — generic engine CLI (v2 thin wrapper).

All intelligence lives in banks.yaml (data) and the petasos package; this
script only exposes the three inspection tools (INV-3):

  --test                          run fixtures/ against fixtures/expected.yaml
  --probe <email.txt>             per-field MATCH/MISS report for one raw email
  --banks                         active registry + formats (+ overlap warnings)

A parse regex that never fires is invisible in --dry runs; --probe is the
cheap catch (a MISS line IS the dead regex), and --test pins the frozen
regression corpus (fixtures are byte-sealed, brief §4).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from petasos.knob import load_knobs          # noqa: E402
from petasos.parse import parse_any, parse_format, read_email  # noqa: E402
from petasos.registry import (RegistryError, fixtures_dir, iter_formats,  # noqa: E402
                              resolve_registry, validate_registry)


def _registry_or_die():
    try:
        knobs = load_knobs(os.environ)
        return resolve_registry(knobs)
    except RegistryError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)


def run_tests():
    reg = _registry_or_die()
    fdir = fixtures_dir(reg)
    if not fdir:
        print('no fixtures/ directory found', file=sys.stderr)
        return 1
    print(f'registry: {reg.path} ({reg.source})\nfixtures: {fdir}\n')
    try:
        import yaml
    except ImportError:
        print('PyYAML is required for banks.yaml — install with: pip install pyyaml',
              file=sys.stderr)
        return 2
    with open(os.path.join(fdir, 'expected.yaml'), encoding='utf-8') as f:
        expected = yaml.safe_load(f)
    fails = 0
    for fname, spec in sorted(expected.items()):
        with open(os.path.join(fdir, fname), encoding='utf-8') as f:
            subj, body = read_email(f.read())
        got = parse_any(reg, subj, body)
        want = spec.get('expect', {})
        missing = {k: v for k, v in want.items() if got.get(k) != v}
        tag = 'synthetic' if spec.get('synthetic') else 'real     '
        if missing or not got:
            fails += 1
            print(f'FAIL [{tag}] {fname}: missing/mismatched {missing}')
            print(f'                     parsed: {got}')
        else:
            amt = got.get('amount')
            amt_s = f'{amt:,} VND' if isinstance(amt, int) else ''
            print(f'PASS [{tag}] {fname}: {got.get("kind")} {amt_s} {got.get("dir", "")}')
    print(f'\n{len(expected) - fails}/{len(expected)} fixtures pass')
    return 1 if fails else 0


def run_probe(path):
    reg = _registry_or_die()
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            raw = f.read()
    except OSError as e:
        print(f'cannot read {path}: {e}', file=sys.stderr)
        return 2
    subj, body = read_email(raw)
    print(f'probe: {path}')
    print(f'registry: {reg.path} ({reg.source})')
    print(f'subject: {subj!r}' + ('' if subj else '  (none — subject matching skipped)'))
    print(f'body: {len(body)} chars (tag-stripped)\n')
    matched = False
    for bank, cfg, fmt in iter_formats(reg.banks):
        subj_hit = any(k in subj for k in fmt.get('subject_any', []))
        print(f'== {bank} / {fmt["name"]}  (kind {fmt["kind"]})   subject: '
              f'{"MATCH" if subj_hit else "no"}')
        tx = {}
        for rx in fmt.get('fields', []):
            m = re.search(rx, body)
            groups = {k: v for k, v in (m.groupdict() if m else {}).items() if v is not None}
            for k, v in groups.items():
                tx.setdefault(k, v.strip())
            label = ', '.join(f'{k}={v!r}' for k, v in groups.items()) or '— no capture'
            shown = rx if len(rx) <= 66 else rx[:63] + '...'
            print(f'   {"MATCH" if m else "MISS "} {label}')
            print(f'         /{shown}')
        req = fmt.get('requires', [])
        req_ok = all(tx.get(r) not in (None, '') for r in req)
        print(f'   requires {req}: {"OK" if req_ok else "MISSING"}')
        if subj_hit and req_ok:
            full = parse_format(fmt, cfg, subj, body)
            if full:
                matched = True
                print(f'   PARSED: {json.dumps(full, ensure_ascii=False)}')
        print()
    print('RESULT: ' + ('matched — the PARSED dict is what the pipeline receives'
                        if matched else
                        'NO format fully matched — a MISS above is the dead regex; fix banks.yaml'))
    return 0 if matched else 1


def run_banks():
    reg = _registry_or_die()
    print(f'active registry: {reg.path}  ({reg.source})\n')
    for warn in validate_registry(reg.banks):
        print(f'warning: {warn}')
    if validate_registry(reg.banks):
        print()
    for bank, cfg in reg.banks.items():
        if not isinstance(cfg, dict):
            continue
        print(f'{bank} — {cfg.get("name", "")}')
        for fmt in cfg.get('formats', []):
            print(f'   {fmt["name"]:28} kind {fmt["kind"]:10} '
                  f'subject_any {fmt.get("subject_any")}')
    return 0


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['--test']:
        sys.exit(run_tests())
    if args[:1] == ['--banks']:
        sys.exit(run_banks())
    if args[:1] == ['--probe'] and len(args) == 2:
        sys.exit(run_probe(args[1]))
    print(__doc__)
    sys.exit(0 if not args else 2)
