import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')

# 1) confirm the live config the running gateway would load contains the new routes
import yaml
root = Path('/home/kensei/.hermes')
checks = {
    'root fallback': root / 'config.yaml',
    'ceecee-brand': root / 'profiles' / 'ceecee-brand' / 'config.yaml',
    'denji': root / 'profiles' / 'denji' / 'config.yaml',
    'octacon': root / 'profiles' / 'octacon' / 'config.yaml',
}
for label, p in checks.items():
    doc = yaml.safe_load(p.read_text())
    fp = doc.get('fallback_providers') or []
    provs = [e.get('provider') for e in fp]
    has_xk = any('xkiro' in str(x) for x in provs)
    has_bai = any(x == 'custom:bai' for x in provs)
    print(f'{label}: xkiro={has_xk} bai={has_bai} providers={provs}')
