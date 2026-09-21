import os, sys, json
from pathlib import Path
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')
os.environ.setdefault('HERMES_HOME', '/home/kensei/.hermes')

from hermes_cli.config import get_custom_provider_extra_headers, load_config  # noqa: E402

cfg = load_config()
cps = cfg.get('custom_providers') or []
for entry in cps:
    if not isinstance(entry, dict):
        continue
    name = entry.get('name', '')
    if name in ('xkiro-free', 'xkiro-pro', 'bai'):
        hdrs = get_custom_provider_extra_headers(entry.get('base_url', ''), cps, cfg)
        print(name, 'base_url=', entry.get('base_url'))
        print('   extra_headers keys:', sorted(hdrs.keys()))
        print('   models count:', len(entry.get('models') or {}))
