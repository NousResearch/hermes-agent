from pathlib import Path
import json, subprocess, yaml
root=Path('/home/kensei/.hermes')
names=['kensei-review','octacon-architect','octacon-backend','wesker','moss','octacon','octacon-frontend','quan']
for name in names:
    home=root/'profiles'/name
    p=subprocess.run(['hermes','config','check'],env={**__import__('os').environ,'HERMES_HOME':str(home)},capture_output=True,text=True,timeout=60)
    print(f'{name}: config_check_exit={p.returncode}')
    if p.returncode:
        print((p.stdout+p.stderr)[-800:])
# Sanitized auth-pool presence only.
auth=json.loads((root/'auth.json').read_text())
pools=auth.get('credential_pool',{})
for provider in ['custom:xkiro-free','custom:xkiro-pro','custom:bai']:
    entries=pools.get(provider,[])
    print(f'{provider}: root_pool_entries={len(entries) if isinstance(entries,list) else 0}')
# Registry gate status only.
reg=yaml.safe_load(Path('/home/kensei/repos/KenseiAgent/route-registry/registry/route-slots.yaml').read_text())
for provider_prefix in ('xkiro/','b-ai/'):
    slots=[s for s in reg['slots'] if str(s.get('provider_account','')).startswith(provider_prefix)]
    print(provider_prefix, [(s['slot'],s['status'],s['disabled']) for s in slots])
