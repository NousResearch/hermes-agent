from pathlib import Path
import json, os, subprocess, sys, uuid

# Non-interactive helper: add a custom-provider api-key credential via the
# same storage path as `hermes auth add` by invoking the CLI in a subprocess
# with the key passed through --api-key (never printed, never logged here).
# Then read back the pool to confirm only label/count, never the secret.

ROOT = Path('/home/kensei/.hermes')
ENV = Path(ROOT / '.env')
AUTH = Path(ROOT / 'auth.json')

def get_env(name: str) -> str:
    line = next((l for l in ENV.read_text().splitlines()
                 if l.startswith(name + '=')), None)
    if not line:
        raise SystemExit(f'MISSING_ENV {name}')
    return line.split('=', 1)[1].strip()

def cli(*args):
    env = {**os.environ, 'HERMES_HOME': str(ROOT)}
    return subprocess.run(['hermes', *args], capture_output=True, text=True,
                          env=env, timeout=120)

def add_key(provider: str, env_var: str, label: str):
    key = get_env(env_var)
    r = cli('auth', 'add', provider, '--type', 'api-key',
            '--api-key', key, '--label', label)
    if r.returncode != 0:
        raise SystemExit(f'ADD_FAIL {provider}: {r.stdout} {r.stderr}')
    # confirm count only
    auth = json.loads(AUTH.read_text())
    entries = (auth.get('credential_pool') or {}).get(provider, [])
    print(f'{provider}: added label={label} count={len(entries)}')
    return r

# xKiro free and paid are distinct custom provider identities.
add_key('custom:xkiro-free', 'XKIRO_FREE_API_KEY', 'xkiro-free-1')
add_key('custom:xkiro-pro', 'XKIRO_PRO_API_KEY', 'xkiro-pro-plus-1')
# B.AI: one provider pool, four fill_first accounts.
for i in range(1, 5):
    add_key('custom:bai', f'BAI_API_KEY_{i}', f'bai-{i}')
print('DONE')
