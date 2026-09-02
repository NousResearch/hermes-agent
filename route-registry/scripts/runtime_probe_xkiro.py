import os, subprocess, sys
from pathlib import Path
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')
os.environ.setdefault('HERMES_HOME', '/home/kensei/.hermes')

# Spawn a tiny one-shot completion via the CLI to prove the runtime can actually
# use a config chain that resolves custom:xkiro-free. Use --model to point at the
# same model that ceecee-brand's chain would serve, through the custom provider.
r = subprocess.run(
    ['hermes', 'chat', '-q', 'Reply with the single word: OK',
     '--provider', 'custom:xkiro-free',
     '--model', 'deepseek/deepseek-v4-flash',
     '--toolsets', ''],
    capture_output=True, text=True, timeout=120,
    env={**os.environ, 'HERMES_HOME': '/home/kensei/.hermes'},
)
print('exit:', r.returncode)
print('stdout:', r.stdout[-500:])
print('stderr:', r.stderr[-500:])
