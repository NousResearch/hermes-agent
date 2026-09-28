"""Actual CLI A/B for observational commands; no user state or credentials."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--fixed', type=Path, required=True)
parser.add_argument('--python', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
# Resolving the venv's interpreter symlink bypasses its site-packages.
PYTHON = str(args.python.absolute())
TREES = {'baseline': args.baseline.resolve(strict=True), 'fixed': args.fixed.resolve(strict=True)}
COMMANDS = {
    'insights': ['insights', '--days', '30', '--source', 'tui'],
    'list': ['sessions', 'list'],
    'stats': ['sessions', 'stats'],
    'pinned': ['sessions', 'pinned', '--json'],
}
EXPECTED = {
    'insights': ['No sessions found in the last 30 days (source: tui).', 'Hermes Insights'],
    'list': ['No sessions found.', 'reader-fixture'],
    'stats': ['Total sessions: 0', 'Total sessions: 1'],
    'pinned': ['[]', 'reader-fixture'],
}
results = []
for (label, tree), (name, command), populated in itertools.product(TREES.items(), COMMANDS.items(), (False, True)):
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True, stdin=subprocess.DEVNULL).strip()
    with tempfile.TemporaryDirectory(prefix='hermes-readers-cli-ab-') as tmp:
        home = Path(tmp) / 'profile'
        home.mkdir()
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
        env.update(HOME=tmp, USERPROFILE=tmp, HERMES_HOME=str(home), PYTHONPATH=str(tree))
        def run(args):
            return subprocess.run([PYTHON, *args], cwd=tree, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=45)
        if populated:
            seed = run(['-c', "from hermes_state import SessionDB; db=SessionDB(); db.create_session('reader-fixture',source='tui'); db.append_message('reader-fixture',role='user',content='fixture'); db.set_session_pinned('reader-fixture',True); db.close()"])
            assert seed.returncode == 0, seed.stderr
        db = home / 'state.db'
        def digest():
            return hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else None
        before = digest()
        result = run(['-m', 'hermes_cli.main', *command])
        after = digest()
        results.append(dict(tree=label, sha=sha, command=name, populated=populated, exit_code=result.returncode,
                            output_ok=EXPECTED[name][int(populated)] in result.stdout,
                            before=before, after=after, mutated=before != after,
                            stdout=result.stdout, stderr=result.stderr))
expected = len(TREES) * len(COMMANDS) * 2
assert len(results) == expected
passed = all(r['exit_code'] == 0 and r['output_ok'] and r['mutated'] == (r['tree'] == 'baseline') for r in results)
path = args.output.resolve()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(dict(passed=passed, count=len(results), cases=results), indent=2))
print(json.dumps(dict(passed=passed, count=len(results), receipt=str(path), cases=[{k:r[k] for k in ('tree','command','populated','exit_code','output_ok','mutated')} for r in results]), indent=2))
raise SystemExit(0 if passed else 1)

