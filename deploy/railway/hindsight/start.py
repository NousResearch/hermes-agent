"""Pinned Hindsight policy; only deployment endpoints and secrets are substituted."""
import json
import os
from pathlib import Path
import subprocess
import sys


def environment(snapshot, supplied):
    env = dict(supplied)
    for name, value in snapshot['environment'].items():
        env[name] = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
    required = ('DATABASE_URL', 'HINDSIGHT_API_KEY', 'HINDSIGHT_INFERENCE_KEY', 'OPENROUTER_API_KEY')
    missing = [name for name in required if not supplied.get(name)]
    if missing:
        raise ValueError('Missing service secrets: ' + ', '.join(missing))
    env.update(
        HINDSIGHT_API_DATABASE_URL=supplied['DATABASE_URL'],
        HINDSIGHT_API_TENANT_API_KEY=supplied['HINDSIGHT_API_KEY'],
        HINDSIGHT_API_LLM_API_KEY=supplied['HINDSIGHT_INFERENCE_KEY'],
        HINDSIGHT_API_LLM_BASE_URL=supplied.get('HINDSIGHT_CODEX_URL', 'http://hermes.railway.internal:8879/v1'),
        HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL='https://openrouter.ai/api/v1',
        HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=supplied['OPENROUTER_API_KEY'],
        HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL='https://openrouter.ai/api/v1/rerank',
        HINDSIGHT_API_RERANKER_OPENROUTER_API_KEY=supplied['OPENROUTER_API_KEY'],
        HINDSIGHT_API_WORKER_ID='employee-hindsight',
        HINDSIGHT_MANAGED_BANK_TEMPLATE=json.dumps(snapshot['environment']['HINDSIGHT_API_DEFAULT_BANK_TEMPLATE']),
        HINDSIGHT_MANAGED_BANK_FIELDS=json.dumps(list(snapshot['environment']['HINDSIGHT_API_DEFAULT_BANK_TEMPLATE']['bank'])),
        HINDSIGHT_BANK_RECONCILE_URL='http://[::1]:8888',
        HINDSIGHT_BANK_RECONCILE_API_KEY=supplied['HINDSIGHT_API_KEY'],
    )
    return env


def main():
    root = Path(__file__).parent
    snapshot = json.loads((root / 'hindsight-config.json').read_text())
    env = environment(snapshot, os.environ)
    # Reconcile existing banks on each start; new banks inherit the same template.
    subprocess.Popen([sys.executable, str(root / 'bank_reconciler.py')], env=env)
    os.execvpe('hindsight-api', ['hindsight-api', '--host', '::', '--port', '8888'], env)


if __name__ == '__main__':
    main()
