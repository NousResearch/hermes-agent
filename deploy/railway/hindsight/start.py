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
    required = ('DATABASE_URL', 'HINDSIGHT_API_KEY', 'HINDSIGHT_INFERENCE_KEY')
    missing = [name for name in required if not supplied.get(name)]
    if missing:
        raise ValueError('Missing service secrets: ' + ', '.join(missing))
    env.update(
        HINDSIGHT_API_DATABASE_URL=supplied['DATABASE_URL'],
        HINDSIGHT_API_TENANT_API_KEY=supplied['HINDSIGHT_API_KEY'],
        HINDSIGHT_API_LLM_API_KEY=supplied['HINDSIGHT_INFERENCE_KEY'],
        HINDSIGHT_API_LLM_BASE_URL=supplied.get('HINDSIGHT_CODEX_URL', 'http://hermes.railway.internal:8879/v1'),
        HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL='https://openrouter.ai/api/v1',
        HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=supplied.get('OPENROUTER_API_KEY') or 'not-configured',
        HINDSIGHT_API_RERANKER_OPENROUTER_BASE_URL='https://openrouter.ai/api/v1/rerank',
        HINDSIGHT_API_RERANKER_OPENROUTER_API_KEY=supplied.get('OPENROUTER_API_KEY') or 'not-configured',
        HINDSIGHT_API_WORKER_ID='employee-hindsight',
        HINDSIGHT_MANAGED_BANK_TEMPLATE=json.dumps(snapshot['environment']['HINDSIGHT_API_DEFAULT_BANK_TEMPLATE']),
        HINDSIGHT_MANAGED_BANK_FIELDS=json.dumps(list(snapshot['environment']['HINDSIGHT_API_DEFAULT_BANK_TEMPLATE']['bank'])),
        HINDSIGHT_BANK_RECONCILE_URL='http://[::1]:8888',
        HINDSIGHT_BANK_RECONCILE_API_KEY=supplied['HINDSIGHT_API_KEY'],
    )
    return env


def main():
    # Hermes owns editable settings. Poll the existing private authenticated
    # service endpoint; restart only the Hindsight child when they change.
    import signal
    import time
    import urllib.request
    root = Path(__file__).parent
    snapshot = json.loads((root / 'hindsight-config.json').read_text(encoding='utf-8-sig'))
    base = os.environ.get('HINDSIGHT_CODEX_URL', 'http://hermes.railway.internal:8879/v1').rstrip('/')
    headers = {'Authorization': 'Bearer ' + os.environ['HINDSIGHT_INFERENCE_KEY'], 'Content-Type': 'application/json'}
    stopped = False
    process = reconciler = None
    revision = None

    def shutdown(*_):
        nonlocal stopped
        stopped = True

    def stop(child):
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        while not stopped:
            try:
                with urllib.request.urlopen(urllib.request.Request(base + '/settings', headers=headers), timeout=10) as response:
                    settings = json.load(response)
                if settings['revision'] != revision or process is None or process.poll() is not None:
                    stop(reconciler)
                    stop(process)
                    env = environment(snapshot, {**os.environ, 'OPENROUTER_API_KEY': settings['openrouter_api_key']})
                    for name in ('llm_model', 'llm_reasoning_effort', 'reflect_llm_reasoning_effort'):
                        env['HINDSIGHT_API_' + name.upper()] = settings[name]
                    process = subprocess.Popen(['hindsight-api', '--host', '::', '--port', '8888'], env=env)
                    reconciler = subprocess.Popen([sys.executable, str(root / 'bank_reconciler.py')], env=env)
                    revision = settings['revision']
                state = 'starting'
                try:
                    with urllib.request.urlopen('http://[::1]:8888/health', timeout=3) as response:
                        state = 'ready' if response.status == 200 else 'error'
                except OSError:
                    pass
                payload = json.dumps({'revision': revision, 'state': state}).encode()
                with urllib.request.urlopen(urllib.request.Request(base + '/settings/status', data=payload, headers=headers), timeout=10):
                    pass
            except (OSError, ValueError, KeyError):
                # Keep the running child on transient control-plane loss; never
                # print settings or HTTP bodies, which include a provider key.
                print('Hindsight settings sync unavailable; retrying.', flush=True)
            for _ in range(5):
                if stopped:
                    break
                time.sleep(1)
    finally:
        stop(reconciler)
        stop(process)


if __name__ == '__main__':
    main()
