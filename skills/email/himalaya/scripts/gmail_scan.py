"""Read-only Gmail cursor scanner with durable, scope-bound continuation."""
import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
try:
    from .scan_support import ScanScope, list_argv, parse_page
    from .operation_support import run_recorded, successful_json
    from .task_support import atomic_json, attempt_path, task_lock, stop_check
except ImportError:
    from scan_support import ScanScope, list_argv, parse_page
    from operation_support import run_recorded, successful_json
    from task_support import atomic_json, attempt_path, task_lock, stop_check


def collect(scope, checkpoint, captures, cwd, executable, stopped, budget=100, runner=run_recorded):
    if type(budget) is not int or budget < 1:
        raise ValueError('Positive page budget required')
    checkpoint, captures = Path(checkpoint), Path(captures)
    if not checkpoint.is_absolute() or not captures.is_absolute() or not Path(cwd).is_absolute():
        raise ValueError('Use absolute native state/capture/cwd paths')
    exe = shutil.which(executable)
    if exe is None:
        raise ValueError('Executable cannot be resolved; use actual native filename')
    runtime = {'executable': str(Path(exe).resolve()), 'cwd': str(Path(cwd).resolve())}
    scope_data = json.loads(json.dumps(asdict(scope)))
    captures.mkdir(parents=True, exist_ok=True)
    with task_lock(checkpoint.with_suffix(checkpoint.suffix+'.lock')):
        if checkpoint.exists():
            state = json.loads(checkpoint.read_bytes())
            if (state.get('schema_version') != 1 or state.get('scope') != scope_data
                    or state.get('runtime') != runtime):
                raise ValueError('Checkpoint scope/runtime mismatch')
            ids = state['ids']
            if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
                raise ValueError('Checkpoint ID collection is invalid')
            if state['complete']:
                return state
        else:
            state = {'schema_version': 1, 'scope': scope_data, 'runtime': runtime, 'ids': [],
                     'next_page': None, 'requested_tokens': [], 'pages': 0, 'complete': False,
                     'reason': 'not_started', 'attempts': []}
        seen = set(state['ids'])
        state['reason'] = 'page_budget'
        for _ in range(budget):
            if stopped():
                state['reason'] = 'stopped'
                break
            token = state['next_page']
            if token in state['requested_tokens']:
                state['reason'] = 'repeated_cursor'
                break
            directory = attempt_path(captures, 'page')
            # Save intent before launch; crashes do not lose its capture location.
            state['attempts'].append(str(directory))
            atomic_json(checkpoint, state)
            try:
                runner(list_argv(scope, token, runtime['executable']), runtime['cwd'],
                       directory, timeout=60, stopped=stopped)
                _, _, payload = successful_json(directory)
                page_ids, next_token = parse_page(payload)
            except Exception as exc:
                state['reason'] = 'stopped' if stopped() else 'error'
                state['error_type'] = type(exc).__name__
                atomic_json(checkpoint, state)
                return state
            for mid in page_ids:
                if mid not in seen:
                    state['ids'].append(mid)
                    seen.add(mid)
            state['requested_tokens'].append(token)
            state['pages'] += 1
            state['next_page'] = next_token
            state['complete'] = next_token is None
            state['reason'] = 'exhausted' if state['complete'] else 'page_budget'
            atomic_json(checkpoint, state)
            if state['complete']:
                break
        atomic_json(checkpoint, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True)
    parser.add_argument('--label', action='append', required=True)
    parser.add_argument('--query', default='')
    parser.add_argument('--config')
    parser.add_argument('--executable', default='himalaya')
    parser.add_argument('--page-size', type=int, default=100)
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--captures', required=True, type=Path)
    parser.add_argument('--cwd', required=True)
    parser.add_argument('--stop-file', required=True, type=Path)
    args = parser.parse_args()
    scope = ScanScope(account=args.account, labels=tuple(args.label), query=args.query,
                      page_size=args.page_size, config=args.config)
    result = collect(scope, args.checkpoint, args.captures, args.cwd, args.executable,
                     stop_check(args.stop_file), args.budget)
    print(json.dumps({'unique_ids': len(result['ids']), 'pages': result['pages'],
                      'complete': result['complete'], 'reason': result['reason']}))
    raise SystemExit(0 if result['complete'] else 2)


if __name__ == '__main__':
    main()
