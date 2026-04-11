#!/usr/bin/env python3
"""Reproducible benchmarks for PR #7703 context-enrichment changes.

Modes:
- live-ab: run the live agent (same model/toolset) against current repo vs a baseline worktree
- compression-fidelity: compare structured-state preservation under context compression
- all: run both

Examples:
  python scripts/context_enrichment_benchmark.py --baseline /path/to/baseline --mode all
  python scripts/context_enrichment_benchmark.py --baseline /path/to/baseline --mode live-ab --task-limit 4
  python scripts/context_enrichment_benchmark.py --baseline /path/to/baseline --mode compression-fidelity
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("HERMES_BENCH_PYTHON", sys.executable)
MODEL = os.environ.get("HERMES_BENCH_MODEL", "gpt-5.4")
PROVIDER = os.environ.get("HERMES_BENCH_PROVIDER", "openai-codex")
BASE_URL = os.environ.get("HERMES_BENCH_BASE_URL", "https://chatgpt.com/backend-api/codex")


def resolve_imports(repo: Path, rel_path: str, limit: int = 3) -> list[str]:
    path = repo / rel_path
    src = path.read_text(errors="ignore")
    tree = ast.parse(src)
    results: list[str] = []
    seen: set[str] = set()
    parent_parts = Path(rel_path).parent.parts

    def add(candidate: Path) -> None:
        candidate_s = candidate.as_posix()
        if candidate_s not in seen and (repo / candidate_s).exists():
            seen.add(candidate_s)
            results.append(candidate_s)

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split('.')
                add(Path(*parts).with_suffix('.py'))
                add(Path(*parts) / '__init__.py')
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                base = list(parent_parts)
                for _ in range(max(node.level - 1, 0)):
                    if base:
                        base.pop()
                if mod:
                    base += mod.split('.')
                if node.names and mod == "":
                    for alias in node.names:
                        add(Path(*base, alias.name).with_suffix('.py'))
                        add(Path(*base, alias.name) / '__init__.py')
                        if len(results) >= limit:
                            return results[:limit]
                else:
                    add(Path(*base).with_suffix('.py'))
                    add(Path(*base) / '__init__.py')
            else:
                if mod:
                    parts = mod.split('.')
                    add(Path(*parts).with_suffix('.py'))
                    add(Path(*parts) / '__init__.py')
        if len(results) >= limit:
            return results[:limit]
    return results[:limit]


def mirrored_test(repo: Path, rel_path: str) -> str | None:
    rel = Path(rel_path)
    stem = rel.stem
    parent = rel.parent
    candidates = [
        repo / 'tests' / parent / f'test_{stem}.py',
        repo / 'tests' / parent / f'{stem}_test.py',
        repo / 'tests' / f'test_{stem}.py',
    ]
    for c in candidates:
        if c.exists():
            return str(c.relative_to(repo))
    return None


def build_live_tasks(repo: Path) -> list[dict[str, Any]]:
    file_tasks = [
        'tools/file_tools.py',
        'agent/context_compressor.py',
        'tools/mcp_tool.py',
        'tools/terminal_tool.py',
        'tools/registry.py',
        'agent/prompt_builder.py',
        'agent/credential_pool.py',
        'agent/model_metadata.py',
    ]
    tasks: list[dict[str, Any]] = []
    for rel in file_tasks:
        expected = list(resolve_imports(repo, rel, limit=3))
        test_path = mirrored_test(repo, rel)
        if test_path:
            expected.append(test_path)
        tasks.append(
            {
                'name': f'imports::{rel}',
                'prompt': (
                    f'Use only file tools. Read {rel} and answer two things: '
                    f'(1) which project-local modules are imported at the top of the file, '
                    f'and (2) what is the mirrored test file path for this module if it exists. '
                    f'Return only bullet points with file paths.'
                ),
                'expected': expected,
            }
        )

    tasks.extend(
        [
            {
                'name': 'search::preview_default',
                'prompt': 'Use only file tools. Find the default preview_lines value for search_files and the exact file/function where search_files is registered in the registry. Return only: default value, schema name, handler name, file path.',
                'expected': ['2', 'SEARCH_FILES_SCHEMA', '_handle_search_files', 'tools/file_tools.py'],
            },
            {
                'name': 'search::file_read_max_chars',
                'prompt': 'Use only file tools. Find the default file_read_max_chars value and the function that loads it. Return only: value, function name, file path.',
                'expected': ['100_000', '_get_max_read_chars', 'tools/file_tools.py'],
            },
            {
                'name': 'search::task_checkpoint_constants',
                'prompt': 'Use only file tools. Find the TaskCheckpoint prefix constant and the max chars limit used for checkpoint injection. Return only: prefix constant name, numeric limit, file path.',
                'expected': ['TASK_CHECKPOINT_PREFIX', '500', 'agent/context_compressor.py'],
            },
            {
                'name': 'search::write_denied_prefixes',
                'prompt': 'Use only file tools. Find the variable name that stores write-denied path prefixes and the file where it is defined. Return only: variable name, file path.',
                'expected': ['WRITE_DENIED_PREFIXES', 'tools/file_operations.py'],
            },
        ]
    )
    return tasks


CHILD_SCRIPT = r'''
import json, os, sys, time
repo=os.getcwd(); sys.path.insert(0, repo)
from run_agent import AIAgent
prompt=os.environ['AB_TASK']
start=time.time()
agent=AIAgent(
    model=os.environ['AB_MODEL'],
    provider=os.environ['AB_PROVIDER'],
    base_url=os.environ['AB_BASE_URL'],
    enabled_toolsets=['file'],
    quiet_mode=True,
    tool_delay=0,
    max_iterations=8,
    skip_context_files=True,
    skip_memory=True,
    persist_session=False,
)
res=agent.run_conversation(prompt, task_id='ab-bench-live')
counts={}
for m in res.get('messages', []):
    for tc in (m.get('tool_calls') or []):
        name=((tc.get('function') or {}).get('name'))
        counts[name]=counts.get(name,0)+1
print(json.dumps({
    'elapsed': round(time.time()-start,2),
    'api_calls': res.get('api_calls'),
    'completed': res.get('completed'),
    'tool_counts': counts,
    'final_response': res.get('final_response','')
}, ensure_ascii=False))
'''


def run_live_task(repo: Path, prompt: str) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            'AB_TASK': prompt,
            'AB_MODEL': MODEL,
            'AB_PROVIDER': PROVIDER,
            'AB_BASE_URL': BASE_URL,
        }
    )
    r = subprocess.run(
        [PYTHON, '-c', CHILD_SCRIPT],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=420,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    js = None
    for ln in reversed(lines):
        if ln.strip().startswith('{') and ln.strip().endswith('}'):
            js = ln.strip()
            break
    if r.returncode != 0 or not js:
        return {'error': (r.stderr or r.stdout)[-1500:]}
    return json.loads(js)


def score_text(text: str, expected: list[str]) -> tuple[int, int]:
    score = sum(1 for item in expected if item in text)
    return score, len(expected)


def run_live_ab(feature_repo: Path, baseline_repo: Path, task_limit: int | None = None) -> dict[str, Any]:
    tasks = build_live_tasks(feature_repo)
    if task_limit:
        tasks = tasks[:task_limit]
    suite: dict[str, Any] = {}
    for label, repo in (('feature', feature_repo), ('baseline', baseline_repo)):
        suite[label] = {}
        for task in tasks:
            data = run_live_task(repo, task['prompt'])
            if 'error' not in data:
                score, total = score_text(data.get('final_response', ''), task['expected'])
                data['score'] = score
                data['total'] = total
            suite[label][task['name']] = data

        tot_api = tot_time = tot_search = tot_read = tot_score = tot_total = 0
        completed = 0
        for d in suite[label].values():
            if 'error' in d:
                continue
            tot_api += d.get('api_calls') or 0
            tot_time += d.get('elapsed') or 0
            tot_search += d.get('tool_counts', {}).get('search_files', 0)
            tot_read += d.get('tool_counts', {}).get('read_file', 0)
            tot_score += d.get('score', 0)
            tot_total += d.get('total', 0)
            completed += 1 if d.get('completed') else 0
        suite[label]['_summary'] = {
            'tasks': len(tasks),
            'tasks_completed': completed,
            'api_calls_total': tot_api,
            'elapsed_total': round(tot_time, 2),
            'search_calls_total': tot_search,
            'read_calls_total': tot_read,
            'score_total': f'{tot_score}/{tot_total}',
        }
    return suite


COMPRESSION_SCRIPT = r'''
import json, os, sys
from unittest.mock import MagicMock, patch
repo=os.getcwd(); sys.path.insert(0, repo)
from agent.context_compressor import ContextCompressor

cases = [
    {
        'name':'pending_write_after_failed_test',
        'messages':[
            {'role':'system','content':'sys'},{'role':'user','content':'head1'},{'role':'assistant','content':'head2'},
            {'role':'user','content':'fix docs'},
            {'role':'assistant','content':'writing parser', 'tool_calls':[{'id':'w1','type':'function','function':{'name':'write_file','arguments':json.dumps({'path':'src/parser.py','content':'patched'})}}]},
            {'role':'tool','tool_call_id':'w1','content':json.dumps({'path':'src/parser.py','status':'updated'})},
            {'role':'assistant','content':'todo sync', 'tool_calls':[{'id':'todo1','type':'function','function':{'name':'todo','arguments':json.dumps({'todos':[{'id':'docs','content':'Write docs/architecture.md','status':'in_progress'}]})}}]},
            {'role':'tool','tool_call_id':'todo1','content':json.dumps({'todos':[{'id':'docs','content':'Write docs/architecture.md','status':'in_progress'}]})},
            {'role':'assistant','content':'run pytest', 'tool_calls':[{'id':'t1','type':'function','function':{'name':'terminal','arguments':json.dumps({'command':'pytest tests/test_docs.py -q'})}}]},
            {'role':'tool','tool_call_id':'t1','content':'FAILED tests/test_docs.py::test_architecture_doc_exists - FileNotFoundError: docs/architecture.md'},
            {'role':'assistant','content':'will write docs now', 'tool_calls':[{'id':'w2','type':'function','function':{'name':'write_file','arguments':json.dumps({'path':'docs/architecture.md','content':'draft'})}}]},
            {'role':'user','content':'continue'},{'role':'assistant','content':'tail a'},{'role':'user','content':'tail b'},{'role':'assistant','content':'tail c'},
        ],
        'expect':['docs/architecture.md','Write docs/architecture.md']
    },
    {
        'name':'failed_write_should_not_count_modified',
        'messages':[
            {'role':'system','content':'sys'},{'role':'user','content':'head1'},{'role':'assistant','content':'head2'},
            {'role':'user','content':'write file'},
            {'role':'assistant','content':'try write', 'tool_calls':[{'id':'wfail','type':'function','function':{'name':'write_file','arguments':json.dumps({'path':'src/bad.py','content':'x'})}}]},
            {'role':'tool','tool_call_id':'wfail','content':json.dumps({'error':'permission denied'})},
            {'role':'user','content':'continue'},{'role':'assistant','content':'tail a'},{'role':'user','content':'tail b'},{'role':'assistant','content':'tail c'},
        ],
        'expect_absent':['src/bad.py']
    },
    {
        'name':'v4a_multi_patch_middle',
        'messages':[
            {'role':'system','content':'sys'},{'role':'user','content':'head user 1'},{'role':'assistant','content':'head assist 1'},
            {'role':'user','content':'apply patch'},
            {'role':'assistant','content':'patching', 'tool_calls':[{'id':'p1','type':'function','function':{'name':'patch','arguments':json.dumps({'mode':'patch','patch':'*** Begin Patch\n*** Update File: src/a.py\n@@\n-old\n+new\n*** Update File: tests/test_a.py\n@@\n-old\n+new\n*** End Patch'})}}]},
            {'role':'tool','tool_call_id':'p1','content':json.dumps({'status':'ok','operations':2})},
            {'role':'user','content':'continue'},{'role':'assistant','content':'tail a'},{'role':'user','content':'tail b'},{'role':'assistant','content':'tail c'},
        ],
        'expect':['src/a.py','tests/test_a.py']
    },
    {
        'name':'pending_terminal_and_plan',
        'messages':[
            {'role':'system','content':'sys'},{'role':'user','content':'head1'},{'role':'assistant','content':'head2'},
            {'role':'user','content':'run tests'},
            {'role':'assistant','content':'todo sync', 'tool_calls':[{'id':'todo2','type':'function','function':{'name':'todo','arguments':json.dumps({'todos':[{'id':'verify','content':'Rerun integration tests','status':'in_progress'}]})}}]},
            {'role':'tool','tool_call_id':'todo2','content':json.dumps({'todos':[{'id':'verify','content':'Rerun integration tests','status':'in_progress'}]})},
            {'role':'assistant','content':'running terminal', 'tool_calls':[{'id':'term2','type':'function','function':{'name':'terminal','arguments':json.dumps({'command':'pytest tests/integration -q'})}}]},
            {'role':'user','content':'continue'},{'role':'assistant','content':'tail a'},{'role':'user','content':'tail b'},{'role':'assistant','content':'tail c'},
        ],
        'expect':['terminal:pytest tests/integration -q','Rerun integration tests']
    },
]
with patch('agent.context_compressor.get_model_context_length', return_value=100000):
    compressor = ContextCompressor(model='test/model', quiet_mode=True, protect_first_n=3, protect_last_n=3)
mock_response = MagicMock(); mock_response.choices=[MagicMock()]; mock_response.choices[0].message.content='[CONTEXT SUMMARY]: compacted handoff'
results=[]
with patch('agent.context_compressor.call_llm', return_value=mock_response):
    for case in cases:
        out = compressor.compress(case['messages'], current_tokens=50000)
        text='\n'.join((m.get('content') or '') for m in out)
        score=0; total=0
        for item in case.get('expect',[]):
            total += 1
            score += 1 if item in text else 0
        for item in case.get('expect_absent',[]):
            total += 1
            score += 1 if item not in text else 0
        results.append({'name':case['name'],'score':score,'total':total})
print(json.dumps(results, ensure_ascii=False))
'''


def run_compression(repo: Path) -> list[dict[str, Any]]:
    r = subprocess.run([PYTHON, '-c', COMPRESSION_SCRIPT], cwd=repo, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[-1500:])
    return json.loads(r.stdout)


def run_compression_fidelity(feature_repo: Path, baseline_repo: Path) -> dict[str, Any]:
    return {
        'feature': run_compression(feature_repo),
        'baseline': run_compression(baseline_repo),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, help='Path to baseline worktree/repo')
    ap.add_argument('--mode', choices=['live-ab', 'compression-fidelity', 'all'], default='all')
    ap.add_argument('--task-limit', type=int, default=None, help='Optional limit for live-ab tasks')
    args = ap.parse_args()

    feature_repo = ROOT
    baseline_repo = Path(args.baseline).resolve()
    result: dict[str, Any] = {'feature_repo': str(feature_repo), 'baseline_repo': str(baseline_repo)}

    if args.mode in ('live-ab', 'all'):
        result['live_ab'] = run_live_ab(feature_repo, baseline_repo, task_limit=args.task_limit)
    if args.mode in ('compression-fidelity', 'all'):
        result['compression_fidelity'] = run_compression_fidelity(feature_repo, baseline_repo)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
