#!/usr/bin/env python3
"""Benchmark Context Governor against full-context retention using real Hermes files."""

import sys
import time
import tempfile
import shutil
import os
import importlib
from pathlib import Path

sys.path.insert(0, '/home/orchestrator/.hermes/hermes-agent')

from hermes_state import SessionDB
from hermes_constants import get_hermes_home
from agent.context_governor import get_context_governor, reset_context_governor


REAL_FILE = Path('/home/orchestrator/.hermes/hermes-agent/agent/context_governor.py')


def setup_temp_home():
    home = Path(tempfile.mkdtemp())
    os.environ['HERMES_HOME'] = str(home)
    import hermes_constants
    importlib.reload(hermes_constants)
    return home


def simulate_real_workflow(governor, n_reads=50):
    governor.on_session_start('hermes-review')
    governor.update_ledger(
        repo='NousResearch/hermes-agent',
        objective='Review and harden Context Governor implementation',
        current_branch='feat/context-governor',
        known_constraints=[
            'Keep changes minimal and surgical',
            'Do not break existing session DB schema',
            'All tests must pass',
        ],
    )

    real_content = REAL_FILE.read_text()

    # 1. Search for related files
    governor.record_tool_call(
        'search_files',
        {'pattern': 'context_governor', 'path': '/home/orchestrator/.hermes/hermes-agent/agent'},
        'total_count: 21',
        0,
    )

    # 2. Read the main file many times (simulating line-by-line review)
    for i in range(n_reads):
        governor.record_tool_call(
            'read_file',
            {'path': str(REAL_FILE), 'offset': i * 10 + 1, 'limit': 30},
            real_content,
            i + 1,
        )

    # 3. Inspect tests
    governor.record_tool_call(
        'read_file',
        {'path': '/home/orchestrator/.hermes/hermes-agent/tests/agent/test_context_governor.py'},
        Path('/home/orchestrator/.hermes/hermes-agent/tests/agent/test_context_governor.py').read_text(),
        n_reads + 1,
    )

    # 4. Patch
    governor.record_tool_call(
        'patch',
        {'path': str(REAL_FILE), 'mode': 'replace'},
        'ok',
        n_reads + 2,
    )

    # 5. Run tests
    governor.record_tool_call(
        'execute_code',
        {'code': 'pytest tests/agent/test_context_governor.py -q'},
        '10 passed in 0.52s',
        n_reads + 3,
    )

    # 6. Check status
    governor.record_tool_call(
        'terminal',
        {'command': 'git status --short'},
        'M agent/context_governor.py\nM agent/agent_init.py\nM run_agent.py',
        n_reads + 4,
    )


def benchmark(config, n_reads=50):
    home = setup_temp_home()
    try:
        reset_context_governor()
        g = get_context_governor()
        for k, v in config.items():
            setattr(g, k, v)

        db = SessionDB(get_hermes_home() / 'sessions.db')
        db.create_session('hermes-review', 'cli', model='test-model')
        db.close()

        start = time.perf_counter()
        simulate_real_workflow(g, n_reads=n_reads)
        ctx = g.get_context_for_model()
        elapsed = time.perf_counter() - start

        return {
            'elapsed_s': elapsed,
            'context_chars': len(ctx),
            'context_words': len(ctx.split()),
            'raw_tool_calls': len(g.raw_tool_calls),
            'summarized_window': len(g.summarized_window),
        }
    finally:
        shutil.rmtree(home, ignore_errors=True)


if __name__ == '__main__':
    N = 50
    gov = benchmark({'raw_tool_window': 5, 'summary_window': 3}, n_reads=N)
    full = benchmark({'raw_tool_window': 100000, 'summary_window': 0}, n_reads=N)

    print(f"Real-file workflow A/B ({N} reads of {REAL_FILE.name})")
    print(f"  Governor:       {gov['context_chars']:,} chars / {gov['context_words']:,} words")
    print(f"  Full context:   {full['context_chars']:,} chars / {full['context_words']:,} words")
    print(f"  Reduction:      {full['context_chars'] / gov['context_chars']:.1f}x")
    print(f"  Governor time:  {gov['elapsed_s']:.4f}s")
    print(f"  Full time:      {full['elapsed_s']:.4f}s")
