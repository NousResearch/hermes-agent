#!/usr/bin/env python3
"""Benchmark Context Governor across three real project workflows."""

import sys
import time
import tempfile
import shutil
import os
import importlib
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

sys.path.insert(0, '/home/orchestrator/.hermes/hermes-agent')

from hermes_state import SessionDB
from hermes_constants import get_hermes_home
from agent.context_governor import get_context_governor, reset_context_governor


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]
    response: str


@dataclass
class Scenario:
    name: str
    repo: str
    objective: str
    constraints: List[str]
    calls: List[ToolCall] = field(default_factory=list)


REAL_FILES = {
    'aktr_test': Path('/home/orchestrator/repos/github/marcusgoll/cfipros/api/tests/unit/services/test_aktr_ocr.py'),
    'aktr_service': Path('/home/orchestrator/repos/github/marcusgoll/cfipros/api/app/services/aktr/ocr.py'),
    'aktr_exceptions': Path('/home/orchestrator/repos/github/marcusgoll/cfipros/api/app/services/aktr/exceptions.py'),
    'sentry_webhook': Path('/home/orchestrator/repos/github/marcusgoll/cfipros/api/app/routers/webhooks/sentry.py'),
    'sentry_health_test': Path('/home/orchestrator/repos/github/marcusgoll/cfipros/tests/test_health_check.py'),
    'obsidian_compose': Path('/home/marcusgoll/docker/obsidian/docker-compose.yml'),
}


def read(p: Path) -> str:
    return p.read_text() if p.exists() else f'[missing: {p}]'


def build_aktr_scenario(n_reads: int = 20) -> Scenario:
    s = Scenario(
        name='CFIPros AKTR OCR test improvement',
        repo='marcusgoll/cfipros',
        objective='Add an edge-case test for lowercase ACS codes and verify the parser behavior',
        constraints=[
            'Do not break existing tests',
            'Only touch the unit test file',
            'Use the existing service pattern',
        ],
    )
    test_content = read(REAL_FILES['aktr_test'])
    service_content = read(REAL_FILES['aktr_service'])
    exceptions_content = read(REAL_FILES['aktr_exceptions'])

    s.calls.append(ToolCall('search_files', {'pattern': 'parse_acs_codes|ACS_CODE_PATTERN', 'path': 'api/app/services/aktr'}, 'total_count: 3'))
    s.calls.append(ToolCall('read_file', {'path': str(REAL_FILES['aktr_test'])}, test_content[:4000]))
    s.calls.append(ToolCall('read_file', {'path': str(REAL_FILES['aktr_service'])}, service_content[:4000]))
    s.calls.append(ToolCall('read_file', {'path': str(REAL_FILES['aktr_exceptions'])}, exceptions_content[:2000]))

    for i in range(n_reads):
        offset = i * 15 + 1
        s.calls.append(ToolCall('read_file', {'path': str(REAL_FILES['aktr_service']), 'offset': offset, 'limit': 30}, service_content))

    s.calls.append(ToolCall('patch', {'path': str(REAL_FILES['aktr_test']), 'mode': 'replace'}, 'ok'))
    s.calls.append(ToolCall('execute_code', {'code': 'pytest tests/unit/services/test_aktr_ocr.py -q'}, '12 passed, 2 skipped'))
    s.calls.append(ToolCall('terminal', {'command': 'cd api && uv run pytest -x -q tests/unit/services/test_aktr_ocr.py'}, '12 passed, 2 skipped'))
    s.calls.append(ToolCall('terminal', {'command': 'git status --short'}, 'M api/tests/unit/services/test_aktr_ocr.py'))
    return s


def build_sentry_scenario(n_reads: int = 20) -> Scenario:
    s = Scenario(
        name='CFIPros Sentry webhook auth investigation',
        repo='marcusgoll/cfipros',
        objective='Verify Sentry webhook HMAC signature validation and admin token enforcement',
        constraints=[
            'Do not change behavior without tests',
            'Focus on auth and audit logging',
            'Check dev vs production env differences',
        ],
    )
    webhook_content = read(REAL_FILES['sentry_webhook'])
    health_test_content = read(REAL_FILES['sentry_health_test'])

    s.calls.append(ToolCall('search_files', {'pattern': 'SENTRY_WEBHOOK_SECRET|hmac|AUTO_INVESTIGATE', 'path': 'api/app/routers/webhooks'}, 'total_count: 8'))
    s.calls.append(ToolCall('read_file', {'path': str(REAL_FILES['sentry_webhook'])}, webhook_content[:5000]))
    s.calls.append(ToolCall('read_file', {'path': str(REAL_FILES['sentry_health_test'])}, health_test_content[:3000]))

    for i in range(n_reads):
        offset = i * 15 + 1
        s.calls.append(ToolCall('read_file', {'path': str(REAL_FILES['sentry_webhook']), 'offset': offset, 'limit': 30}, webhook_content))

    s.calls.append(ToolCall('patch', {'path': str(REAL_FILES['sentry_webhook']), 'mode': 'replace'}, 'ok'))
    s.calls.append(ToolCall('execute_code', {'code': 'pytest tests/test_health_check.py -q'}, '2 passed'))
    s.calls.append(ToolCall('terminal', {'command': 'cd api && uv run pytest -x -q tests/routers/webhooks'}, '3 passed'))
    return s


def build_homelab_scenario(n_reads: int = 15) -> Scenario:
    s = Scenario(
        name='Homelab Obsidian Docker restart loop',
        repo='marcusgoll/docker/obsidian',
        objective='Investigate why the Obsidian container is restarting and verify compose health',
        constraints=[
            'Do not restart production without evidence',
            'Check logs and compose config first',
            'Preserve user data in ./config',
        ],
    )
    compose_content = read(REAL_FILES['obsidian_compose'])

    s.calls.append(ToolCall('terminal', {'command': 'docker ps --filter name=obsidian --format "{{.Names}} {{.Status}} {{.Ports}}"'}, 'obsidian Up 3 days 0.0.0.0:3000->3000/tcp'))
    s.calls.append(ToolCall('read_file', {'path': str(REAL_FILES['obsidian_compose'])}, compose_content))
    s.calls.append(ToolCall('read_file', {'path': '/home/marcusgoll/docker/obsidian/.env'}, 'PUID=1000\nPGID=1000\nTZ=America/Chicago\nHTTP_PORT=3000\nHTTPS_PORT=3001'))
    s.calls.append(ToolCall('terminal', {'command': 'docker logs --tail 50 obsidian 2>&1'}, 'INFO  Server started on port 3000\nINFO  ...' * 20))
    s.calls.append(ToolCall('terminal', {'command': 'docker inspect obsidian --format="{{.State.RestartCount}} {{.State.Status}}"'}, '0 running'))

    for i in range(n_reads):
        s.calls.append(ToolCall('terminal', {'command': 'docker logs --tail 30 obsidian 2>&1'}, 'INFO  ...' * 50))

    s.calls.append(ToolCall('read_file', {'path': '/home/marcusgoll/docker/obsidian/config/obsidian.log'}, 'INFO  ...' * 30))
    s.calls.append(ToolCall('patch', {'path': str(REAL_FILES['obsidian_compose']), 'mode': 'replace'}, 'ok'))
    s.calls.append(ToolCall('terminal', {'command': 'docker compose -f /home/marcusgoll/docker/obsidian/docker-compose.yml config'}, 'services:\n  obsidian:...'))
    return s


def setup_temp_home() -> Path:
    home = Path(tempfile.mkdtemp())
    os.environ['HERMES_HOME'] = str(home)
    import hermes_constants
    importlib.reload(hermes_constants)
    return home


def run_scenario(scenario: Scenario, raw_window: int, summary_window: int) -> Dict[str, Any]:
    home = setup_temp_home()
    try:
        reset_context_governor()
        g = get_context_governor()
        g.raw_tool_window = raw_window
        g.summary_window = summary_window

        db = SessionDB(get_hermes_home() / 'sessions.db')
        db.create_session('benchmark-' + scenario.name.replace(' ', '-').lower(), 'cli', model='test-model')
        db.close()

        g.on_session_start('benchmark-' + scenario.name.replace(' ', '-').lower())
        g.update_ledger(
            repo=scenario.repo,
            objective=scenario.objective,
            current_branch='main',
            known_constraints=scenario.constraints,
        )

        start = time.perf_counter()
        for i, call in enumerate(scenario.calls):
            g.record_tool_call(call.tool, call.args, call.response, i)
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


def main():
    scenarios = [
        build_aktr_scenario(n_reads=20),
        build_sentry_scenario(n_reads=20),
        build_homelab_scenario(n_reads=15),
    ]

    results = []
    for s in scenarios:
        gov = run_scenario(s, raw_window=5, summary_window=3)
        full = run_scenario(s, raw_window=100000, summary_window=0)
        results.append({
            'scenario': s.name,
            'governor_chars': gov['context_chars'],
            'full_chars': full['context_chars'],
            'reduction': full['context_chars'] / gov['context_chars'] if gov['context_chars'] else 0,
            'governor_time': gov['elapsed_s'],
            'full_time': full['elapsed_s'],
        })

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
