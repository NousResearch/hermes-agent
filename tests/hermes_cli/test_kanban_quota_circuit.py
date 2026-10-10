"""Durable quota regression for lab-ops t_a13f0407; real SQLite and dispatch path."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading

import httpx
import openai
import pytest

import cli

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_dispatch as dispatch
from hermes_cli import kanban_quota as quota, kanban_db_notify as notify
from agent.error_classifier import classify_api_error, FailoverReason


@pytest.fixture
def lab(tmp_path, monkeypatch, all_assignees_spawnable):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_DB', str(home / 'board.db'))
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    monkeypatch.setattr(kb, '_pid_alive', lambda _: False)
    for profile in ('a', 'b'):
        root = home / 'profiles' / profile
        root.mkdir(parents=True)
        (root / 'config.yaml').write_text('model:\n  provider: xai-oauth\n  default: grok\n')
        (root / '.env').write_text(f'OPENAI_API_KEY=canary-{profile}\n')
    clock = [1800000000.0]
    monkeypatch.setattr(quota.time, 'time', lambda: clock[0])
    from datetime import datetime
    from agent import retry_utils
    class LabDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(clock[0], tz)
    monkeypatch.setattr(retry_utils, 'datetime', LabDatetime)
    kb.init_db()
    yield home, clock


def error(status=403, code='personal-team-blocked:spending-limit', retry=None):
    response = httpx.Response(status, request=httpx.Request('POST', 'https://example.invalid'),
                              headers={} if retry is None else {'Retry-After': str(retry)})
    return openai.APIStatusError('provider refusal', response=response,
                                body={'error': {'code': code, 'message': 'provider refusal'}})


def spawn_failure(conn, tid, monkeypatch, exc=None, provider='xai-oauth'):
    task = kb.claim_task(conn, tid)
    assert task is not None
    exc = exc or error()
    verdict = classify_api_error(exc, provider=provider)
    evidence = quota.failure_evidence(SimpleNamespace(provider=provider), verdict, exc)
    monkeypatch.setenv('HERMES_KANBAN_TASK', tid)
    monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(task.current_run_id))
    monkeypatch.setenv('HERMES_KANBAN_CLAIM_LOCK', task.claim_lock)
    assert cli._single_query_exit_code({'failed': True, 'failure_reason': verdict.reason.value,
                                        'provider_failure': evidence}) == 75
    conn.execute('UPDATE tasks SET worker_pid=?, started_at=? WHERE id=?', (900001, 1, tid))
    conn.commit()
    # Durable trailer fallback is exercised after clearing the process-local registry.
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('[kanban-worker-exit] rc=75\n')
    dispatch._recent_worker_exits.clear()
    return verdict


def tick(conn, launched):
    return dispatch.dispatch_once(conn, spawn_fn=lambda task, _: launched.append(task.id),
                                  max_in_progress=100, max_in_progress_per_profile=100,
                                  failure_limit=2)


def test_hard_quota_survives_334_ticks_restart_and_atomic_sibling_claims(lab, monkeypatch):
    home, clock = lab
    from agent.secret_scope import set_multiplex_context, reset_multiplex_context
    token = set_multiplex_context(True)
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title='incident', assignee='a')
            sibling = kb.create_task(conn, title='same credential', assignee='a')
            healthy = kb.create_task(conn, title='Codex', assignee='a', model_override='gpt-5', provider_override='openai-codex')
            other = kb.create_task(conn, title='other profile', assignee='b')
            notify.add_notify_sub(conn, task_id=tid, platform='telegram', chat_id='chat')
            a = quota.identity(conn.execute('SELECT * FROM tasks WHERE id=?', (tid,)).fetchone())
            b = quota.identity(conn.execute('SELECT * FROM tasks WHERE id=?', (other,)).fetchone())
            assert a != b
            from agent.secret_scope import get_secret
            for name in ('a', 'b', 'a'):
                with dispatch._worker_profile_scope(str(home / 'profiles' / name)):
                    assert get_secret('OPENAI_API_KEY') == f'canary-{name}'
            assert quota.identity(conn.execute('SELECT * FROM tasks WHERE id=?', (tid,)).fetchone()) == a
            assert spawn_failure(conn, tid, monkeypatch).reason == FailoverReason.billing
            # Guard is visible before the dead-worker sweep and from a second connection.
            with kbc.connect() as second:
                assert kb.claim_task(second, sibling) is None
            launched = []
            tick(conn, launched)
            assert set(launched) == {healthy, other}
            assert kb.get_task(conn, tid).status == 'blocked'
            assert kb.get_task(conn, sibling).status == 'blocked'
            # Completed tool work is never requeued when another provider is parked.
            marker = home / 'tool-effect'
            marker.write_text('written once')
            kb.complete_task(conn, healthy, summary='tool write complete')
            kb.complete_task(conn, other, summary='healthy profile complete')
        # Reopen the DB every tick: no agent cooldown or connection-local state can protect it.
        for _ in range(334):
            clock[0] += 361
            with kbc.connect() as conn:
                tick(conn, launched)
                assert kb.get_task(conn, tid).status == 'blocked'
        assert tid not in launched and sibling not in launched
        assert launched.count(healthy) == 1 and launched.count(other) == 1
        assert marker.read_text() == 'written once'
        with kbc.connect() as conn:
            assert conn.execute('SELECT count(*) FROM task_runs WHERE task_id=?', (tid,)).fetchone()[0] == 1
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='gave_up'", (tid,)).fetchone()[0] == 1
            first = notify.claim_unseen_events_for_sub(conn, task_id=tid, platform='telegram', chat_id='chat', kinds=['gave_up'])[2]
            second = notify.claim_unseen_events_for_sub(conn, task_id=tid, platform='telegram', chat_id='chat', kinds=['gave_up'])[2]
            assert len(first) == 1 and second == []
            from gateway.kanban_watchers_notifier import _KanbanNotification
            n = SimpleNamespace(task=kb.get_task(conn, tid), head='task', task_id=tid)
            text = _KanbanNotification.format_event(n, first[0])
            assert 'automatically' not in text and 'retry' not in text.lower()
            text = _KanbanNotification.format_event(n, SimpleNamespace(kind='crashed', payload={}))
            assert 'retry' not in text.lower()
            persisted = [tuple(row) for row in conn.execute('SELECT * FROM provider_circuits')]
            persisted += [tuple(row) for row in conn.execute('SELECT metadata FROM task_runs')]
            assert 'canary-' not in str(persisted)
            # Operator recovery clears only the task's relevant provider guard.
            assert kb.unblock_task(conn, tid)
            barrier = threading.Barrier(2)
            def claim():
                with kbc.connect() as connection:
                    barrier.wait()
                    return kb.claim_task(connection, tid) is not None
            with ThreadPoolExecutor(2) as pool:
                results = list(pool.map(lambda _: claim(), range(2)))
            assert sorted(results) == [False, True]
    finally:
        reset_multiplex_context(token)


@pytest.mark.parametrize('recovery', ['reset', 'reset_success', 'reset_date', 'reset_probe_switch', 'switch', '429', 'backoff', 'transient_switch'])
def test_finite_retry_and_safe_recovery(lab, monkeypatch, recovery):
    _, clock = lab
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='recover', assignee='a')
        launched = []
        if recovery == 'backoff':
            conn.execute('UPDATE tasks SET max_retries=8 WHERE id=?', (tid,))
            conn.commit()
            delays = []
            for _ in range(8):
                spawn_failure(conn, tid, monkeypatch, error(429, 'rate_limit_exceeded'))
                tick(conn, launched)
                meta = kb._json_dict(conn.execute('SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1', (tid,)).fetchone()[0])
                delays.append(meta['retry_at'] - clock[0])
                clock[0] = meta['retry_at']
            assert delays == sorted(delays) and max(delays) <= 3600
            assert delays[0] < delays[-1] and delays[-1] == delays[-2]
            assert kb.get_task(conn, tid).status == 'blocked'
            assert launched == []
        elif recovery in {'429', 'transient_switch'}:
            verdict = spawn_failure(conn, tid, monkeypatch, error(429, 'rate_limit_exceeded', 900))
            assert verdict.reason == FailoverReason.rate_limit
            assert conn.execute('SELECT count(*) FROM provider_circuits').fetchone()[0] == 0
            tick(conn, launched)
            assert kb.get_task(conn, tid).consecutive_failures == 1
            assert kb.claim_task(conn, tid) is None
            clock[0] += 899
            tick(conn, launched)
            assert launched == []
            clock[0] += 1
            spawn_failure(conn, tid, monkeypatch, error(429, 'rate_limit_exceeded', 900))
            tick(conn, launched)
            assert kb.get_task(conn, tid).status == 'blocked'
            for _ in range(334):
                clock[0] += 361
                tick(conn, launched)
            assert launched == []
            if recovery == 'transient_switch':
                kb.set_model_override(conn, tid, 'gpt-5', 'openai-codex')
                tick(conn, launched)
                assert launched == [tid]
            else:
                assert kb.unblock_task(conn, tid)
                assert kb.claim_task(conn, tid) is not None
        else:
            retry = 900 if recovery.startswith('reset') else None
            if recovery == 'reset_date':
                from datetime import datetime, timezone
                from email.utils import format_datetime
                retry = format_datetime(datetime.fromtimestamp(clock[0] + 900, timezone.utc), usegmt=True)
            spawn_failure(conn, tid, monkeypatch, error(retry=retry))
            tick(conn, launched)
            assert kb.get_task(conn, tid).status == 'blocked'
            if recovery == 'switch':
                kb.set_model_override(conn, tid, 'gpt-5', 'openai-codex')
                tick(conn, launched)
                assert launched == [tid]
                sibling = kb.create_task(conn, title='old provider still guarded', assignee='a')
                assert kb.claim_task(conn, sibling) is None
            else:
                clock[0] += 899
                tick(conn, launched)
                assert launched == []
                clock[0] += 1
                kb.recompute_ready(conn)
                sibling = kb.create_task(conn, title='one reset probe only', assignee='a')
                if recovery == 'reset_probe_switch':
                    spawn_failure(conn, tid, monkeypatch, error(429, 'rate_limit_exceeded'))
                    tick(conn, launched)
                    kb.set_model_override(conn, tid, 'gpt-5', 'openai-codex')
                    kb.recompute_ready(conn)
                    assert kb.claim_task(conn, tid) is not None
                    kb.set_model_override(conn, tid, 'grok', 'xai-oauth')
                    kb.complete_task(conn, tid, summary='finished on switched provider')
                    kb.recompute_ready(conn)
                    assert kb.claim_task(conn, sibling) is None
                    assert kb.get_task(conn, sibling).status == 'blocked'
                    assert conn.execute('SELECT count(*) FROM provider_circuits').fetchone()[0] == 1
                    return
                if recovery == 'reset_success':
                    assert kb.claim_task(conn, tid) is not None
                    assert kb.claim_task(conn, sibling) is None
                    kb.complete_task(conn, tid, summary='probe succeeded')
                    kb.recompute_ready(conn)
                    assert kb.claim_task(conn, sibling) is not None
                    return
                spawn_failure(conn, tid, monkeypatch, error(retry=900))
                assert kb.claim_task(conn, sibling) is None
                tick(conn, launched)
                for _ in range(334):
                    clock[0] += 361
                    tick(conn, launched)
                assert launched == []
        # Ordinary forbidden and ambiguous billing must never bench a provider.
        forbidden = classify_api_error(error(403, 'forbidden'), provider='xai-oauth')
        assert forbidden.is_auth
        foreign = classify_api_error(error(), provider='openrouter')
        assert foreign.is_auth
        ambiguous = SimpleNamespace(reason=FailoverReason.billing, billing_unverified=True, error_context={})
        assert not quota.failure_evidence(SimpleNamespace(provider='anthropic'), ambiguous, error())['hard_quota']
        assert not quota.failure_evidence(SimpleNamespace(provider='xai-oauth'), forbidden, error(403, 'forbidden'))['hard_quota']


def test_legacy_spending_limit_log_parks_after_one_dispatch(lab):
    _, clock = lab
    launched = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='legacy incident worker', assignee='a')
        def spawn(task, _):
            launched.append(task.id)
            log = kb.worker_log_path(task.id)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("Error code: 403 - {'error': {'code': 'personal-team-blocked:spending-limit'}}\n"
                           '[kanban-worker-exit] rc=75\n')
            return 900002
        for _ in range(334):
            dispatch.dispatch_once(conn, spawn_fn=spawn, max_in_progress=100)
            clock[0] += 361
        assert launched == [tid]
        assert kb.get_task(conn, tid).status == 'blocked'
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='gave_up'", (tid,)).fetchone()[0] == 1


@pytest.mark.parametrize('case', ['custom', 'wave', 'switch_during_run', 'adoption', 'profile_retry', 'tool_results', 'bare_custom'])
def test_route_and_recovery_interleavings(lab, monkeypatch, case):
    home, clock = lab
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=case, assignee='a')
        sibling = kb.create_task(conn, title='sibling', assignee='a')
        if case == 'custom':
            cfg = home / 'profiles' / 'a' / 'config.yaml'
            cfg.write_text('model:\n  provider: custom:alpha\n  default: fake\n'
                           'providers:\n  alpha:\n    base_url: https://blocked.invalid/v1\n'
                           '  beta:\n    base_url: https://healthy.invalid/v1\n')
            task = kb.claim_task(conn, tid)
            monkeypatch.setenv('HERMES_KANBAN_TASK', tid)
            monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(task.current_run_id))
            monkeypatch.setenv('HERMES_KANBAN_CLAIM_LOCK', task.claim_lock)
            quota.record_worker_result({'provider_failure': {'hard_quota': True, 'provider': 'custom',
                 'endpoint': quota._endpoint('https://blocked.invalid/v1'), 'retry_at': None}})
            assert kb.claim_task(conn, sibling) is None
            kb.set_model_override(conn, sibling, 'fake', 'custom:beta')
            kb.recompute_ready(conn)
            assert kb.claim_task(conn, sibling) is not None
        elif case == 'bare_custom':
            cfg = home / 'profiles' / 'a' / 'config.yaml'
            cfg.write_text('model:\n  provider: custom\n  default: fake\n  base_url: https://blocked.invalid/v1\n')
            task = kb.claim_task(conn, tid)
            monkeypatch.setenv('HERMES_KANBAN_TASK', tid)
            monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(task.current_run_id))
            monkeypatch.setenv('HERMES_KANBAN_CLAIM_LOCK', task.claim_lock)
            quota.record_worker_result({'provider_failure': {'hard_quota': True, 'provider': 'custom',
                'endpoint': quota._endpoint('https://blocked.invalid/v1')}})
            assert kb.claim_task(conn, sibling) is None
            cfg.write_text('model:\n  provider: custom\n  default: fake\n  base_url: https://healthy.invalid/v1\n')
            kb.recompute_ready(conn)
            assert kb.claim_task(conn, sibling) is not None
        elif case == 'tool_results':
            task = kb.claim_task(conn, tid)
            monkeypatch.setenv('HERMES_KANBAN_TASK', tid)
            monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(task.current_run_id))
            monkeypatch.setenv('HERMES_KANBAN_CLAIM_LOCK', task.claim_lock)
            quota.record_worker_result({'provider_failure': quota.failure_evidence(SimpleNamespace(provider='xai-oauth'), classify_api_error(error(retry=900), provider='xai-oauth'), error(retry=900)), 'messages': [{'role':'tool','content':'already written'}]})
            conn.execute('UPDATE tasks SET worker_pid=?, started_at=1 WHERE id=?', (900011, tid))
            conn.commit()
            dispatch._record_worker_exit(900011, 75 << 8)
            dispatch.detect_crashed_workers(conn)
            clock[0] += 10000
            kb.recompute_ready(conn)
            assert kb.get_task(conn, tid).status == 'blocked'
            assert kb.claim_task(conn, tid) is None
        elif case == 'wave':
            second = kb.claim_task(conn, sibling)
            spawn_failure(conn, tid, monkeypatch, error(retry=900))
            monkeypatch.setenv('HERMES_KANBAN_TASK', sibling)
            monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(second.current_run_id))
            monkeypatch.setenv('HERMES_KANBAN_CLAIM_LOCK', second.claim_lock)
            quota.record_worker_result({'provider_failure': quota.failure_evidence(SimpleNamespace(provider='xai-oauth'), classify_api_error(error(retry=900), provider='xai-oauth'), error(retry=900))})
            assert conn.execute('SELECT retry_at FROM provider_circuits').fetchone()[0] == clock[0] + 900
            clock[0] += 900
            candidates = [kb.create_task(conn, title='probe', assignee='a') for _ in range(2)]
            barrier = threading.Barrier(2)
            def claim(candidate):
                with kbc.connect() as connection:
                    barrier.wait()
                    return kb.claim_task(connection, candidate) is not None
            with ThreadPoolExecutor(2) as pool:
                assert sorted(pool.map(claim, candidates)) == [False, True]
        elif case == 'switch_during_run':
            task = kb.claim_task(conn, tid)
            kb.set_model_override(conn, tid, 'gpt-5', 'openai-codex')
            monkeypatch.setenv('HERMES_KANBAN_TASK', tid)
            monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(task.current_run_id))
            monkeypatch.setenv('HERMES_KANBAN_CLAIM_LOCK', task.claim_lock)
            quota.record_worker_result({'provider_failure': quota.failure_evidence(SimpleNamespace(provider='xai-oauth'), classify_api_error(error(), provider='xai-oauth'), error())})
            conn.execute('UPDATE tasks SET worker_pid=?, started_at=1 WHERE id=?', (900009, tid))
            conn.commit()
            dispatch._record_worker_exit(900009, 75 << 8)
            dispatch.detect_crashed_workers(conn)
            kb.recompute_ready(conn)
            assert kb.claim_task(conn, tid) is not None
            assert kb.claim_task(conn, sibling) is None
        elif case == 'adoption':
            task = kb.claim_task(conn, tid)
            # Pre-upgrade ended rate-limit run: no immutable route/evidence metadata.
            conn.execute("UPDATE tasks SET status='ready',claim_lock=NULL,current_run_id=NULL WHERE id=?", (tid,))
            conn.execute("UPDATE task_runs SET outcome='rate_limited',ended_at=?,metadata=NULL WHERE id=?", (clock[0], task.current_run_id))
            conn.commit()
            log = kb.worker_log_path(tid)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("Error code: 403 - personal-team-blocked:spending-limit\n[kanban-worker-exit] rc=75\n")
            assert kb.claim_task(conn, sibling) is None
            assert kb.get_task(conn, tid).status == 'blocked'
            assert conn.execute('SELECT count(*) FROM provider_circuits').fetchone()[0] == 1
        else:
            spawn_failure(conn, tid, monkeypatch, error(429, 'rate_limit_exceeded', 900))
            dispatch.detect_crashed_workers(conn)
            assert kb.claim_task(conn, tid) is None
            kb.assign_task(conn, tid, 'b')
            assert dispatch.check_respawn_guard(conn, tid) is None
            task = kb.claim_task(conn, tid)
            assert task is not None
