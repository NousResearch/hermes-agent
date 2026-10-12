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
            circuit = conn.execute('SELECT * FROM provider_circuits').fetchone()
            claimed = kb.get_task(conn, circuit['probe_task'])
            assert circuit['probe_run_id'] == claimed.current_run_id
            assert kb.get_run(conn, circuit['probe_run_id']).status == 'running'
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


@pytest.mark.parametrize('lane', ['ready', 'review'])
@pytest.mark.parametrize('recovery', ['direct', 'restart_recompute'])
@pytest.mark.parametrize('result', ['genuine', 'scope', 'backend', 'missing_route',
                                   'unended', 'nonterminal', 'failed', 'successor', 'hard_failure'])
def test_reserved_probe_run_is_the_only_recovery_witness(lab, monkeypatch, lane, recovery, result):
    """Regression for PR136166: historical/successor successes cannot prove a probe."""
    home, clock = lab
    db_path = home / 'board.db'
    claim = kb.claim_review_task if lane == 'review' else kb.claim_task
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='historical success then quota', assignee='a')
        historical = kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, summary='historical success')
        # Seed the supported done -> ready/review reopening state.
        with kb.write_txn(conn):
            conn.execute('UPDATE tasks SET status=?,completed_at=NULL WHERE id=?', (lane, tid))
        failed = claim(conn, tid)
        exc = error(retry=900)
        verdict = classify_api_error(exc, provider='xai-oauth')
        evidence = quota.failure_evidence(SimpleNamespace(provider='xai-oauth'), verdict, exc)
        assert evidence['hard_quota']
        monkeypatch.setenv('HERMES_KANBAN_TASK', tid)
        monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(failed.current_run_id))
        monkeypatch.setenv('HERMES_KANBAN_CLAIM_LOCK', failed.claim_lock)
        quota.record_worker_result({'provider_failure': evidence})
        with kb.write_txn(conn):
            conn.execute('UPDATE tasks SET worker_pid=?,started_at=1 WHERE id=?', (900021, tid))
        dispatch._record_worker_exit(900021, 75 << 8)
        dispatch.detect_crashed_workers(conn)
        assert kb.get_task(conn, tid).status == 'blocked'
        clock[0] += 900
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == lane
        if result == 'genuine':
            # Failure after opening/binding must roll back the entire reservation.
            append = kb._append_event
            def fail_claim_event(connection, task_id, kind, *args, **kwargs):
                if kind == 'claimed':
                    raise RuntimeError('claim event failure')
                return append(connection, task_id, kind, *args, **kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(kb, '_append_event', fail_claim_event)
                with pytest.raises(RuntimeError, match='claim event failure'):
                    claim(conn, tid)
            with kbc.connect_closing() as observer:
                circuit = observer.execute('SELECT * FROM provider_circuits').fetchone()
                assert circuit['retry_at'] == clock[0] and circuit['probe_task'] is None
                assert circuit['probe_run_id'] is None
                assert kb.get_task(observer, tid).status == lane
                assert len(kb.list_runs(observer, tid)) == 2
        probe = claim(conn, tid)
        assert probe is not None
        probe_id = probe.current_run_id
        assert historical.current_run_id < failed.current_run_id < probe_id

    def check_gate(expect_open):
        # Reopen through full schema initialization to mimic a fresh dispatcher.
        if recovery == 'restart_recompute':
            kbc._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with kbc.connect_closing() as connection:
            candidate = kb.create_task(connection, title='same route candidate', assignee='a')
            if lane == 'review':
                with kb.write_txn(connection):
                    connection.execute("UPDATE tasks SET status='review' WHERE id=?", (candidate,))
            if recovery == 'restart_recompute':
                kb.recompute_ready(connection)
            assert (claim(connection, candidate) is not None) == expect_open
            circuit = connection.execute('SELECT * FROM provider_circuits').fetchone()
            if expect_open:
                assert circuit is None
            else:
                assert circuit['probe_task'] == tid
                assert circuit['probe_run_id'] == probe_id

    # The previous head opens this gate using the historical completed run.
    check_gate(False)
    with kbc.connect_closing() as conn:
        run = kb.get_run(conn, probe_id)
        assert run.outcome is None and run.ended_at is None and run.status == 'running'
        if result == 'hard_failure':
            monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(probe_id))
            monkeypatch.setenv('HERMES_KANBAN_CLAIM_LOCK', probe.claim_lock)
            quota.record_worker_result({'provider_failure': evidence})
            circuit = conn.execute('SELECT * FROM provider_circuits').fetchone()
            assert circuit['attempts'] == 2 and circuit['retry_at'] is None
            assert circuit['probe_task'] is None and circuit['probe_run_id'] is None
            kb.recompute_ready(conn)
            candidate = kb.create_task(conn, title='failed hard probe stays closed', assignee='a')
            assert kb.claim_task(conn, candidate) is None
            return
        with kb.write_txn(conn):
            metadata = kb._json_dict(conn.execute('SELECT metadata FROM task_runs WHERE id=?', (probe_id,)).fetchone()[0])
            if result in {'scope', 'backend'}:
                metadata['quota_route'][result] += '-different'
            elif result == 'missing_route':
                metadata.pop('quota_route')
            conn.execute('UPDATE task_runs SET metadata=? WHERE id=?', (kb._json_or_null(metadata), probe_id))
            # Complete through the production end-run primitive so completion's
            # eager recompute cannot release the gate before each witness is tested.
            kb._end_run(conn, tid, outcome='crashed' if result in {'failed', 'successor'} else 'completed',
                        status='running' if result == 'nonterminal' else 'done')
            conn.execute("UPDATE tasks SET status='done',claim_lock=NULL,claim_expires=NULL WHERE id=?", (tid,))
            if result == 'unended':
                conn.execute('UPDATE task_runs SET ended_at=NULL WHERE id=?', (probe_id,))
            elif result == 'successor':
                # Even a later same-task, same-route terminal success cannot substitute.
                successor = conn.execute("INSERT INTO task_runs(task_id,profile,status,outcome,started_at,ended_at,metadata) "
                                         "SELECT task_id,profile,'done','completed',started_at,ended_at,metadata "
                                         "FROM task_runs WHERE id=?", (probe_id,)).lastrowid
                # A hard failure attributed to that successor must not consume
                # the original reservation either (same task is insufficient).
                row = dict(conn.execute('SELECT * FROM tasks WHERE id=?', (tid,)).fetchone())
                row['current_run_id'] = successor
                quota.arm(conn, row, evidence, metadata)
                circuit = conn.execute('SELECT * FROM provider_circuits').fetchone()
                assert circuit['attempts'] == 1 and circuit['probe_run_id'] == probe_id
    check_gate(result == 'genuine')


@pytest.mark.parametrize('legacy_retry', [None, 'elapsed'])
def test_legacy_reserved_circuit_migration_never_infers_a_probe_run(lab, legacy_retry):
    home, clock = lab
    db_path = home / 'board.db'
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title='old successful reservation', assignee='a')
        success = kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, summary='old success')
        row = conn.execute('SELECT * FROM tasks WHERE id=?', (tid,)).fetchone()
        scope, backend = quota.identity(row)
        # Recreate only the previous circuit schema; all ledger/wait data remains real.
        conn.execute('DROP TABLE provider_circuits')
        conn.execute('CREATE TABLE provider_circuits(scope TEXT NOT NULL,backend TEXT NOT NULL,'
                     'retry_at REAL,attempts INTEGER NOT NULL DEFAULT 1,probe_task TEXT,PRIMARY KEY(scope,backend))')
        retry_at = None if legacy_retry is None else clock[0] - 1
        conn.execute('INSERT INTO provider_circuits VALUES(?,?,?,?,?)', (scope, backend, retry_at, 1, tid))
        sibling = kb.create_task(conn, title='legacy waiter', assignee='a')
        with kb.write_txn(conn):
            quota.park(conn, conn.execute('SELECT * FROM tasks WHERE id=?', (sibling,)).fetchone(), scope, backend, 'ready')
        before = dict(conn.execute('SELECT * FROM provider_circuits').fetchone())
    # The real initialization/migration path, repeated to prove idempotence.
    for _ in range(2):
        kb.init_db(db_path)
        with kbc.connect_closing() as conn:
            circuit = conn.execute('SELECT * FROM provider_circuits').fetchone()
            assert {key: circuit[key] for key in before} == before
            assert circuit['probe_run_id'] is None
            assert kb.get_run(conn, success.current_run_id).outcome == 'completed'
            kb.recompute_ready(conn)
            assert kb.get_task(conn, sibling).status == 'blocked'
            candidate = kb.create_task(conn, title='no inferred legacy recovery', assignee='a')
            assert kb.claim_task(conn, candidate) is None
            assert conn.execute('SELECT probe_run_id FROM provider_circuits').fetchone()[0] is None
