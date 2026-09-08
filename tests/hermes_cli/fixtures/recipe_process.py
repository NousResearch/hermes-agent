"""Disposable recipe E2E processes; never starts Hermes or an LLM worker.

The CLI mode uses the real shared argparse builder + kanban_command. Faults
are process-local monkeypatches, not alternate task/recipe implementations.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def wait_file(path):
    deadline = time.monotonic() + 30
    while not Path(path).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(path)
        time.sleep(0.01)


def publish(path, value):
    path = Path(path)
    staging = path.with_suffix('.tmp')
    staging.write_text(json.dumps(value))
    staging.replace(path)


def snapshot(conn):
    tables = ('tasks', 'task_links', 'task_events', 'task_runs',
              'recipe_instances', 'recipe_instance_tasks')
    return {table: [dict(row) for row in conn.execute(f'SELECT * FROM {table}')]
            for table in tables}


def cli(options, words):
    from hermes_cli import kanban, kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import kanban_recipes_store as store

    if options.get('mutate_outer_transaction'):
        original_txn = kbc.write_txn

        def only_remove_recipe_outer(conn, **kwargs):
            # Native constructor savepoints/transactions remain untouched.
            if sys._getframe(1).f_code is store.instantiate.__code__:
                return nullcontext(conn)
            return original_txn(conn, **kwargs)

        kbc.write_txn = only_remove_recipe_outer

    def pause():
        Path(options['paused']).touch()
        wait_file(options['release'])

    if options.get('pause') == 'before':
        original_create = kb.create_task
        paused = False

        def create_then_pause(*args, **kwargs):
            nonlocal paused
            result = original_create(*args, **kwargs)
            if not paused:
                paused = True
                pause()
            return result

        kb.create_task = create_then_pause
    elif options.get('pause') == 'after':
        original_instantiate = store.instantiate

        def commit_then_pause(*args, **kwargs):
            result = original_instantiate(*args, **kwargs)
            pause()
            return result

        store.instantiate = commit_then_pause
    if options.get('start'):
        Path(options['ready']).touch()
        wait_file(options['start'])
    parser = argparse.ArgumentParser()
    kanban.build_parser(parser.add_subparsers())
    return kanban.kanban_command(parser.parse_args(['kanban', *words])) or 0


def worker(options):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    wait_file(options['release'])
    with kbc.connect_closing(Path(options['db'])) as conn:
        task_id, run_id = options['task'], options['run']
        task = kb.get_task(conn, task_id)
        assert task.current_run_id == run_id and task.claim_lock
        action = options['action']
        if action == 'review':
            ok = kb.request_review(conn, task_id, reviewer='reviewer',
                                   summary='Local implementation', expected_run_id=run_id)
        elif action == 'rereview':
            ok = kb.request_review(conn, task_id, summary='Local correction',
                                   expected_run_id=run_id)
        elif action == 'changes':
            ok, routed = kb.request_changes(conn, task_id, reason='Correct local fixture',
                                            expected_run_id=run_id)
            assert routed == 'default'
        else:
            assert action == 'complete'
            # A stale owner must not complete the current run.
            assert not kb.complete_task(conn, task_id, expected_run_id=run_id + 100000)
            ok = kb.complete_task(conn, task_id, summary='Harmless local child completed',
                                  expected_run_id=run_id)
        assert ok
        print(json.dumps({'pid': os.getpid(), 'task': task_id, 'run': run_id,
                          'action': action, 'status': kb.get_task(conn, task_id).status}), flush=True)
    return 0


def observer(options):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli.profiles import resolve_profile_env

    # Connect/migrate BEFORE the creator holds its transaction. Reconnecting
    # during the pause would wait on init's write lock instead of observing WAL.
    with kbc.connect_closing(Path(options['db'])) as conn:
        publish(options['ready'], {'pid': os.getpid()})
        for line in sys.stdin:
            command = json.loads(line)
            if command['action'] == 'snapshot':
                publish(command['output'], snapshot(conn))
                continue
            assert command['action'] == 'dispatch'
            children = []
            records = []
            release = Path(command['output'] + '.workers-release')

            def spawn(task, workspace, *, board=None):
                home = resolve_profile_env(task.assignee)
                argv = dispatch._worker_argv(task, task.assignee, home)
                child_env = os.environ.copy()
                child_env.update(HERMES_HOME=home, HERMES_PROFILE=task.assignee,
                                 HERMES_KANBAN_DB=options['db'], HERMES_KANBAN_BOARD='default',
                                 HERMES_KANBAN_TASK=task.id,
                                 HERMES_KANBAN_RUN_ID=str(task.current_run_id),
                                 HERMES_KANBAN_CLAIM_LOCK=task.claim_lock)
                spec = {'db': options['db'], 'release': str(release), 'task': task.id,
                        'run': task.current_run_id,
                        'action': command.get('actions', {}).get(task.id, 'complete')}
                child = subprocess.Popen([sys.executable, __file__, 'worker', json.dumps(spec)],
                                         cwd=workspace, env=child_env, text=True,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                children.append(child)
                git_branch = None
                if task.workspace_kind == 'worktree':
                    assert (Path(workspace) / '.git').is_file()
                    git_branch = subprocess.run(
                        ['git', '-C', workspace, 'branch', '--show-current'],
                        check=True, capture_output=True, text=True, timeout=10,
                    ).stdout.strip()
                records.append({'task': task.id, 'run': task.current_run_id,
                                'git_branch_at_spawn': git_branch,
                                'pid': child.pid, 'assignee': task.assignee, 'argv': argv,
                                'workspace': workspace, 'skills': task.skills,
                                'goal_mode': task.goal_mode, 'goal_max_turns': task.goal_max_turns,
                                'model': task.model_override, 'provider': task.provider_override,
                                'reasoning': task.reasoning_effort,
                                'visible_at_spawn': snapshot(conn)})
                return child.pid

            # This receipt proves the real dispatcher entered BEGIN IMMEDIATE
            # on its pre-opened connection while the CLI writer was paused.
            def trace(sql):
                if sql == 'BEGIN IMMEDIATE':
                    Path(command['output'] + '.entered').touch()

            conn.set_trace_callback(trace)
            try:
                before = snapshot(conn)
                result = dispatch.dispatch_once(conn, spawn_fn=spawn, board='default',
                                                reconcile_orphans=False, max_spawn=32)
                claimed = snapshot(conn)
                release.touch()
                outputs = []
                for child in children:
                    stdout, stderr = child.communicate(timeout=25)
                    assert child.returncode == 0, (stdout, stderr)
                    outputs.append(json.loads(stdout))
                publish(command['output'], {'before': before, 'claimed': claimed,
                                            'after': snapshot(conn), 'workers': outputs,
                                            'records': records, 'spawned': result.spawned})
            finally:
                conn.set_trace_callback(None)
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.communicate(timeout=5)
    return 0


if __name__ == '__main__':
    mode, raw, *words = sys.argv[1:]
    options = json.loads(raw)
    sys.exit({'cli': lambda: cli(options, words),
              'worker': lambda: worker(options),
              'observer': lambda: observer(options)}[mode]())
