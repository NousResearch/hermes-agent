#!/usr/bin/env python3
"""Rebuild the pinned public Q/core/Input proof in an absent destination directory."""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

PUBLIC = 'https://github.com/dokterdok/hermes-agent.git'
BASE = '485d5f6848c2598ca00fa7704e50e8ae66d0983a'
Q = '43b9f02183444de0d48dc72170341346d58e09dc'
CORE = '2fb90a347b5a0d7864367b221544dc058c6cf70c'
CORE_PUBLIC_TIP = '03ab73504dae640e8cfef0b19841c7053aa31cbe'
COMMON_BASE = 'b36398f71929d6069905ccca59af8d1516c5ee3a'
INPUT = 'c061ac2cad7e699823b1f5166be795298e64b994'
INPUT_PARENT = 'ccbd43704e4ab12ae32586075831d7c45338bca6'
INPUT_TREE = '2639c4f767717442f9b207afe0f8cb0b7f0c884e'
PATCH_SHA256 = '6b9dd749ce20bea533427225bdc5b0b3d389b6c4a85177b74e34ce4d94a5300d'
EXPECTED_TREE = '5ff1604d08aa31d9286a79c443af94a92a7c22b1'
CONFLICTS = {'gateway/run_runtime.py', 'gateway/session_hosted_rpc.py', 'hermes_state_runtime.py'}
OPEN, MID, END = '<' * 7, '=' * 7, '>' * 7


def run(repo, *args, input_bytes=None, allow_failure=False, timeout=110):
    command = ['git', '-C', str(repo), *args] if repo else ['git', *args]
    print('+', ' '.join(command), flush=True)
    result = subprocess.run(command, input=input_bytes, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, env=GIT_ENV)
    if not allow_failure and result.returncode:
        raise RuntimeError(f"{command!r} exited {result.returncode}: {result.stderr.decode(errors='replace')[-2000:]}")
    if result.stderr:
        print(result.stderr.decode(errors='replace')[-2000:], file=sys.stderr)
    return result.stdout, result.returncode


def value(repo, *args):
    return run(repo, *args)[0].decode().strip()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def replace_exact(path, old, new):
    content = path.read_text()
    require(content.count(old) == 1, f'Unexpected conflict geometry: {path}: {old[:60]!r}')
    path.write_text(content.replace(old, new))


def replay(destination):
    require(not destination.exists(), f'Destination exists: {destination}')
    require(destination.parent.is_dir(), f'Create parent first: {destination.parent}')
    run(None, 'init', str(destination))
    require(not (destination / '.git/objects/info/alternates').exists(), 'Alternates forbidden')
    require(value(destination, 'remote') == '', 'Unexpected local remote')
    # Fetch only exact public objects. Depth 10 exposes Q's eight-commit path to
    # the common base; fetching the base alone shallowly cannot prove ancestry.
    pins = (BASE, Q, CORE, CORE_PUBLIC_TIP, COMMON_BASE, INPUT, INPUT_PARENT)
    require(all(re.fullmatch(r'[0-9a-f]{40}', pin) for pin in pins), 'Invalid pin syntax')
    run(destination, '-c', 'protocol.version=2', 'fetch', '--no-tags', '--depth=1', PUBLIC, *pins)
    run(destination, '-c', 'protocol.version=2', 'fetch', '--no-tags', '--depth=10', PUBLIC, Q, CORE)
    for pin in pins:
        require(value(destination, 'cat-file', '-t', pin) == 'commit', f'Missing public commit {pin}')
    require(value(destination, 'merge-base', Q, CORE) == COMMON_BASE, 'Common base not proven')
    headers = value(destination, 'cat-file', '-p', INPUT).split('\n\n', 1)[0].splitlines()
    require([line[7:] for line in headers if line.startswith('parent ')] == [INPUT_PARENT],
            'Input parent changed')
    require(value(destination, 'rev-parse', f'{INPUT}^{{tree}}') == INPUT_TREE, 'Input tree changed')
    require(value(destination, 'rev-parse', f'{CORE_PUBLIC_TIP}^{{tree}}'), 'Missing public core tip tree')
    require(not (destination / '.git/objects/info/alternates').exists(), 'Alternates forbidden')
    require(value(destination, 'remote') == '', 'Local remote forbidden')
    partial, rc = run(destination, 'config', '--local', '--get', 'extensions.partialClone',
                      allow_failure=True)
    require(rc == 1 and not partial, 'Promisor repository forbidden')
    patch, _ = run(destination, 'diff', '--binary', '--abbrev=11', BASE, INPUT)
    require(hashlib.sha256(patch).hexdigest() == PATCH_SHA256, 'Published Input delta differs')
    run(destination, 'config', 'user.name', 'David Dudok de Wit')
    run(destination, 'config', 'user.email', 'david@dudokdewit.net')
    run(destination, 'switch', '-c', 'proof/input-declared', Q)
    run(destination, 'merge', '--no-ff', '--no-commit', CORE)
    run(destination, 'commit', '-m', 'proof: merge pinned core into pinned Q')
    _, rc = run(destination, 'apply', '--3way', '-', input_bytes=patch, allow_failure=True)
    require(rc == 1, f'Unexpected Input patch status {rc}')
    conflicts = set(value(destination, 'diff', '--name-only', '--diff-filter=U').splitlines())
    require(conflicts == CONFLICTS, f'Unexpected conflict paths: {conflicts}')
    replace_exact(destination / 'gateway/run_runtime.py', f'{OPEN} ours\n{MID}\n', '')
    replace_exact(destination / 'gateway/run_runtime.py',
                  f'{END} theirs\nasync def initialize_gateway_runtime',
                  '\nasync def initialize_gateway_runtime')
    replace_exact(destination / 'gateway/session_hosted_rpc.py', f'''{OPEN} ours
        from gateway.session_hosted_attachments import submission_payload
        payload = await asyncio.to_thread(
            submission_payload, self, params['prompt'], params.get('attachments'))
        if self.authorizer('submit', task, generation) is not True:
            raise RuntimeStoreError('permission_denied')
{MID}
        prepared = await asyncio.to_thread(prepare_hosted_input, self, request_id=request_id,
            prompt=params['prompt'], attachments=params.get('attachments'))
{END} theirs''', '''        prepared = await asyncio.to_thread(prepare_hosted_input, self, request_id=request_id,
            prompt=params['prompt'], attachments=params.get('attachments'))
        if self.authorizer('submit', task, generation) is not True:
            raise RuntimeStoreError('permission_denied')''')
    replace_exact(destination / 'hermes_state_runtime.py', f'''{OPEN} ours
                        _authorize_write=None) -> dict:
    """Admit input; the trusted private guard raises to refuse a NEW write.

    The guard receives the owning transaction connection, must not commit it or
    perform external effects, and may run again on SQLite retry. Exact existing
    and terminal replays bypass it: they cannot create or change accepted work.
    """
{MID}
                        input_custody=None) -> dict:
{END} theirs''', '''                        input_custody=None, _authorize_write=None) -> dict:
    """Admit input; the trusted private guard raises to refuse a NEW write.

    The guard receives the owning transaction connection, must not commit it or
    perform external effects, and may run again on SQLite retry. Exact existing
    and terminal replays bypass it: they cannot create or change accepted work.
    """''')
    replace_exact(destination / 'hermes_state_runtime.py', f'''{OPEN} ours
        if _authorize_write is not None:
            _authorize_write(conn)
{MID}
        if input_custody is not None:
            from hermes_state_input_custody import AcceptedInputHandle
            if isinstance(input_custody, AcceptedInputHandle):
                raise RuntimeStoreError('admission_conflict')
{END} theirs''', '''        if input_custody is not None:
            from hermes_state_input_custody import AcceptedInputHandle
            if isinstance(input_custody, AcceptedInputHandle):
                raise RuntimeStoreError('admission_conflict')
        if _authorize_write is not None:
            _authorize_write(conn)''')
    run(destination, 'add', *sorted(CONFLICTS))
    run(destination, 'diff', '--cached', '--check')
    run(destination, 'commit', '-m', 'proof: reconcile public Input with Q and core')
    tree = value(destination, 'rev-parse', 'HEAD^{tree}')
    require(tree == EXPECTED_TREE, f'Composed tree differs: {tree}')
    require(value(destination, 'status', '--porcelain') == '', 'Replay checkout not clean')
    require(not (destination / '.git/objects/info/alternates').exists(), 'Alternates forbidden')
    require(not list((destination / '.git/objects/pack').glob('*.promisor')), 'Promisor pack forbidden')
    print(f'PUBLIC PINNED COMPOSITION: tree {tree}; clean; no alternates or promisor packs')
    print(f'Input {INPUT}; Q {Q}; core {CORE}; local proof commits are synthetic')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: python3 scripts/replay_input_declared_composition.py ABSENT_DESTINATION')
    # Ignore ambient Git object/config overrides rather than trusting local checkout state.
    GIT_ENV = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    GIT_ENV.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_TERMINAL_PROMPT='0', GIT_NO_REPLACE_OBJECTS='1')
    replay(Path(sys.argv[1]).resolve())
