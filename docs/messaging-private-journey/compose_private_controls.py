#!/usr/bin/env python3
"""Compose public Messaging, selected-route and delegated Route with owner overlays.

For local uncommitted development use --permission-dir and --consumer-dir.
For public reproduction pin both published owner commits instead. The base recipe
joins M73, Permission, selected-route and Messaging; this recipe adds only the
six public Route delegated-control paths and each explicit A3 owner path.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).with_name('compose.py')
MESSAGING = 'd4d9f905e8123eea38ad81c4cdd6ac44257315d8'
ROUTE = '8a29b6d13bc558221130b4216a93f373afa68caf'
ROUTE_FILES = (
    'gateway/hosted_rooms.py', 'gateway/hosted_room_delegated_control.py',
    'gateway/hosted_room_driver.py', 'tui_gateway/hosted_room_service.py',
    'tui_gateway/hosted_room_driver.py', 'tests/tui_gateway/test_hosted_room_delegated_control.py',
)
PERMISSION_FILES = ('gateway/session_group_controls.py',
                    'gateway/session_group_messaging_control.py',
                    'tests/gateway/test_messaging_room_control_binding.py')
CONSUMER_FILES = ('gateway/group_chat_private_send.py',
                  'gateway/group_chat_private_read.py',
                  'gateway/group_chat_private_control.py',
                  'tests/gateway/test_canonical_group_private_control.py')


def git(root, *args):
    return subprocess.check_output(('git', '-C', str(root), *args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--baseline-dir', type=Path,
                        help='existing read-only exact public composition (local fast path)')
    parser.add_argument('--permission-dir', type=Path)
    parser.add_argument('--consumer-dir', type=Path)
    parser.add_argument('--permission-commit')
    parser.add_argument('--consumer-commit')
    args = parser.parse_args()
    if bool(args.permission_dir) == bool(args.permission_commit) or bool(args.consumer_dir) == bool(args.consumer_commit):
        parser.error('supply exactly one directory or commit for each owner')
    for commit in (args.permission_commit, args.consumer_commit):
        if commit is not None and (len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit)):
            parser.error('owner commits must be full SHA-1 pins')
    destination = args.destination.absolute()
    if destination.exists():
        parser.error('destination must be absent')
    if args.baseline_dir:
        baseline = args.baseline_dir.resolve(strict=True)
        manifest = json.loads((baseline / 'RECONSTRUCTION.json').read_text())
        if manifest['candidate'] != MESSAGING or manifest['sources']['permission'] != '34332b47b3fb3c8394879e7179e157b89be115de':
            parser.error('not the exact public baseline')
        shutil.copytree(baseline, destination, symlinks=False, ignore=shutil.ignore_patterns('.git'))
        # A new independent repository provides pinned public Git blobs for both modes.
        git(destination, 'init', '-q')
        git(destination, 'remote', 'add', 'origin', 'https://github.com/dokterdok/hermes-agent.git')
    else:
        subprocess.run((sys.executable, str(BASE), str(destination), '--candidate-commit', MESSAGING), check=True)
        manifest = json.loads((destination / 'RECONSTRUCTION.json').read_text())
    commits = [ROUTE, *(c for c in (args.permission_commit, args.consumer_commit) if c)]
    git(destination, '-c', 'protocol.file.allow=never', 'fetch', '-q', '--no-tags',
        '--filter=blob:none', '--depth=1', 'origin', *commits)
    if git(destination, 'rev-parse', ROUTE + '^{commit}').decode().strip() != ROUTE:
        raise RuntimeError('delegated Route source identity changed')
    # Avoid one lazy-fetch network round trip per selected blob.
    objects = set()
    for commit, paths in ((ROUTE, ROUTE_FILES),
                          (args.permission_commit, PERMISSION_FILES),
                          (args.consumer_commit, CONSUMER_FILES)):
        if commit:
            rows = git(destination, 'ls-tree', '-r', commit, '--', *paths).splitlines()
            if len(rows) != len(paths):
                raise RuntimeError('owner commit is missing a declared path')
            objects.update(row.split()[2].decode('ascii') for row in rows)
    git(destination, '-c', 'protocol.file.allow=never', 'fetch', '-q', '--no-tags',
        'origin', *sorted(objects))
    for path in ROUTE_FILES:
        (destination / path).write_bytes(git(destination, 'show', ROUTE + ':' + path))
    for name, paths, local, commit in (
            ('permission', PERMISSION_FILES, args.permission_dir, args.permission_commit),
            ('consumer', CONSUMER_FILES, args.consumer_dir, args.consumer_commit)):
        for path in paths:
            source = (local.resolve(strict=True) / path) if local else None
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes() if source else git(destination, 'show', commit + ':' + path))
        manifest[name + '_control_source'] = str(local.resolve()) if local else commit
        manifest[name + '_control_paths'] = list(paths)
    manifest['delegated_route_commit'] = ROUTE
    manifest['delegated_route_paths'] = list(ROUTE_FILES)
    (destination / 'RECONSTRUCTION.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print('composed', destination, 'Route', ROUTE)


if __name__ == '__main__':
    main()
