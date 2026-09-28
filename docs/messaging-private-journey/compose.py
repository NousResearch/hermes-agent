#!/usr/bin/env python3
"""Reconstruct a private Messaging candidate atop freshly fetched public owners.

Use --candidate-dir for the six frozen pending files, or --candidate-commit
for their eventual immutable public owner commit. Never use local lower code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REMOTE = 'https://github.com/dokterdok/hermes-agent.git'
SOURCES = {
    'm73': '73fbc700664c56eabd9d7a55f178320662ef0c47',
    'permission': '34332b47b3fb3c8394879e7179e157b89be115de',
    'route': '1fa3c0addd0c3eec671f3019c443dd3e449db134',
    'selected-route': '8c36f8dd3130397c6388d7517d51a0ed8c700279',
}
ROUTE = {
    'gateway/session_hosted_controls.py': '04ed4b1048e2f334eeeede3870c401a891376f75',
    'gateway/session_group_state.py': '8634628ee86af069bfe089d96df737dbb0f2295f',
    'gateway/session_hosted_peer_retry.py': 'b1f2fd854b7ecc3b091284df28d5b8597861e566',
    'gateway/session_group_setup.py': 'e786466786f5816bfef10dda65f71c67df1cd872',
    'gateway/session_group_peers.py': 'f39e15ba973cd8ff3490d829064fbcb4a71f6a75',
    'gateway/hosted_room_grant_state.py': '6c9f73ec7ef6c1cd8329f0b2d926ea8fa365726a',
    'gateway/session_hosted_service.py': 'c35ffede2f71e73066f4a1ef462ee865bf12186c',
    'tui_gateway/hosted_room_service.py': '7a3042ccd9ffc64a50e1e8bd99bb147ef0681707',
    'gateway/hosted_room_link_records.py': '778dbed575dd44c2ae587604eef94045599123ac',
    'tui_gateway/hosted_room_peer_status.py': 'a5c38c086f51a7a10b36c468245835102a663601',
    'tui_gateway/hosted_room_peer_http.py': '5f76a652a0933be9c8375244b3d3684d3bd9c042',
    'tui_gateway/hosted_room_driver.py': 'dfd2a8c766a497c183372b1141a0bfa3cb02a67c',
    'gateway/session_hosted_attachments.py': '04b77957f3b41ca4502bd82cbf371b358e491889',
    'gateway/hosted_rooms.py': '86a366c81c7c4345578554a07a67f27dcbed52fc',
    'gateway/hosted_room_route_schema.py': '04dd05e850e36b8710f2c186ce21b11ec1152dfd',
    'gateway/hosted_rooms_legacy_import.py': '3aae609e073109588e413b025b6f97a802578b89',
    'gateway/hosted_room_attachments.py': 'a8716dc08ba9586bf8744cf2062ba6aebb971715',
}
CANDIDATE = (
    'gateway/group_chat_slash.py',
    'gateway/group_chat_private_send.py',
    'gateway/group_chat_private_read.py',
    'tests/gateway/test_canonical_group_messaging_send.py',
    'tests/gateway/test_canonical_group_private_read.py',
    'tests/gateway/test_canonical_group_private_journey.py',
)
# Published M73 legacy helpers absent from the Permission checkout. This is
# their bounded import closure, not an implementation copied from a proof tree.
LEGACY = tuple('gateway/' + name + '.py' for name in (
    'choice_picker', 'desktop_room_mailbox', 'group_chat_approval_permissions',
    'group_chat_work', 'group_home_consent', 'group_home_identity',
    'hosted_room_approval_rules', 'hosted_room_control_client',
    'hosted_room_controls', 'hosted_room_messaging',
    'hosted_room_messaging_approvals', 'hosted_room_messaging_presentation',
    'hosted_room_messaging_retries',
))
PIN = re.compile(r'[0-9a-f]{40}\Z')
HERE = Path(__file__).resolve().parent


def git(repo, *args):
    done = subprocess.run(['git', '-C', str(repo), *args], capture_output=True)
    if done.returncode:
        raise RuntimeError(f'git {args[:3]} failed: {done.stderr.decode(errors="replace")[-1200:]}')
    return done.stdout


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='absent directory under existing real parents')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--candidate-dir', type=Path)
    group.add_argument('--candidate-commit', type=str)
    args = parser.parse_args()
    if args.candidate_commit and not PIN.fullmatch(args.candidate_commit):
        parser.error('candidate commit must be a full, immutable 40-character Git SHA')
    dest = args.destination.absolute()
    if dest.exists() or dest.is_symlink():
        parser.error('destination must be absent')
    ancestor = dest.parent
    while True:
        if ancestor.is_symlink() or not ancestor.is_dir():
            parser.error('all destination parents must be existing real directories')
        if ancestor == ancestor.parent:
            break
        ancestor = ancestor.parent
    if args.candidate_dir:
        root = args.candidate_dir.resolve(strict=True)
        if not root.is_dir():
            parser.error('candidate-dir must be a directory')
        frozen = json.loads((HERE / 'frozen-candidate.json').read_text())
        candidate = {}
        for path in CANDIDATE:
            file = root / path
            if not file.is_file() or file.is_symlink():
                raise RuntimeError(f'missing or nonregular candidate: {path}')
            content = file.read_bytes()
            if digest(content) != frozen[path]:
                raise RuntimeError(f'frozen candidate drift: {path}')
            candidate[path] = content
    else:
        candidate = None
    dest.mkdir(mode=0o700)
    git(dest, 'init', '-q')
    git(dest, 'remote', 'add', 'origin', REMOTE)
    pins = list(SOURCES.values()) + ([args.candidate_commit] if args.candidate_commit else [])
    git(dest, '-c', 'protocol.file.allow=never', 'fetch', '-q', '--no-tags',
        '--filter=blob:none', '--depth=1', 'origin', *pins)
    for pin in pins:
        if git(dest, 'rev-parse', pin + '^{commit}').decode().strip() != pin:
            raise RuntimeError(f'not a fetched public commit: {pin}')
    git(dest, '-c', 'core.hooksPath=/dev/null', 'checkout', '-q', '--detach', SOURCES['permission'])
    # One non-overlapping policy join: M73's three legacy helpers and imports,
    # then the public Permission private-admin predicates and necessary import.
    legacy = git(dest, 'show', SOURCES['m73'] + ':gateway/group_chat_policy.py')
    private = git(dest, 'show', SOURCES['permission'] + ':gateway/group_chat_policy.py')
    private_lines = private.splitlines(keepends=True)
    assert private_lines[5].startswith(b'from gateway.session_group_messaging_identity import ')
    assert sum(line.startswith(b'def private_admin_receiver(') for line in private_lines) == 1
    assert sum(line.startswith(b'def private_admin_event(') for line in private_lines) == 1
    assert private_lines[-1].startswith(b'    return private_admin_receiver(')
    policy = legacy.rstrip(b'\n') + b'\n\n' + b''.join(private_lines[5:])
    (dest / 'gateway/group_chat_policy.py').write_bytes(policy)
    # M73 home-control helpers were removed from the Permission slash-access
    # surface but remain imports of M73 policy and the candidate bridge.
    access_legacy = git(dest, 'show', SOURCES['m73'] + ':gateway/slash_access.py')
    access_current = (dest / 'gateway/slash_access.py').read_bytes()
    marker = b'def _platform_extra('
    if access_legacy.count(marker) != 1 or marker in access_current:
        raise RuntimeError('M73 slash-access helper join changed')
    access = access_current.rstrip(b'\n') + b'\n\n' + access_legacy[access_legacy.index(marker):]
    (dest / 'gateway/slash_access.py').write_bytes(access)
    manifest = {'sources': SOURCES, 'candidate': args.candidate_commit or 'frozen-local-six',
                'policy_join_sha256': digest(policy), 'slash_access_join_sha256': digest(access),
                'candidate_files': {}, 'legacy_public_blobs': {}}
    legacy_oids = {path: git(dest, 'rev-parse', SOURCES['m73'] + ':' + path).decode().strip()
                   for path in LEGACY}
    for path, oid in ROUTE.items():
        if git(dest, 'rev-parse', SOURCES['route'] + ':' + path).decode().strip() != oid:
            raise RuntimeError(f'Route public blob identity changed: {path}')
    git(dest, '-c', 'protocol.file.allow=never', 'fetch', '-q', '--no-tags',
        'origin', *dict.fromkeys([*legacy_oids.values(), *ROUTE.values()]))
    for path, oid in ROUTE.items():
        target = dest / path
        target.write_bytes(git(dest, 'cat-file', 'blob', oid))
    manifest['route_public_blobs'] = ROUTE
    # The separate selected-route owner supplies the external-writer lifetime
    # used by Permission's Send attestation. Never replace it with a local shim.
    state = git(dest, 'show', SOURCES['selected-route'] + ':hermes_state.py')
    (dest / 'hermes_state.py').write_bytes(state)
    manifest['selected_route_state_sha256'] = digest(state)
    for path in LEGACY:
        content = git(dest, 'show', SOURCES['m73'] + ':' + path)
        target = dest / path
        if target.exists():
            raise RuntimeError(f'M73 legacy import no longer absent from Permission: {path}')
        target.write_bytes(content)
        manifest['legacy_public_blobs'][path] = legacy_oids[path]
    for path in CANDIDATE:
        content = candidate[path] if candidate is not None else git(dest, 'show', args.candidate_commit + ':' + path)
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest['candidate_files'][path] = digest(content)
    (dest / 'RECONSTRUCTION.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'reconstructed {dest} from public Permission {SOURCES["permission"]}, '
          f'M73 {SOURCES["m73"]}, candidate {manifest["candidate"]}')
    print('policy join:', manifest['policy_join_sha256'])


if __name__ == '__main__':
    main()
