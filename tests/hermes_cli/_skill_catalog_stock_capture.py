"""Capture offline stock HTTP bytes from a pinned pure-main checkout.

Run with --root PURE_MAIN --home DISPOSABLE_HOME --output FIXTURE. Branch tests
reuse only the offline I/O boundaries, never regenerate expected responses.
"""
import argparse
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


BASE = '31ce88f86e6cf4a66ef6aa315e4657a31bcf9915'
STATES = ('never-used', 'previously-disabled', 'flag-off', 'signed-out')
REQUESTS = (
    ('GET', '/api/skills', None),
    ('GET', '/api/skills/hub/official', None),
    ('GET', '/api/skills/hub/sources', None),
    ('GET', '/api/skills/hub/search?q=fixture', None),
    ('GET', '/api/skills/hub/updates', None),
    ('GET', '/api/skills/hub/preview?identifier=fixture/absent', None),
    ('GET', '/api/skills/hub/scan?identifier=fixture/absent', None),
    ('POST', '/api/skills/hub/install', {'identifier': 'fixture/absent'}),
    ('POST', '/api/skills/hub/uninstall', {'name': 'local'}),
    ('POST', '/api/skills/hub/update', {}),
    ('POST', '/api/skills/hub/review', {}),
    ('POST', '/api/skills/hub/confirm', {}),
    ('GET', '/api/skills/hub/operation', None),
)


def seed(home, state):
    """Identical local bytes, including the formerly managed target after disable."""
    home.mkdir(parents=True, exist_ok=True)
    local = home / 'skills/local'
    local.mkdir(parents=True, exist_ok=True)
    (local / 'SKILL.md').write_text('---\nname: local\ndescription: kept local bytes\n---\n\nLocal\n')
    configs = {
        'never-used': {},
        'previously-disabled': {'plugins': {'enabled': [], 'disabled': ['fixture-catalog'],
            'entries': {'fixture-catalog': {'enabled': False, 'granted_capabilities': ['skills.catalog']}}}},
        'flag-off': {'plugins': {'enabled': [], 'entries': {'fixture-catalog': {'enabled': False}}}},
        'signed-out': {},
    }
    (home / 'config.yaml').write_text(json.dumps(configs[state]))


def capture(home, state):
    """Use the real ASGI/auth/routes with only external catalog/process I/O stubbed."""
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_cli.web_routers import skills
    from tools import skills_hub_install
    from tools.skills_hub_official import OptionalSkillSource

    seed(home, state)
    with ExitStack() as stack:
        stack.enter_context(patch.object(OptionalSkillSource, 'list_local', lambda self: []))
        stack.enter_context(patch.object(skills, '_hub_sources', lambda profile: []))
        stack.enter_context(patch.object(skills, '_resolve_hub_skill', lambda *args: (None, None, None)))
        stack.enter_context(patch.object(skills_hub_install, 'check_for_skill_updates', list))
        stack.enter_context(patch.object(skills, 'spawn_profile_action',
            lambda *args, **kwargs: dict(ok=True, pid=123, name='stock-action')))
        client = TestClient(app)
        stack.callback(client.close)
        if state != 'signed-out':
            client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        rows = []
        for method, path, body in REQUESTS:
            response = client.request(method, path, **({'json': body} if body is not None else {}))
            rows.append(dict(method=method, path=path, status=response.status_code,
                content_type=response.headers.get('content-type'), body=response.content.decode('utf-8')))
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert sha == BASE, sha
    assert not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)
    sys.path.insert(0, str(root))
    os.environ['HERMES_HOME'] = str(args.home.resolve())
    # One disposable current profile keeps absolute paths out of expected bytes.
    from hermes_cli.web_routers import skills
    assert Path(skills.__file__).is_relative_to(root), skills.__file__
    states = {state: capture(args.home, state) for state in STATES}
    payload = dict(base_sha=sha, boundaries='Offline catalog reads and process spawn only', states=states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(base_sha=sha, profiles=len(states), requests=sum(map(len, states.values())))))


if __name__ == '__main__':
    main()
