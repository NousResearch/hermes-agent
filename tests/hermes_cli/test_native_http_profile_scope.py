"""Native Desktop grants may edit their own named profile, never another home."""

import pytest
from starlette.requests import Request


def _request(path, method='GET', query=''):
    async def receive():
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    return Request({
        'type': 'http', 'method': method, 'path': path, 'query_string': query.encode(),
        'headers': [], 'state': {}, 'scheme': 'http', 'server': ('testserver', 80),
    }, receive)


@pytest.mark.asyncio
async def test_named_profile_routes_ignore_an_ambient_selector_but_keep_home_boundary(tmp_path, monkeypatch):
    """The path already binds a profile route to the ticket's home.

    Registry-pinned Desktop requests used to append the profile shown in the
    sidebar.  That stale selector must not turn a correctly routed same-home
    grant into a 403, while a grant still cannot cross into another profile.
    """
    from hermes_cli.dashboard_auth.native_http import _check_profile_scope, _check_registry_scope
    from hermes_cli import web_server_profiles

    root = tmp_path / 'state'
    beta = root / 'profiles' / 'beta'
    beta.mkdir(parents=True)

    def resolve(name):
        return root if name == 'default' else root / 'profiles' / name

    monkeypatch.setattr(web_server_profiles, '_resolve_profile_dir', resolve)

    await _check_profile_scope(_request('/api/profiles/beta/soul', query='profile=default'), str(beta))
    # Rename/delete retain main's dedicated registry grant: they are allowed for
    # the grant's own secondary profile but cannot use an ordinary HTTP grant.
    _check_registry_scope(_request('/api/profiles/beta', 'PATCH'), str(beta), str(root))
    _check_registry_scope(_request('/api/profiles/beta', 'DELETE'), str(beta), str(root))

    with pytest.raises(PermissionError):
        await _check_profile_scope(_request('/api/profiles/default/soul'), str(beta))
    with pytest.raises(PermissionError):
        await _check_profile_scope(_request('/api/config', query='profile=default'), str(beta))
    with pytest.raises(PermissionError):
        _check_registry_scope(_request('/api/profiles/default', 'DELETE'), str(root), str(root))
