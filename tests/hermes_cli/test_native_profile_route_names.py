"""Profiles NAMED like a collection route (``active``, ``import``, ``sessions``) are ordinary profiles.

``/api/profiles/active`` is the sticky-default collection for GET/POST only, ``/import`` for POST,
``/sessions`` for GET; PATCH/DELETE of those paths reach ``/api/profiles/{name}`` for the profile of
that name. The registry grant and the native-http scope check used to be method-blind, so those
profiles could never be renamed or deleted from Desktop, and an ordinary native-http grant could
PATCH/DELETE the profile named ``sessions`` through the aggregate branch (review of #140970).
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers, QueryParams

from hermes_cli.dashboard_auth import native_http


def _request(method, path, query=''):
    async def json():
        raise ValueError
    return SimpleNamespace(method=method, url=SimpleNamespace(path=path), query_params=QueryParams(query),
                           headers=Headers({}), json=json)


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    for name in ('active', 'import', 'sessions', 'beta'):
        (root / 'profiles' / name).mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(root))
    from hermes_cli import web_server_profiles
    # The profile registry's own name -> dir mapping (validation + existence), rooted here.
    monkeypatch.setattr(web_server_profiles, '_resolve_profile_dir',
                        lambda name: root / 'profiles' / name if name != 'default' else root)
    return root.resolve()


@pytest.mark.parametrize('name', ['active', 'import', 'sessions'])
@pytest.mark.parametrize('method', ['PATCH', 'DELETE'])
def test_registry_grant_renames_and_deletes_a_profile_named_like_a_collection(root, name, method):
    request = _request(method, f'/api/profiles/{name}')
    assert native_http._registry_target(request) == name
    native_http._check_registry_scope(request, str(root / 'profiles' / name), str(root))
    with pytest.raises(PermissionError):  # still bound to the grant's own profile
        native_http._check_registry_scope(request, str(root / 'profiles' / 'beta'), str(root))


@pytest.mark.parametrize('method,path', [('GET', '/api/profiles/active'), ('POST', '/api/profiles/active'),
                                         ('POST', '/api/profiles/import'), ('GET', '/api/profiles/sessions')])
def test_collection_routes_are_never_registry_targets(method, path):
    assert native_http.profile_collection_route(path, method) is not None
    assert native_http._registry_target(_request(method, path)) is None or method in {'GET', 'POST'}


def test_native_http_grant_cannot_rename_or_delete_the_profile_named_sessions(root):
    # @dokterdok: the aggregate-selector branch checked only ?profile=, with no method check.
    for method in ('PATCH', 'DELETE'):
        request = _request(method, '/api/profiles/sessions', 'profile=sessions')
        with pytest.raises(PermissionError):
            asyncio.run(native_http._check_profile_scope(request, str(root / 'profiles' / 'sessions')))
    # Its collection read keeps working for the profile it names.
    asyncio.run(native_http._check_profile_scope(_request('GET', '/api/profiles/sessions', 'profile=sessions'),
                                                 str(root / 'profiles' / 'sessions')))


def test_registry_scope_fails_closed_when_the_launch_home_is_unknown(root):
    # @kshitijk4poor: _launch_home() returning None skipped the launch-home guard.
    with pytest.raises(PermissionError):
        native_http._check_registry_scope(_request('DELETE', '/api/profiles/beta'),
                                          str(root / 'profiles' / 'beta'), None)


def test_launch_home_guard_compares_home_keys_not_raw_paths(root, tmp_path):
    # A launch home reached through a symlink is still the launch home (hermes_home_key, as
    # SessionAuthorities keys it), so deleting it stays refused.
    link = tmp_path / 'link'
    link.symlink_to(root / 'profiles' / 'beta')
    with pytest.raises(PermissionError):
        native_http._check_registry_scope(_request('DELETE', '/api/profiles/beta'),
                                          str(root / 'profiles' / 'beta'), str(link))
    assert Path(link).resolve() == root / 'profiles' / 'beta'
