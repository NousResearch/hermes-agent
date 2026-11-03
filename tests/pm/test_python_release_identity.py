"""Automatic Python candidates reconstruct only filenames the release advertises."""
from urllib.parse import unquote, urlparse

import pytest

from pm import packages, update


@pytest.mark.parametrize('target,triple', [
    ('linux-x64', 'x86_64-unknown-linux-gnu'),
    ('win32-arm64', 'aarch64-pc-windows-msvc'),
    ('darwin-arm64', 'aarch64-apple-darwin'),
])
def test_python_candidate_preserves_the_advertised_patch_and_build(monkeypatch, target, triple):
    tag = '20990102'
    identity = f'3.14.8+{tag}'
    advertised = f'cpython-{identity}-{triple}-install_only.tar.gz'
    release = {'tag_name': tag, 'assets': [{'name': name} for name in [
        f'cpython-3.15.0+{tag}-{triple}-install_only.tar.gz',
        f'cpython-{identity}-{triple}-freethreaded-install_only.tar.gz',
        f'cpython-{identity}-freethreaded-{triple}-install_only.tar.gz',
        advertised,
    ]]}
    monkeypatch.setattr(update, '_get_json', lambda _url: [release])
    package = packages.Python()
    result = package.latest_versions(target, locked='3.14.7+20981231')
    assert result == [identity]
    download = urlparse(package.fetch_url(result[0], target))
    assert unquote(download.path.rsplit('/', 1)[1]) == advertised
    assert download.path.split('/')[-2] == tag


_REJECTED_ASSETS = [
    'cpython-3.14.8+20981231-x86_64-unknown-linux-gnu-install_only.tar.gz',
    'cpython-3.15.0+20990102-x86_64-unknown-linux-gnu-install_only.tar.gz',
    'cpython-3.14.8+20990102-x86_64-unknown-linux-gnu-freethreaded-install_only.tar.gz',
    'cpython-3.14.8+20990102-freethreaded-x86_64-unknown-linux-gnu-install_only.tar.gz',
    'cpython-3.14.8+20990102-x86_64-unknown-linux-gnu-install_only.tar.gz.extra',
    'cpython-3.14.8+20990102-x86_64-unknown-linux-gnu-install_only.tar.gz\n',
]


@pytest.mark.parametrize("name", _REJECTED_ASSETS)
def test_python_candidates_reject_nonmatching_assets_and_leave_manual_targets_alone(monkeypatch, name):
    tag = '20990102'
    calls = []
    monkeypatch.setattr(update, '_get_json', lambda url, name=name: calls.append(url) or [
        {'tag_name': tag, 'assets': [{'name': name}]},
    ])
    assert packages.Python().latest_versions('linux-x64', locked='3.14.7+20981231') == []
    calls.clear()
    assert packages.Python().latest_versions('linux-arm64-bionic', locked='3.14.7+20981231') == []
    assert calls == []
