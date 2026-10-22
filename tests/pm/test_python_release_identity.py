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


def test_python_candidates_reject_nonmatching_assets_and_leave_manual_targets_alone(monkeypatch):
    tag = '20990102'
    triple = 'x86_64-unknown-linux-gnu'
    rejected = [
        f'cpython-3.14.8+20981231-{triple}-install_only.tar.gz',
        f'cpython-3.15.0+{tag}-{triple}-install_only.tar.gz',
        f'cpython-3.14.8+{tag}-{triple}-freethreaded-install_only.tar.gz',
        f'cpython-3.14.8+{tag}-freethreaded-{triple}-install_only.tar.gz',
        f'cpython-3.14.8+{tag}-{triple}-install_only.tar.gz.extra',
        f'cpython-3.14.8+{tag}-{triple}-install_only.tar.gz\n',
    ]
    calls = []
    for name in rejected:
        monkeypatch.setattr(update, '_get_json', lambda url, name=name: calls.append(url) or [
            {'tag_name': tag, 'assets': [{'name': name}]},
        ])
        assert packages.Python().latest_versions('linux-x64', locked='3.14.7+20981231') == []
    calls.clear()
    assert packages.Python().latest_versions('linux-arm64-bionic', locked='3.14.7+20981231') == []
    assert calls == []
