"""A fork parked on a custom branch must still advance its ``main`` toward upstream.

``_sync_with_upstream_if_needed`` used to reach ``origin/main`` with ``git pull --ff-only upstream
main``, but ``git pull`` fast-forwards the CHECKED-OUT branch. On a custom branch carrying commits
upstream lacks the pull cannot fast-forward, so the fork's ``main`` stayed behind forever while the
run still printed "Fork is up to date".
"""
import subprocess

import pytest

from hermes_cli.update_cmd_git import (_advance_fork_main_without_checkout, _pull_would_advance_the_fork,
                                       _sync_with_upstream_if_needed)


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding='utf-8').stdout.strip()


def _commit(root, value, message):
    (root / 'cli.py').write_text(f'value = {value}\n', encoding='utf8')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', message)
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def world(tmp_path, monkeypatch):
    """upstream (tip ``up``) -> bare fork (``main`` at ``base``) -> checkout parked on ``local/fixes``.

    Returns ``(checkout, fork, shas)`` where ``shas`` holds ``base``/``up``/``local``.
    """
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(tmp_path / 'git-config'))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('GIT_ALLOW_PROTOCOL', 'file')

    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    git(upstream, 'init', '-q', '-b', 'main')
    git(upstream, 'config', 'user.name', 'Fixture')
    git(upstream, 'config', 'user.email', 'fixture@example.com')
    base = _commit(upstream, 1, 'base')
    up = _commit(upstream, 2, 'upstream work')

    # A bare fork, like a real one on GitHub: nothing is checked out, so main can be pushed into it.
    fork = tmp_path / 'fork.git'
    git(tmp_path, 'init', '-q', '--bare', '-b', 'main', str(fork))
    git(upstream, 'push', '-q', str(fork), f'{base}:refs/heads/main')

    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    git(checkout, 'init', '-q', '-b', 'main')
    git(checkout, 'config', 'user.name', 'Fixture')
    git(checkout, 'config', 'user.email', 'fixture@example.com')
    git(checkout, 'remote', 'add', 'origin', str(fork))
    git(checkout, 'remote', 'add', 'upstream', str(upstream))
    git(checkout, 'fetch', '-q', 'origin', '+refs/heads/main:refs/remotes/origin/main')
    git(checkout, 'fetch', '-q', 'upstream', '+refs/heads/main:refs/remotes/upstream/main')
    git(checkout, 'checkout', '-q', '-B', 'main', 'refs/remotes/origin/main')
    git(checkout, 'checkout', '-q', '-b', 'local/fixes')
    local = _commit(checkout, 99, 'local patch')
    return checkout, fork, {'base': base, 'up': up, 'local': local}


def _sync(checkout):
    return _sync_with_upstream_if_needed(["git"], checkout, assume_yes=True, input_fn=lambda *a: "n")


@pytest.mark.parametrize("branch, advances", [
    ("main", True),
    ("local/fixes", False),
    ("feature/x", False),
    ("HEAD", False),
])
def test_pull_would_advance_the_fork(branch, advances):
    assert _pull_would_advance_the_fork(branch) is advances


def test_parked_checkout_advances_the_fork_and_keeps_local_commits(world):
    checkout, fork, shas = world
    assert _sync(checkout) is True
    # the fork caught up ...
    assert git(fork, 'rev-parse', 'main') == shas['up']
    # ... and the parked branch still carries its own commit, untouched.
    assert git(checkout, 'rev-parse', '--abbrev-ref', 'HEAD') == 'local/fixes'
    assert git(checkout, 'rev-parse', 'HEAD') == shas['local']
    assert git(checkout, 'rev-parse', 'main') == shas['base']


def test_main_checkout_still_fast_forwards_as_before(world):
    checkout, fork, shas = world
    git(checkout, 'checkout', '-q', 'main')
    assert _sync(checkout) is True
    assert git(checkout, 'rev-parse', 'main') == shas['up']
    assert git(fork, 'rev-parse', 'main') == shas['up']


def test_lease_refuses_to_overwrite_a_moved_origin(world):
    checkout, fork, shas = world
    # A lease bound to a SHA that is not origin/main must fail instead of force-overwriting.
    assert _advance_fork_main_without_checkout(["git"], checkout, expected_origin_sha=shas['local']) is False
    assert git(fork, 'rev-parse', 'main') == shas['base']