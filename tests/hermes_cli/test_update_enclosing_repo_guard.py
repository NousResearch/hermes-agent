"""A checkout git no longer recognizes must never let ``hermes update`` act on an enclosing repo.

git walks up from a damaged ``.git`` to the nearest parent repository, so an install under a home
directory that is itself a git repo would otherwise stash, reset and clean the user's home (#146032).
"""
import subprocess

import pytest

from hermes_cli import main as cli_main
from tests.hermes_cli.test_update_target_identity import git, update_tree


def _wrap_in_enclosing_repo(t):
    """Make the fixture's parent dir a git repo with an untracked credential, then break the checkout."""
    home = t.clone.parent
    git(home, 'init', '-q', '-b', 'main')  # the update's own branch: no parked-branch stop
    git(home, 'config', 'user.name', 'Home')
    git(home, 'config', 'user.email', 'home@example.invalid')
    (home / 'tracked.txt').write_text('home file\n', encoding='utf-8')
    git(home, 'add', 'tracked.txt')
    git(home, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'home')
    git(home, 'remote', 'add', 'origin', str(t.origin))  # so the update's fetch succeeds on the wrong repo
    (home / 'tracked.txt').write_text('edited home file\n', encoding='utf-8')
    secret = home / '.ssh' / 'id_ed25519'
    secret.parent.mkdir()
    secret.write_text('PRIVATE KEY\n', encoding='utf-8')
    (t.clone / '.git' / 'HEAD').unlink()  # git now resolves the checkout to the enclosing repo
    assert git(t.clone, 'rev-parse', '--show-toplevel') == str(home.resolve())
    return home, secret


def test_update_refuses_to_operate_on_the_repo_enclosing_a_damaged_checkout(update_tree, capsys):  # health: allow F811 -- pytest injects the imported update_tree fixture by parameter name
    t = update_tree
    t.args.channel = 'main'
    home, secret = _wrap_in_enclosing_repo(t)

    with pytest.raises(SystemExit) as stopped:
        cli_main.cmd_update(t.args)

    assert stopped.value.code == 1
    assert secret.read_text(encoding='utf-8') == 'PRIVATE KEY\n'
    assert (home / 'tracked.txt').read_text(encoding='utf-8') == 'edited home file\n'
    assert git(home, 'stash', 'list') == ''
    assert git(home, 'status', '--porcelain', '--untracked-files=all') != ''
    assert t.requests == []
    assert 'refusing to stash, reset or clean it' in capsys.readouterr().out


def test_plugin_pull_refuses_a_plugin_dir_git_resolves_to_an_enclosing_repo(tmp_path, monkeypatch):
    import hermes_cli.plugins_cmd as pc

    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(tmp_path / 'git-config'))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    home = tmp_path / 'home'
    plugin = home / 'plugins' / 'demo'
    plugin.mkdir(parents=True)
    git(home, 'init', '-q', '-b', 'master')
    (home / 'tracked.txt').write_text('home file\n', encoding='utf-8')
    git(home, 'add', 'tracked.txt')
    git(home, '-c', 'user.name=H', '-c', 'user.email=h@example.invalid', '-c', 'commit.gpgsign=false',
        'commit', '-qm', 'home')
    (home / 'tracked.txt').write_text('edited home file\n', encoding='utf-8')
    secret = home / '.ssh' / 'id_ed25519'
    secret.parent.mkdir()
    secret.write_text('PRIVATE KEY\n', encoding='utf-8')
    (plugin / '.git').mkdir()  # an empty/broken plugin .git: git walks up to the home repo
    assert subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=plugin, capture_output=True,
                          text=True, check=True).stdout.strip() == str(home.resolve())

    ok, message = pc._git_pull_plugin_dir(plugin)

    assert ok is False
    assert 'not its own git checkout' in message
    assert secret.read_text(encoding='utf-8') == 'PRIVATE KEY\n'
    assert (home / 'tracked.txt').read_text(encoding='utf-8') == 'edited home file\n'
    assert git(home, 'stash', 'list') == ''


def test_update_git_ignores_an_inherited_git_dir(tmp_path, monkeypatch):
    """A GIT_DIR from the user's shell (a dotfiles alias, a git hook) must not retarget the updater."""
    from hermes_cli.update_custody import run_git

    home, checkout = tmp_path / 'home', tmp_path / 'checkout'
    for repo in (home, checkout):
        repo.mkdir()
        git(repo, 'init', '-q')
    monkeypatch.setenv('GIT_DIR', str(home / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(home))

    top = run_git(['git'], ['rev-parse', '--show-toplevel'], cwd=str(checkout), capture_output=True, text=True)

    assert top.stdout.strip() == str(checkout.resolve())


def test_pinned_feed_update_refuses_a_plugin_dir_git_resolves_to_an_enclosing_repo(tmp_path):
    from hermes_cli.plugins_cmd_git import _foreign_checkout_error

    home = tmp_path / 'home'
    plugin = home / 'plugins' / 'demo'
    plugin.mkdir(parents=True)
    git(home, 'init', '-q')
    (plugin / '.git').mkdir()

    assert 'not its own git checkout' in _foreign_checkout_error('git', plugin)
    git(plugin, 'init', '-q')
    assert _foreign_checkout_error('git', plugin) == ''


def test_update_plugin_to_a_feed_commit_refuses_a_plugin_dir_git_resolves_to_an_enclosing_repo(
        tmp_path, monkeypatch):
    """The pinned-feed path inspects and checks out with git -C <plugin>; from a damaged plugin
    .git that is the git-managed home, so update_plugin must stop before touching it."""
    import json

    import hermes_cli.plugins_updates as updates
    from hermes_cli.plugins_cmd import PluginOperationError
    from hermes_cli.plugins_transaction import update_plugin

    home = tmp_path / 'home'
    plugin = home / 'plugins' / 'demo'
    plugin.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(tmp_path / 'git-config'))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    feed = 'https://feed.example/demo.yml'
    (plugin / 'plugin.yaml').write_text(f'name: demo\nupdate_url: {feed}\n', encoding='utf-8')
    (home / 'plugins' / '.install-metadata.json').write_text(json.dumps(
        {'demo': {'source': 'https://git.example/demo.git', 'update_url': feed}}), encoding='utf-8')
    git(home, 'init', '-q', '-b', 'main')
    git(home, 'add', '--all')
    git(home, '-c', 'user.name=H', '-c', 'user.email=h@example.invalid', '-c', 'commit.gpgsign=false',
        'commit', '-qm', 'home')
    head = git(home, 'rev-parse', 'HEAD')
    (plugin / '.git').mkdir()  # an empty/broken plugin .git: git walks up to the home repo
    monkeypatch.setattr(updates, 'default_fetch', lambda url: 'version: 2.0.0\nartifacts:\n  git: ' + 'a' * 40 + '\n')

    with pytest.raises(PluginOperationError, match='not its own git checkout'):
        update_plugin(plugin)

    assert git(home, 'rev-parse', 'HEAD') == head
    assert git(home, 'stash', 'list') == ''


def test_plugin_guard_ignores_an_inherited_git_dir(tmp_path, monkeypatch):
    """GIT_DIR disables discovery, so the toplevel would be the plugin dir even when git acts on another repo."""
    from hermes_cli.plugins_cmd_git import _foreign_checkout_error

    home = tmp_path / 'home'
    plugin = home / 'plugins' / 'demo'
    plugin.mkdir(parents=True)
    git(home, 'init', '-q')
    (plugin / '.git').mkdir()
    monkeypatch.setenv('GIT_DIR', str(home / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(plugin))

    assert 'not its own git checkout' in _foreign_checkout_error('git', plugin)
