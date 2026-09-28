"""Non-divergence fast-forward failures must not trigger a destructive reset."""
import subprocess

import pytest

from hermes_cli import main as cli_main
from tests.hermes_cli.test_update_target_identity import git, update_tree  # noqa: F401


@pytest.mark.parametrize('failure', ['blob', 'ancestry', 'late-lock', 'none', 'diverged'])
def test_ff_failure(update_tree, monkeypatch, capsys, failure):
    t = update_tree
    # Select the fixture checkout explicitly when using a shared dev interpreter.
    monkeypatch.setenv('PYTHONPATH', str(t.clone))
    monkeypatch.setattr('hermes_cli.update_cmd._is_fork', lambda *_: False)
    git(t.clone, 'checkout', '-q', 'main')
    t.args.channel = 'main'
    if failure == 'diverged':
        (t.clone / 'local.txt').write_text('local commit\n', encoding='utf-8')
        git(t.clone, 'add', 'local.txt')
        git(t.clone, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'local')
    before = git(t.clone, 'rev-parse', 'HEAD')
    original = subprocess.run
    mutations = []

    def inject_failure(command, *args, **kwargs):
        if 'merge' in command and '--ff-only' in command:
            # Alter real fixture data, not Git's response. Ancestry survives a
            # missing blob, but deleting the commit makes ancestry indeterminate.
            if failure in {'blob', 'ancestry'}:
                ref = 'origin/main:content.txt' if failure == 'blob' else 'origin/main'
                oid = git(t.clone, 'rev-parse', ref)
                object_path = t.clone / '.git' / 'objects' / oid[:2] / oid[2:]
                object_path.chmod(0o600)
                object_path.unlink()
            elif failure == 'late-lock':
                # A lock appearing after preflight belongs to the same failure
                # classifier; this test does not change the lock cleanup policy.
                (t.clone / '.git' / 'index.lock').write_text('busy', encoding='utf-8')
        if 'reset' in command or 'update-ref' in command:
            mutations.append(command)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', inject_failure)
    if failure in {'none', 'diverged'}:
        cli_main.cmd_update(t.args)
        assert git(t.clone, 'rev-parse', 'HEAD') == t.newer
        assert len(t.requests) == 1
        assert bool(mutations) is (failure == 'diverged')
        if failure == 'diverged':
            assert before in git(t.clone, 'for-each-ref', '--format=%(objectname)',
                                 'refs/hermes-update-backups/')
        return

    with pytest.raises(SystemExit) as error:
        cli_main.cmd_update(t.args)
    assert error.value.code == 1
    assert not mutations, 'an I/O failure is not permission to reset or create divergence refs'
    assert git(t.clone, 'rev-parse', 'HEAD') == before
    assert not t.requests
    output = capsys.readouterr().out
    assert 'diverged' not in output and 'reset --hard' not in output
    assert 're-run `hermes update`' in output
    expected = {'blob': 'unable to read', 'ancestry': 'Could not determine Git ancestry',
                'late-lock': 'index.lock'}
    assert expected[failure] in output
    if failure == 'late-lock':
        assert (t.clone / '.git' / 'index.lock').read_text(encoding='utf-8') == 'busy'
