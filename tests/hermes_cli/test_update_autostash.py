"""Real Git local-work safety: caller divergence, restore faults and rescue retention."""
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from hermes_cli import main as hermes_main, update_cmd
from tests.hermes_cli.test_update_target_identity import git, update_tree  # noqa: F401


@pytest.mark.parametrize('history,failure,keep', [
    ('ordinary', None, False), ('ordinary', None, True),
    ('ordinary', 'reset', False), ('ordinary', 'reset', True),
    ('orphan', None, False), ('orphan', 'reset', False),
    ('orphan', 'ref', False), ('orphan', 'head', False),
])
def test_update_preserves_local_work_and_rescues_orphan_before_reset(
    update_tree, monkeypatch, capsys, history, failure, keep,
):
    t = update_tree
    git(t.clone, 'checkout', '-q', 'main')
    if history == 'orphan':
        git(t.clone, 'checkout', '--orphan', 'fresh')
        git(t.clone, 'branch', '-D', 'main')
        git(t.clone, 'branch', '-m', 'main')
    (t.clone / 'local.txt').write_text('committed\n', encoding='utf-8')
    git(t.clone, 'add', 'local.txt')
    git(t.clone, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'local history')
    before = git(t.clone, 'rev-parse', 'HEAD')
    (t.clone / 'untracked.txt').write_text('local edit\n', encoding='utf-8')
    t.args.channel, t.args.keep_stash = 'main', keep
    monkeypatch.setattr(hermes_main, '_sync_with_upstream_if_needed', update_cmd._sync_with_upstream_if_needed)
    monkeypatch.setattr(update_cmd, '_UPDATE_CRITICAL_MODULES', ())
    original = subprocess.run
    resets = []

    def fault(command, *args, **kwargs):
        if 'reset' in command and '--hard' in command:
            refs = original(['git', 'for-each-ref', '--format=%(objectname)',
                             'refs/hermes-update-backups/'], cwd=t.clone,
                            check=True, capture_output=True, text=True).stdout.split()
            assert refs == ([before] if history == 'orphan' and failure not in {'ref', 'head'} else [])
            resets.append(command)
        if ((failure == 'ref' and 'update-ref' in command and '-d' not in command)
                or (failure == 'reset' and 'reset' in command and '--hard' in command)):
            return subprocess.CompletedProcess(command, 128, stdout='', stderr='fixture I/O refusal')
        return original(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', fault)
    if failure == 'head':
        monkeypatch.setattr(update_cmd, '_capture_head_sha', lambda *_: None)
    if failure == 'reset':
        with pytest.raises(SystemExit) as error:
            hermes_main.cmd_update(t.args)
        assert error.value.code == 1
        assert not t.requests
        assert git(t.clone, 'rev-parse', 'HEAD') == before
        assert not (t.clone / 'untracked.txt').exists()
    else:
        hermes_main.cmd_update(t.args)
        assert len(t.requests) == 1
        assert git(t.clone, 'rev-parse', 'HEAD') == t.newer
        assert (t.clone / 'untracked.txt').exists() is (not keep)
    assert len(resets) == 1
    stashes = git(t.clone, 'stash', 'list')
    assert bool(stashes) is (keep or failure == 'reset')
    if stashes:
        assert git(t.clone, 'show', 'stash@{0}^3:untracked.txt') == 'local edit'
    output = capsys.readouterr().out
    if failure == 'ref':
        assert 'backup write failed' in output and 'backed up current HEAD' not in output
    if failure == 'reset':
        assert 'preserved in stash' in output
    if history == 'orphan' and failure not in {'ref', 'head'}:
        assert f'expires after {update_cmd._ORPHAN_RESCUE_REF_MAX_AGE_DAYS} days' in output


@pytest.mark.parametrize('mode', ['count', 'age', 'unparseable'])
def test_rescue_retention_uses_real_refs(tmp_path, monkeypatch, mode):
    from datetime import datetime, timedelta, timezone

    git(tmp_path, 'init', '-q', '-b', 'main')
    git(tmp_path, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'base')
    now = datetime.now(timezone.utc)
    prefix = 'refs/hermes-update-backups/orphan-main-'
    if mode == 'count':
        refs = [prefix + (now - timedelta(hours=20-i)).strftime('%Y%m%d-%H%M%S') + '-abc'
                for i in range(update_cmd._ORPHAN_RESCUE_REFS_TO_KEEP + 2)]
        removed = set(refs[:2])
    elif mode == 'age':
        refs = [prefix + '20200101-000000-abc', prefix + now.strftime('%Y%m%d-%H%M%S') + '-abc']
        removed = {refs[0]}
    else:
        refs, removed = [prefix + 'not-a-timestamp-abc'], set()
    for ref in refs:
        git(tmp_path, 'update-ref', ref, 'HEAD')
    update_cmd._prune_orphan_rescue_refs(['git'], tmp_path, 'main')
    assert set(git(tmp_path, 'for-each-ref', '--format=%(refname)', 'refs/hermes-update-backups/').splitlines()) == set(refs) - removed


@pytest.mark.parametrize('fault,body,modules,message', [
    ('syntax', '<<<<<<< Updated upstream\nVALUE = 2\n', (), 'made the Hermes agent unexecutable'),
    ('import', "raise RuntimeError('restored local failure')\n", ('consumer',), 'restored local failure'),
    ('preexisting', 'VALUE = 2\n', ('first',), None),
    ('later', "raise RuntimeError('restored later failure')\n", ('first', 'consumer'), 'restored later failure'),
    ('exit', "raise SystemExit('restored exit')\n", ('first', 'consumer'), 'restored exit'),
    ('terminated', 'import os\nos._exit(7)\n', ('consumer',), 'exit code 7'),
    ('paths', 'VALUE = 2\n', (), 'restored Python source discovery'),
])
def test_restore_validates_real_stash_and_each_import(tmp_path, monkeypatch, capsys, fault, body, modules, message):
    import hermes_cli.update_cmd_stash as stash

    git(tmp_path, 'init', '-q', '-b', 'main')
    (tmp_path / 'first.py').write_text("raise RuntimeError('missing local config')\n", encoding='utf-8')
    source = tmp_path / 'consumer.py'
    source.write_text('VALUE = 1\n', encoding='utf-8')
    git(tmp_path, 'add', 'first.py', 'consumer.py')
    git(tmp_path, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base')
    source.write_text(body, encoding='utf-8')
    ref = hermes_main._stash_local_changes_if_needed(['git'], tmp_path)
    assert ref
    monkeypatch.setattr(update_cmd, '_UPDATE_CRITICAL_MODULES', modules)
    if fault == 'paths':
        monkeypatch.setattr(stash, '_restored_python_paths', lambda *_: None)
        monkeypatch.setattr(update_cmd, '_restored_python_paths', lambda *_: None)
    if message is None:
        assert hermes_main._restore_stashed_changes(['git'], tmp_path, ref, prompt_user=False)
        assert source.read_text(encoding='utf-8') == body
        assert not git(tmp_path, 'stash', 'list')
    else:
        with pytest.raises(SystemExit) as error:
            hermes_main._restore_stashed_changes(['git'], tmp_path, ref, prompt_user=False)
        assert error.value.code == 1
        assert source.read_text(encoding='utf-8') == 'VALUE = 1\n'
        assert not git(tmp_path, 'status', '--porcelain')
        assert git(tmp_path, 'stash', 'list')
        output = capsys.readouterr().out
        assert message in output and 'gateway was not restarted' in output
        assert f'git stash apply {ref}' in output


@pytest.mark.parametrize('error', [EOFError(), UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid')])
def test_unreadable_stash_prompt_keeps_work(tmp_path, monkeypatch, capsys, error):
    def unreadable(*_):
        raise error
    monkeypatch.setattr('builtins.input', unreadable)
    assert hermes_main._restore_stashed_changes(['git'], tmp_path, 'stash@{0}', prompt_user=True) is False
    assert 'git stash apply stash@{0}' in capsys.readouterr().out


def test_unreadable_upstream_prompt_does_not_add_remote(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q', '-b', 'main')
    def unreadable(*_):
        raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid')
    monkeypatch.setattr('builtins.input', unreadable)
    monkeypatch.setattr(update_cmd, '_should_skip_upstream_prompt', lambda *_: False)
    update_cmd._sync_with_upstream_if_needed(['git'], tmp_path)
    assert not git(tmp_path, 'remote')


def test_update_parser_accepts_keep_stash():
    """The flag parses and defaults off."""
    import argparse

    from hermes_cli.subcommands.update import build_update_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    build_update_parser(subparsers, cmd_update=lambda args: None)

    args = parser.parse_args(["update", "--keep-stash"])
    assert args.keep_stash is True
    args = parser.parse_args(["update"])
    assert args.keep_stash is False



def test_bootstrap_marker_not_autostashed_by_update(tmp_path):
    """#38529: the Desktop bootstrap marker must be git-ignored so that
    ``hermes update``'s ``git stash push --include-untracked`` does not sweep it
    into an autostash on every run.

    Behavioral + hermetic: build a throwaway repo that adopts the project's real
    ``.gitignore`` (the contract under test), drop the marker, and confirm the
    same stash invocation the updater uses leaves it untouched.
    """
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    repo_gitignore = Path(hermes_main.__file__).resolve().parents[1] / ".gitignore"

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text(repo_gitignore.read_text())
    (tmp_path / "tracked.txt").write_text("x\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    marker = tmp_path / ".hermes-bootstrap-complete"
    marker.write_text("")

    # Exact flags used by hermes update (hermes_cli/main.py).
    git("stash", "push", "--include-untracked", "-m", "hermes-update-autostash")

    assert marker.exists(), (
        ".hermes-bootstrap-complete was swept into the update autostash — it must "
        "be listed in .gitignore so `git stash -u` skips it (#38529)."
    )
    # It must not even register as a dirty/untracked change.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert ".hermes-bootstrap-complete" not in status



def test_update_autostash_survives_undeletable_untracked_dir(tmp_path):
    """Behavioral E2E of the whole permission-denied class with real git:
    root-owned-style undeletable untracked dir → stash succeeds, update-style
    reset works, restore round-trips, nothing lost. (#70127 follow-up)"""
    import os
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")
    if os.name == "nt":
        pytest.skip("POSIX permission semantics")
    if os.geteuid() == 0:
        pytest.skip("root ignores directory write bits")

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=check
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("v1\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    (tmp_path / "tracked.txt").write_text("v2 local change\n")
    pkg = tmp_path / "packaging" / "homebrew"
    pkg.mkdir(parents=True)
    (pkg / "hermes-agent.rb").write_text("formula\n")
    os.chmod(pkg, 0o555)  # undeletable contents, like a root-owned dir
    try:
        stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
        assert stash_ref

        # The tracked change is stashed; simulate the updater's checkout window.
        assert (tmp_path / "tracked.txt").read_text() == "v1\n"

        restored = hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )
        assert restored is True
        assert (tmp_path / "tracked.txt").read_text() == "v2 local change\n"
        assert (pkg / "hermes-agent.rb").read_text() == "formula\n"
    finally:
        os.chmod(pkg, 0o755)



def test_restore_stays_parked_when_untracked_baseline_is_unknown(
    monkeypatch, tmp_path, capsys
):
    """Unknown cleanup scope must not turn into a destructive empty baseline."""
    from hermes_cli import update_cmd
    import hermes_cli.update_cmd_stash as update_cmd_stash

    monkeypatch.setattr(update_cmd, "_git_untracked_paths", lambda *_args: None)
    monkeypatch.setattr(update_cmd_stash, "_git_untracked_paths", lambda *_args: None)

    restored = hermes_main._restore_stashed_changes(
        ["git"], tmp_path, "stash@{0}", prompt_user=False
    )

    assert restored is False
    output = capsys.readouterr().out
    assert "cleanup baseline is unknown" in output
    assert "git stash apply stash@{0}" in output



def test_reject_does_not_claim_cleanup_when_git_state_is_unknown(
    monkeypatch, tmp_path, capsys
):
    """Cleanup failures must not be reported as a restored clean tree."""
    from hermes_cli import update_cmd
    import hermes_cli.update_cmd_stash as update_cmd_stash

    monkeypatch.setattr(update_cmd, "_git_untracked_paths", lambda *_args: None)
    monkeypatch.setattr(update_cmd_stash, "_git_untracked_paths", lambda *_args: None)

    with pytest.raises(SystemExit):
        update_cmd._reject_unsafe_stash_restore(
            ["git"], tmp_path, "stash@{0}", set(), "consumer.py", "invalid"
        )

    output = capsys.readouterr().out
    assert "could not be fully restored automatically" in output
    assert "The clean updated tree has been restored" not in output



def test_gateway_restore_prompt_defaults_to_keep_stash(tmp_path, capsys):
    prompts = []

    restored = hermes_main._restore_stashed_changes(
        ["git"],
        tmp_path,
        "stash@{0}",
        prompt_user=True,
        input_fn=lambda prompt, default: prompts.append((prompt, default)) or "",
    )

    assert restored is False
    assert prompts == [("Restore local changes now? [y/N]", "n")]
    assert "still preserved in git stash" in capsys.readouterr().out



def test_prune_orphan_rescue_refs_with_real_git_unpins_objects(tmp_path):
    """End-to-end with real git: an orphan rescue ref pins a snapshot's
    objects against gc; pruning the ref (age-expired) makes them collectable.
    This is the sabotage/size test for the #87745 bounded-growth mitigation."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=check
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("config", "gc.auto", "0")
    (tmp_path / "f.txt").write_text("base\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    # Snapshot commit carrying a "large" payload (scaled down for CI).
    import os

    (tmp_path / "big.bin").write_bytes(os.urandom(512 * 1024))
    git("add", "-A")
    git("commit", "-qm", "snapshot")
    snap_sha = git("rev-parse", "HEAD").stdout.strip()

    # Rewind, park the snapshot behind an AGE-EXPIRED rescue ref.
    git("reset", "-q", "--hard", "HEAD~1")
    old_ref = "refs/hermes-update-backups/orphan-main-20200101-000000-" + snap_sha[:12]
    git("update-ref", old_ref, snap_sha)

    # With the ref present, gc cannot drop the snapshot objects.
    git("reflog", "expire", "--expire=now", "--all")
    git("gc", "-q", "--prune=now")
    assert git("cat-file", "-e", snap_sha, check=False).returncode == 0

    # Prune (the ref's 2020 timestamp is way past the age window) → ref gone.
    update_cmd._prune_orphan_rescue_refs(["git"], tmp_path, "main")
    remaining = git("for-each-ref", "refs/hermes-update-backups/").stdout
    assert old_ref not in remaining

    # And gc can now reclaim the snapshot's objects.
    git("gc", "-q", "--prune=now")
    assert git("cat-file", "-e", snap_sha, check=False).returncode != 0
