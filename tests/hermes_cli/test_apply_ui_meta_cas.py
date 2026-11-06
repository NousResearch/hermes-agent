"""Canonical metadata CAS semantics and cooperative cross-process exclusion."""

from __future__ import annotations

import json

import subprocess

import sys

import threading

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

import pytest

from hermes_cli.profiles_ui_meta import apply_ui_meta_cas

@pytest.fixture
def profile_dir(tmp_path):
    d = tmp_path / "default"
    d.mkdir()
    return d

def _read_yaml(path: Path):
    import hermes_yaml as yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def test_first_write_applies_and_advances_revision(profile_dir):
    result = apply_ui_meta_cas(profile_dir, {"hermes-bots": {"title": "Scout"}})
    assert result == {"applied": True, "revisions": {"hermes-bots": 1}}
    doc = _read_yaml(profile_dir / "profile.yaml")
    assert doc["ui_meta"]["hermes-bots"] == {"title": "Scout"}
    assert doc["_ui_meta_revisions"]["hermes-bots"] == 1

def test_expected_revision_zero_on_existing_key_conflicts(profile_dir):
    apply_ui_meta_cas(profile_dir, {"hermes-bots": {"title": "Scout"}})
    stale = apply_ui_meta_cas(
        profile_dir, {"hermes-bots": {"title": "Imposter"}}, expected_revisions={"hermes-bots": 0})
    assert stale["applied"] is False
    assert stale["conflicts"] == {"hermes-bots": {"expected": 0, "actual": 1}}
    assert _read_yaml(profile_dir / "profile.yaml")["ui_meta"]["hermes-bots"] == {"title": "Scout"}

def test_matching_expected_revision_applies(profile_dir):
    apply_ui_meta_cas(profile_dir, {"hermes-bots": {"title": "Scout"}})
    update = apply_ui_meta_cas(
        profile_dir, {"hermes-bots": {"title": "Scout II"}}, expected_revisions={"hermes-bots": 1})
    assert update == {"applied": True, "revisions": {"hermes-bots": 2}}

def test_deletion_survives_revision_and_blocks_stale_resurrection(profile_dir):
    apply_ui_meta_cas(profile_dir, {"shared-room": {"messages": ["one"]}})
    deleted = apply_ui_meta_cas(profile_dir, {"shared-room": None}, expected_revisions={"shared-room": 1})
    assert deleted["applied"] is True
    assert deleted["revisions"] == {"shared-room": 2}
    assert "shared-room" not in _read_yaml(profile_dir / "profile.yaml").get("ui_meta", {})
    stale = apply_ui_meta_cas(
        profile_dir, {"shared-room": {"messages": ["resurrected"]}}, expected_revisions={"shared-room": 1})
    assert stale["applied"] is False
    assert stale["conflicts"]["shared-room"]["actual"] == 2

def test_oversize_payload_rejects_without_writing(profile_dir):
    big = {"k": "x" * 70000}
    result = apply_ui_meta_cas(profile_dir, big)
    assert result["applied"] is False
    assert not (profile_dir / "profile.yaml").exists()

def test_unrelated_keys_and_fields_are_preserved(profile_dir):
    (profile_dir / "profile.yaml").write_text(
        "description: watch CI\ndisplay_name: Scout\nui_meta:\n  other: keep\n", encoding="utf-8")
    apply_ui_meta_cas(profile_dir, {"hermes-bots": {"title": "Scout"}})
    doc = _read_yaml(profile_dir / "profile.yaml")
    assert doc["description"] == "watch CI"
    assert doc["display_name"] == "Scout"
    assert doc["ui_meta"]["other"] == "keep"
    assert doc["ui_meta"]["hermes-bots"] == {"title": "Scout"}

def test_missing_profile_dir_raises(profile_dir):
    with pytest.raises(FileNotFoundError):
        apply_ui_meta_cas(profile_dir.parent / "nope", {"k": 1})

def test_in_process_concurrent_writers_exactly_one_applies(profile_dir):
    """Two threads CAS the same fresh key: one applies, the other conflicts (the thread
    lock inside apply_ui_meta_cas serializes the read-check-write)."""
    def write(label):
        return apply_ui_meta_cas(
            profile_dir, {"hermes-bots": {"title": label}}, expected_revisions={"hermes-bots": 0})

    barrier = threading.Barrier(2)
    def run(label):
        barrier.wait()
        return write(label)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("alpha", "beta")))

    assert sum(r["applied"] for r in results) == 1
    loser = next(r for r in results if not r["applied"])
    assert loser["conflicts"]["hermes-bots"]["actual"] == 1

_CAS_CHILD = """
import json, sys
sys.path.insert(0, {repo!r})
from hermes_cli.profiles_ui_meta import apply_ui_meta_cas
result = apply_ui_meta_cas({profile!r}, {{"hermes-bots": {{"title": {title!r}}}}},
                           expected_revisions={{"hermes-bots": 0}})
print(json.dumps(result))
"""

def test_two_processes_cas_same_fresh_key_exactly_one_applies(tmp_path):
    """The race this module exists to fence: TWO PROCESSES (two independent clients) CAS the same fresh key. Without the cross-process file lock both would
    read an empty file, both pass expected=0, and one silently overwrites the other. With
    it, exactly one applies and the loser observes the winner's committed revision."""
    repo = str(Path(__file__).resolve().parent.parent.parent)
    profile = tmp_path / "default"
    profile.mkdir()
    env = {k: v for k, v in __import__("os").environ.items()}

    def child(title):
        return subprocess.run(
            [sys.executable, "-c", _CAS_CHILD.format(repo=repo, profile=str(profile), title=title)],
            capture_output=True, text=True, env=env, timeout=60, check=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(child, ("A", "B")))

    assert a.returncode == 0 and b.returncode == 0, (a.stderr[-500:], b.stderr[-500:])
    results = [json.loads(a.stdout.strip().splitlines()[-1]),
               json.loads(b.stdout.strip().splitlines()[-1])]
    assert sum(r["applied"] for r in results) == 1
    loser = next(r for r in results if not r["applied"])
    assert loser["conflicts"] == {"hermes-bots": {"expected": 0, "actual": 1}}
    import hermes_yaml as yaml
    doc = yaml.safe_load((profile / "profile.yaml").read_text(encoding="utf-8"))
    assert doc["ui_meta"]["hermes-bots"]["title"] in {"A", "B"}  # the winner's write survived
    assert doc["_ui_meta_revisions"]["hermes-bots"] == 1

def test_second_process_waits_for_held_lock_then_applies(tmp_path):
    """A live holder (another process mid-CAS) must make a contender WAIT, not fail: it
    acquires the lock after release and applies cleanly against the still-fresh key."""
    repo = str(Path(__file__).resolve().parent.parent.parent)
    profile = tmp_path / "default"
    profile.mkdir()

    holder = subprocess.Popen(
        [sys.executable, "-c",
         f"import time, sys; sys.path.insert(0, {repo!r});\n"
         f"from hermes_cli.profiles_ui_meta import _profile_ui_meta_file_lock\n"
         f"with _profile_ui_meta_file_lock({str(profile)!r}):\n"
         f"    print('locked', flush=True)\n"
         f"    time.sleep(1.5)\n"
         f"print('released', flush=True)"],
        stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "locked"  # holder is INSIDE the lock now

    import time as _time
    started = _time.monotonic()
    contender = subprocess.run(
        [sys.executable, "-c", _CAS_CHILD.format(repo=repo, profile=str(profile), title="B")],
        capture_output=True, text=True, timeout=60, check=False)
    waited = _time.monotonic() - started
    holder.wait(timeout=30)

    assert contender.returncode == 0, contender.stderr[-500:]
    result = json.loads(contender.stdout.strip().splitlines()[-1])
    assert result["applied"] is True          # waited, then applied — no lost lock failure
    assert waited >= 0.5                      # it genuinely blocked on the held lock


@pytest.mark.parametrize('wanted', [True, False, -1, '0', 0.0, None])
def test_invalid_expected_revision_rejects_without_write(profile_dir, wanted):
    path = profile_dir / 'profile.yaml'
    path.write_text('description: unchanged\n')
    before = path.read_bytes()
    result = apply_ui_meta_cas(profile_dir, {'k': 'value'}, expected_revisions={'k': wanted})
    assert result['applied'] is False
    assert result['conflicts'] == {'k': {'expected': wanted, 'actual': 0}}
    assert path.read_bytes() == before


def test_malformed_revisions_are_normalized(profile_dir):
    path = profile_dir / 'profile.yaml'
    path.write_text('_ui_meta_revisions:\n  negative: -5\n  boolean: true\n  string: "4"\n  7: 3\n  good: 2\n')
    result = apply_ui_meta_cas(profile_dir, {'negative': 1, 'boolean': 1, 'string': 1},
                               expected_revisions={'negative': 0, 'boolean': 0, 'string': 0})
    assert result['applied'] is True
    assert _read_yaml(path)['_ui_meta_revisions'] == {
        'negative': 1, 'boolean': 1, 'string': 1, '7': 3, 'good': 2}


def test_multi_key_conflict_is_all_or_nothing(profile_dir):
    apply_ui_meta_cas(profile_dir, {'a': 'first', 'b': 'second'})
    path = profile_dir / 'profile.yaml'
    before = path.read_bytes()
    result = apply_ui_meta_cas(profile_dir, {'a': None, 'b': 'replacement', 'new': 1},
                               expected_revisions={'a': 1, 'b': 0, 'new': 0})
    assert result == {'applied': False, 'revisions': {'a': 1, 'b': 1, 'new': 0},
                      'conflicts': {'b': {'expected': 0, 'actual': 1}}}
    assert path.read_bytes() == before


def test_exact_serialized_size_boundary(profile_dir):
    payload = {'k': 'x' * (65536 - len(json.dumps({'k': ''})))}
    assert len(json.dumps(payload)) == 65536
    assert apply_ui_meta_cas(profile_dir, payload)['applied'] is True
    before = (profile_dir / 'profile.yaml').read_bytes()
    payload['k'] += 'x'
    assert apply_ui_meta_cas(profile_dir, payload)['oversize'] is True
    assert (profile_dir / 'profile.yaml').read_bytes() == before


@pytest.mark.parametrize('raw', [None, [], 'bad'])
def test_non_mapping_stored_revisions_start_at_zero(profile_dir, raw):
    import hermes_yaml as yaml
    (profile_dir / 'profile.yaml').write_text(yaml.safe_dump({'_ui_meta_revisions': raw}))
    assert apply_ui_meta_cas(profile_dir, {'k': 1}, expected_revisions={'k': 0})['revisions'] == {'k': 1}


@pytest.mark.parametrize('incoming, expected', [([], None), ({}, []), ('bad', {})])
def test_invalid_input_types_raise_without_writing(profile_dir, incoming, expected):
    with pytest.raises(ValueError):
        apply_ui_meta_cas(profile_dir, incoming, expected_revisions=expected)
    assert not (profile_dir / 'profile.yaml').exists()


def test_unsupported_locking_fails_closed(profile_dir, monkeypatch):
    from hermes_cli import profiles_ui_meta as profiles
    monkeypatch.setattr(profiles, '_uimeta_fcntl', None)
    monkeypatch.setattr(profiles, '_uimeta_msvcrt', None)
    with pytest.raises(NotImplementedError):
        apply_ui_meta_cas(profile_dir, {'k': 1})
    assert not (profile_dir / 'profile.yaml').exists()


def test_noncontention_lock_error_fails_closed(profile_dir, monkeypatch):
    import errno
    import types
    from hermes_cli import profiles_ui_meta as profiles
    def fail(*args):
        raise OSError(errno.ENOSYS, 'unsupported filesystem lock')
    monkeypatch.setattr(profiles, '_uimeta_fcntl', types.SimpleNamespace(flock=fail, LOCK_EX=1, LOCK_NB=2))
    with pytest.raises(OSError) as exc:
        apply_ui_meta_cas(profile_dir, {'k': 1})
    assert exc.value.errno == errno.ENOSYS
    assert not (profile_dir / 'profile.yaml').exists()


def test_lock_open_failure_leaves_yaml_unchanged(profile_dir, monkeypatch):
    import builtins
    path = profile_dir / 'profile.yaml'
    path.write_text('description: keep\n')
    before = path.read_bytes()
    original = builtins.open
    def guarded(file, *args, **kwargs):
        if str(file).endswith('.uimeta.lock'):
            raise PermissionError('synthetic lock permission refusal')
        return original(file, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', guarded)
    with pytest.raises(PermissionError):
        apply_ui_meta_cas(profile_dir, {'k': 1})
    assert path.read_bytes() == before


def test_atomic_write_failure_releases_lock_and_preserves_yaml(profile_dir, monkeypatch):
    import utils
    path = profile_dir / 'profile.yaml'
    path.write_text('description: keep\n')
    before = path.read_bytes()
    original = utils.atomic_yaml_write
    def fail(*args, **kwargs):
        raise OSError('synthetic replacement failure')
    monkeypatch.setattr(utils, 'atomic_yaml_write', fail)
    with pytest.raises(OSError):
        apply_ui_meta_cas(profile_dir, {'k': 1})
    assert path.read_bytes() == before
    monkeypatch.setattr(utils, 'atomic_yaml_write', original)
    assert apply_ui_meta_cas(profile_dir, {'k': 1})['applied'] is True


@pytest.mark.parametrize('existing', [b'', b'existing sidecar'])
def test_windows_backend_locks_and_unlocks_same_initialized_byte(profile_dir, monkeypatch, existing):
    """Backend protocol simulation, not native Windows qualification."""
    import os
    import errno
    import types
    from hermes_cli import profiles_ui_meta as profiles
    lock_path = profile_dir / 'profile.yaml.uimeta.lock'
    lock_path.write_bytes(existing)
    calls = []
    def locking(fd, operation, count):
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        assert os.fstat(fd).st_size >= 1
        assert count == 1
        calls.append(operation)
        if calls == [1]:
            raise OSError(errno.EACCES, 'synthetic contention')
    monkeypatch.setattr(profiles, '_uimeta_fcntl', None)
    monkeypatch.setattr(profiles, '_uimeta_msvcrt', types.SimpleNamespace(locking=locking, LK_NBLCK=1, LK_UNLCK=2))
    monkeypatch.setattr(profiles, '_PROFILE_UI_META_LOCK_POLL_S', 0)
    assert apply_ui_meta_cas(profile_dir, {'k': 1})['applied'] is True
    assert calls == [1, 1, 2]


_HOLDER_CHILD = '''
import sys
from pathlib import Path
from hermes_cli.profiles_ui_meta import _profile_ui_meta_file_lock
with _profile_ui_meta_file_lock(Path(sys.argv[1])):
    print('locked', flush=True)
    sys.stdin.readline()
'''


def _start_holder(profile_dir):
    proc = subprocess.Popen([sys.executable, '-c', _HOLDER_CHILD, str(profile_dir)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == 'locked'
    return proc


def test_lock_timeout_preserves_yaml_and_release_allows_retry(profile_dir, monkeypatch):
    from hermes_cli import profiles_ui_meta as profiles
    path = profile_dir / 'profile.yaml'
    path.write_text('description: keep\n')
    before = path.read_bytes()
    holder = _start_holder(profile_dir)
    try:
        monkeypatch.setattr(profiles, '_PROFILE_UI_META_LOCK_TIMEOUT_S', 0.1)
        with pytest.raises(TimeoutError):
            apply_ui_meta_cas(profile_dir, {'k': 1})
        assert path.read_bytes() == before
    finally:
        _, err = holder.communicate('\n', timeout=30)
        assert holder.returncode == 0, err
    assert apply_ui_meta_cas(profile_dir, {'k': 1})['applied'] is True


def test_process_death_releases_advisory_lock(profile_dir):
    holder = _start_holder(profile_dir)
    try:
        holder.kill()
        holder.communicate(timeout=30)
        assert holder.returncode != 0
        assert apply_ui_meta_cas(profile_dir, {'k': 1}, expected_revisions={'k': 0})['applied'] is True
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.communicate(timeout=30)
