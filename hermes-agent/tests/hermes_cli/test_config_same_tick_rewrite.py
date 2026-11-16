"""Config caches see a same-size in-place rewrite made in the timestamp tick of the cached read.

Kernels without multigrain timestamps (CI runners, older hosts) stamp mtime/ctime from a coarse
clock, so ``config.yaml`` rewritten in place to a same-length value (``port 3`` -> ``port 5``) in the
tick of an earlier read keeps mtime, ctime, size and inode: the stat signature every config cache
keys on is unchanged, and the older parse was served for the rest of the process. This host may
have fine-grained timestamps, so ``coarse_clock`` reports both writes in one frozen tick, which is
what such a kernel returns."""
import json
import os
import time

import pytest


def _write(path, port):
    path.write_text(json.dumps({'model': {'provider': 'custom', 'base_url': f'http://127.0.0.1:{port}/v1'}}))


def _readers():
    from hermes_cli.config import load_config, load_config_readonly, read_raw_config
    from hermes_cli.config_backend import read_config_doc_readonly
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.config import get_config_path
    path = get_config_path()
    return {
        'load_config': lambda: load_config(),
        'load_config_readonly': lambda: load_config_readonly(),
        'read_raw_config': lambda: read_raw_config(),
        'load_user_config_effective': lambda: load_user_config_effective(path),
        'read_config_doc_readonly': lambda: read_config_doc_readonly(path),
    }


class _Clock:
    """``utils``' view of the ``time`` module with ``time_ns`` under the test's control."""

    def __init__(self, now):
        self.now = now

    def time_ns(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def coarse_clock(monkeypatch):
    """Every stat signature reports one frozen mtime/ctime tick, like a coarse filesystem clock; the
    returned clock is what ``utils`` reads as now, so a test can move past the racy window."""
    import utils
    real, tick = utils.file_signature, time.time_ns()
    monkeypatch.setattr(utils, 'file_signature', lambda st: (tick, *real(st)[1:3], tick))
    clock = _Clock(tick)
    monkeypatch.setattr(utils, 'time', clock)
    return clock


@pytest.fixture
def digest_opens(monkeypatch):
    """Paths opened through ``utils`` (the cache-key content digest, and some parsers)."""
    import builtins
    import utils
    opened = []

    def _open(file, *args, **kwargs):
        opened.append(os.fspath(file))
        return builtins.open(file, *args, **kwargs)

    monkeypatch.setattr(utils, 'open', _open, raising=False)
    return opened


@pytest.fixture
def frozen_now(monkeypatch):
    """``utils`` reads now as the test's start, so every file the test writes stays racy (fresh)."""
    import utils
    monkeypatch.setattr(utils, 'time', _Clock(time.time_ns()))

def test_same_size_rewrite_in_one_tick_reaches_every_config_reader(coarse_clock):
    from hermes_cli.config import get_config_path
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    readers = _readers()
    _write(path, 3)
    assert {name: read()['model']['base_url'] for name, read in readers.items()} == dict.fromkeys(
        readers, 'http://127.0.0.1:3/v1')
    _write(path, 5)
    assert {name: read()['model']['base_url'] for name, read in readers.items()} == dict.fromkeys(
        readers, 'http://127.0.0.1:5/v1')


def test_an_unchanged_file_stays_one_parse_while_fresh_and_once_settled(coarse_clock, digest_opens):
    import utils
    from hermes_cli.config import get_config_path
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    readonly = {k: v for k, v in _readers().items() if k in ('load_config_readonly', 'read_config_doc_readonly')}
    _write(path, 3)
    for read in readonly.values():
        read()
    digest_opens.clear()
    for read in readonly.values():  # inside the racy window: hashed for the key, the digest matches, a hit
        assert read() is read()
    assert str(path) in digest_opens
    digest_opens.clear()
    coarse_clock.now += utils.RACY_MTIME_WINDOW_NS + 1
    for read in readonly.values():
        read()  # settled: one re-parse onto the stat-only key
    digest_opens.clear()
    for read in readonly.values():  # then hits that never open the file
        assert read() is read()
    assert str(path) not in digest_opens


@pytest.mark.platforms("linux", "macos")
@pytest.mark.skipif(hasattr(os, 'geteuid') and os.geteuid() == 0, reason='root reads a mode-000 file')
def test_a_fresh_config_turned_unreadable_keeps_the_last_good_settings(frozen_now, monkeypatch, tmp_path, caplog):
    """A fresh file that stats but cannot be read reaches each loader's read-error recovery: the
    content digest in the cache key must not raise where the stat-only key never opened the file."""
    from hermes_cli.config import get_config_path, load_config
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.managed_scope import invalidate_managed_cache, load_managed_config
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    managed = tmp_path / 'managed'
    managed.mkdir()
    monkeypatch.setenv('HERMES_MANAGED_DIR', str(managed))
    invalidate_managed_cache()
    (managed / 'config.yaml').write_text(json.dumps({'agent': {'max_turns': 7}}))
    _write(path, 3)
    assert load_config()['model']['base_url'] == 'http://127.0.0.1:3/v1'
    assert load_user_config_effective(path)['model']['base_url'] == 'http://127.0.0.1:3/v1'
    assert load_managed_config() == {'agent': {'max_turns': 7}}
    path.chmod(0)
    (managed / 'config.yaml').chmod(0)
    try:
        assert load_config()['model']['base_url'] == 'http://127.0.0.1:3/v1'
        assert load_user_config_effective(path)['model']['base_url'] == 'http://127.0.0.1:3/v1'
        assert load_managed_config() == {}
        assert 'IGNORING this managed file' in caplog.text
    finally:
        path.chmod(0o600)
        (managed / 'config.yaml').chmod(0o600)
        invalidate_managed_cache()
