import os

import pytest

from hermes_cli.web_server import _save_anthropic_oauth_creds


class _DummyPool:
    def entries(self):
        return []

    def remove_entry(self, _id):
        return None

    def add_entry(self, _entry):
        return None


@pytest.fixture
def oauth_file(monkeypatch, tmp_path):
    target = tmp_path / '.anthropic_oauth.json'
    monkeypatch.setattr('agent.anthropic_adapter._get_hermes_oauth_file', lambda: target)
    monkeypatch.setattr('agent.credential_pool.load_pool', lambda _provider: _DummyPool())
    return target


def test_dashboard_oauth_write_uses_owner_only_permissions(oauth_file):
    old_umask = os.umask(0o022)
    try:
        _save_anthropic_oauth_creds('access-token', 'refresh-token', 123456)
    finally:
        os.umask(old_umask)

    assert oauth_file.exists()
    mode = oauth_file.stat().st_mode & 0o777
    assert mode == 0o600


def test_dashboard_oauth_write_is_atomic_and_cleans_temp_on_failure(oauth_file, monkeypatch):
    """If the atomic replace fails, no partial file or temp file is left."""
    import utils

    def flaky_replace(src, dst):
        raise OSError('simulated replace failure')

    monkeypatch.setattr(utils, 'atomic_replace', flaky_replace)

    with pytest.raises(OSError, match='simulated replace failure'):
        _save_anthropic_oauth_creds('access-token', 'refresh-token', 123456)

    assert not oauth_file.exists()
    # atomic_json_write stages to ``.<stem>_*.tmp`` and unlinks it on failure.
    assert not list(oauth_file.parent.glob('*.tmp'))


def test_dashboard_oauth_write_uses_atomic_json_write_with_owner_only_mode(oauth_file, monkeypatch):
    """The OAuth token file must be written 0o600 from creation via
    ``atomic_json_write(mode=0o600)``, so it is never briefly world-readable
    (the old ``os.replace`` + post-hoc ``chmod`` TOCTOU)."""
    import utils

    calls = {}
    real = utils.atomic_json_write

    def spy(path, data, **kwargs):
        calls['mode'] = kwargs.get('mode')
        return real(path, data, **kwargs)

    monkeypatch.setattr(utils, 'atomic_json_write', spy)

    _save_anthropic_oauth_creds('access-token', 'refresh-token', 123456)

    assert calls.get('mode') == 0o600, \
        'OAuth creds must be written 0o600 atomically (no chmod-after-replace window)'
    assert oauth_file.exists()
