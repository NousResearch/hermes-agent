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
    monkeypatch.setattr('agent.anthropic_adapter._HERMES_OAUTH_FILE', target)
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


def test_dashboard_oauth_write_uses_atomic_replace_and_cleans_temp_files(oauth_file, monkeypatch):
    replace_calls = []

    def flaky_replace(src, dst):
        replace_calls.append((src, dst))
        raise OSError('simulated replace failure')

    monkeypatch.setattr('hermes_cli.web_server.os.replace', flaky_replace)

    with pytest.raises(OSError, match='simulated replace failure'):
        _save_anthropic_oauth_creds('access-token', 'refresh-token', 123456)

    assert replace_calls, 'helper should attempt atomic os.replace()'
    assert not oauth_file.exists()
    assert not list(oauth_file.parent.glob(f'{oauth_file.name}.tmp*'))
