"""An unimplemented plugin workspace must never become the gateway's filesystem."""
import pytest
from starlette.testclient import TestClient

from hermes_cli import ssh_workspace_fs, web_server
from hermes_cli.ssh_workspace_fs import SshWorkspaceFsError
from hermes_cli.web_routers import files


@pytest.mark.parametrize('backend', ['isolated_remote', 'unloaded_remote'])
def test_custom_backend_refuses_host_filesystem_without_dial(backend, monkeypatch):
    monkeypatch.setattr(ssh_workspace_fs, 'SSHEnvironment', lambda **kw: pytest.fail('unexpected SSH dial'))
    with pytest.raises(SshWorkspaceFsError) as raised:
        ssh_workspace_fs.get_ssh_workspace_fs('custom-profile', {'backend': backend})
    assert raised.value.code == 'EUNSUPPORTED'


def test_local_backend_still_uses_host_filesystem():
    assert ssh_workspace_fs.get_ssh_workspace_fs('local-profile', {'backend': 'local'}) is None


@pytest.mark.parametrize('route', ['/api/fs/list', '/api/fs/read-text'])
def test_files_api_does_not_read_same_named_host_path(route, tmp_path, monkeypatch):
    host_file = tmp_path / 'host-only.txt'
    host_file.write_text('MUST NOT LEAK FROM GATEWAY')
    monkeypatch.setattr(files, 'load_config', lambda: {'terminal': {'backend': 'isolated_remote'}})
    monkeypatch.setattr(files, 'load_env', dict)
    monkeypatch.setattr(web_server.app.state, 'auth_required', False, raising=False)
    monkeypatch.setattr(web_server.app.state, 'bound_host', None, raising=False)
    client = TestClient(web_server.app)
    try:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        response = client.get(route, params={'path': str(tmp_path if route.endswith('list') else host_file)})
    finally:
        client.close()
    assert response.status_code == 503
    assert 'MUST NOT LEAK' not in response.text
    assert host_file.read_text() == 'MUST NOT LEAK FROM GATEWAY'


def test_files_api_cannot_overwrite_same_named_host_path(tmp_path, monkeypatch):
    host_file = tmp_path / 'host-only.txt'
    host_file.write_text('ORIGINAL HOST CONTENT')
    monkeypatch.setattr(files, 'load_config', lambda: {'terminal': {'backend': 'unloaded_remote'}})
    monkeypatch.setattr(files, 'load_env', dict)
    monkeypatch.setattr(web_server.app.state, 'auth_required', False, raising=False)
    monkeypatch.setattr(web_server.app.state, 'bound_host', None, raising=False)
    client = TestClient(web_server.app)
    try:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        response = client.post('/api/fs/write-text', json={'path': str(host_file), 'content': 'WRONG HOST'})
    finally:
        client.close()
    assert response.status_code == 503
    assert host_file.read_text() == 'ORIGINAL HOST CONTENT'
