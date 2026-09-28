"""Native dashboard inspection survives while retired authoring paths refuse writes."""
from fastapi.testclient import TestClient


def test_employee_dashboard_exclusions_and_person_memory(tmp_path, monkeypatch):
    from hermes_cli import web_server
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    people = home / 'memory' / 'people'
    people.mkdir(parents=True)
    (people / 'p1.md').write_text('Person one')
    (home / 'config.yaml').write_text('memory:\n  provider: honcho\n')
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    for path, method in [('/api/cron/jobs', 'post'), ('/api/cron/jobs/old', 'put'),
                         ('/api/cron/jobs/old', 'delete'), ('/api/cron/blueprints/instantiate', 'post'),
                         ('/api/profiles/default/soul', 'put'), ('/api/profiles/default/soul', 'get')]:
        response = client.request(method, path, json={})
        assert response.status_code == 410, response.text
    for selection in ({'keep_skills': []}, {'hub_skills': ['old']}):
        assert client.post('/api/profiles', json={'name': 'retired', **selection}).status_code == 410
    assert not (home / 'profiles' / 'retired').exists()
    assert client.get('/api/skills').status_code == 404
    from hermes_cli.web_server_dashboard import _discover_dashboard_plugins, _mount_plugin_api_routes
    assert 'kanban' not in {plugin['name'] for plugin in _discover_dashboard_plugins()}
    _mount_plugin_api_routes()
    assert client.post('/api/plugins/kanban/dispatch', json={}).status_code == 404
    assert client.get('/api/cron/jobs?profile=default').status_code == 200
    status = client.get('/api/memory?profile=default').json()
    assert status['active'] == 'hindsight'
    assert {p['name'] for p in status['providers']} == {'hindsight'}
    assert status['builtin_files']['user'] == len('Person one')
    assert client.put('/api/memory/provider', json={'provider': ''}).status_code == 400
    assert client.put('/api/dashboard/plugin-providers', json={'memory_provider': 'honcho'}).status_code == 400
    assert client.get('/api/memory/providers/honcho/config').status_code == 404
    assert client.post('/api/memory/reset?profile=default', json={'target': 'user'}).status_code == 400
    assert (people / 'p1.md').read_text() == 'Person one'


def test_cli_keeps_cron_inspection_but_rejects_legacy_authoring(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from hermes_cli.main import _build_cli_parser
    from hermes_cli.cron import cron_command
    from tools.skills_sync import sync_skills
    from agent.employee_policy import CRON_AUTHORING_COMMANDS
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    parser, _ = _build_cli_parser()
    for name in ('skills', 'bundles', 'curator', 'kanban', 'sync'):
        with pytest.raises(SystemExit) as error:
            parser.parse_args([name])
        assert error.value.code == 2
    assert parser.parse_args(['cron', 'list']).cron_command == 'list'
    for command in CRON_AUTHORING_COMMANDS:
        assert cron_command(SimpleNamespace(cron_command=command)) == 2
    assert not (tmp_path / 'cron' / 'jobs.json').exists()
    before = sorted(path.relative_to(tmp_path) for path in (tmp_path / 'skills').rglob('*'))
    assert sync_skills(quiet=True)['copied'] == []
    assert sorted(path.relative_to(tmp_path) for path in (tmp_path / 'skills').rglob('*')) == before


def test_memory_setup_has_only_the_runtime_provider(tmp_path, monkeypatch):
    import pytest
    from hermes_cli import memory_setup
    from hermes_cli.plugins_cmd import _get_current_memory_provider, _save_memory_provider
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('memory:\n  provider: honcho\n')
    assert {row[0] for row in memory_setup._get_available_providers()} == {'hindsight'}
    assert _get_current_memory_provider() == 'hindsight'
    for name in ('', 'honcho'):
        with pytest.raises(ValueError):
            _save_memory_provider(name)
    selected = []
    monkeypatch.setattr(memory_setup, 'cmd_setup_provider', selected.append)
    memory_setup.cmd_setup(None)
    assert selected == ['hindsight']
