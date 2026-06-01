from pathlib import Path

from gateway.config import Platform, load_gateway_config


def test_load_gateway_config_reads_api_server_settings_from_hermes_dotenv(monkeypatch, tmp_path):
    hermes_home = tmp_path / '.hermes'
    hermes_home.mkdir(parents=True)
    (hermes_home / '.env').write_text(
        'API_SERVER_ENABLED=true\nAPI_SERVER_HOST=127.0.0.1\nAPI_SERVER_PORT=8642\n',
        encoding='utf-8',
    )
    (hermes_home / 'config.yaml').write_text('', encoding='utf-8')

    monkeypatch.setenv('HERMES_HOME', str(hermes_home))
    monkeypatch.delenv('API_SERVER_ENABLED', raising=False)
    monkeypatch.delenv('API_SERVER_HOST', raising=False)
    monkeypatch.delenv('API_SERVER_PORT', raising=False)

    config = load_gateway_config()

    assert Platform.API_SERVER in config.platforms
    api_cfg = config.platforms[Platform.API_SERVER]
    assert api_cfg.enabled is True
    assert api_cfg.extra['host'] == '127.0.0.1'
    assert api_cfg.extra['port'] == 8642

    # load_gateway_config bridges dotenv values into os.environ for downstream users too.
    assert str(Path(hermes_home / '.env')).endswith('.env')
