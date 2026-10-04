def test_artifact_ignore_default_and_dashboard_schema():
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.web_server_config import CONFIG_SCHEMA
    assert DEFAULT_CONFIG['desktop']['artifacts']['ignore'] == []
    field = CONFIG_SCHEMA['desktop.artifacts.ignore']
    assert field['type'] == 'list'
