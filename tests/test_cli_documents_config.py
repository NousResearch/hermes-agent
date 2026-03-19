from cli import load_cli_config


def test_load_cli_config_defaults_include_documents_settings():
    config = load_cli_config()

    assert config["documents"]["parser_backend"] == "auto"
    assert config["documents"]["liteparse"]["ocr_language"] == "en"
    assert config["documents"]["liteparse"]["image_format"] == "png"
