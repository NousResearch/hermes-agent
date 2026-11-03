"""file_tools.hosted_ocr is the only off switch for hosted document OCR, which uploads
scanned pages to Firecrawl. The config tooling has to know the key, and every way a
user can write "off" has to actually turn it off while FIRECRAWL_API_KEY is set."""

import pytest

from hermes_cli import config as cfg
from tools.read_extract import hosted_ocr_available


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-synthetic")
    return tmp_path


def test_config_set_false_is_a_known_key_and_disables_with_a_key(home, capsys):
    assert hosted_ocr_available() is True

    cfg.set_config_value("file_tools.hosted_ocr", "false")

    assert "not a recognized config key" not in capsys.readouterr().out
    assert cfg.load_config()["file_tools"]["hosted_ocr"] is False
    assert hosted_ocr_available() is False


@pytest.mark.parametrize("written", ["false", "False", "off", "0"])
def test_text_off_values_disable_with_a_key(home, written):
    # The dashboard edits this null-default key as a text field and saves the string as typed.
    (home / "config.yaml").write_text(
        f'file_tools:\n  hosted_ocr: "{written}"\n', encoding="utf-8")

    assert hosted_ocr_available() is False
