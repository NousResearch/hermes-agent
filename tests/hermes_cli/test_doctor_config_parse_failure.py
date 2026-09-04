"""hermes doctor flags an unparseable config.yaml as blocking (#102945).

``load_config()`` falls back to ``DEFAULT_CONFIG`` on YAML errors, so every
user override is silently ignored and the startup stderr warning scrolls
away. Doctor must surface the recorded parse failure as a red check plus a
manual action item; valid configs stay quiet.
"""

import pytest

import hermes_cli.doctor as doctor_mod

CORRUPT_YAML = "model:\n  provider: openai-codex\n  default: gpt-5.5\n broken: [unterminated\n"
VALID_YAML = "gateway:\n  enabled: false\n"


def _setup_home(tmp_path, monkeypatch, config_text):
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg = home / "config.yaml"
    cfg.write_text(config_text)
    return home, cfg


class TestDoctorConfigParseFailure:
    def test_corrupt_config_appends_blocking_item(self, tmp_path, monkeypatch, capsys):
        _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        manual = []
        doctor_mod.check_config_parse_failure(manual)
        assert len(manual) == 1
        assert "config.yaml" in manual[0]
        out = capsys.readouterr().out
        assert "IGNORED" in out

    def test_valid_config_stays_quiet(self, tmp_path, monkeypatch, capsys):
        _setup_home(tmp_path, monkeypatch, VALID_YAML)
        manual = []
        doctor_mod.check_config_parse_failure(manual)
        assert manual == []
        out = capsys.readouterr().out
        assert "IGNORED" not in out

    def test_item_clears_once_file_fixed(self, tmp_path, monkeypatch):
        _home, cfg = _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        manual = []
        doctor_mod.check_config_parse_failure(manual)
        assert len(manual) == 1
        cfg.write_text(VALID_YAML)  # user fixes the YAML — different size/mtime
        manual = []
        doctor_mod.check_config_parse_failure(manual)
        assert manual == []
