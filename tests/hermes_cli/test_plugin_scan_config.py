"""The plugin scanner's existing setting is discoverable through config get (#135032)."""
import json


def test_unset_scan_setting_reports_the_runtime_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import get_config_value
    from hermes_cli.plugins_cmd import _scan_on_install_enabled

    get_config_value("plugins.scan_on_install", as_json=True)

    captured = capsys.readouterr()
    assert json.loads(captured.out) is True
    assert captured.err == ""
    assert _scan_on_install_enabled() is True


def test_explicit_scan_setting_survives_default_merge(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import get_config_value, set_config_value
    from hermes_cli.plugins_cmd import _scan_on_install_enabled

    set_config_value("plugins.scan_on_install", "false")
    capsys.readouterr()
    get_config_value("plugins.scan_on_install", as_json=True)

    captured = capsys.readouterr()
    assert json.loads(captured.out) is False
    assert captured.err == ""
    assert _scan_on_install_enabled() is False
