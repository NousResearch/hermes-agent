"""An operator-owned task must survive automatic template reconciliation."""
from pathlib import Path

import pytest

from hermes_cli import gateway_windows


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("entry", ["reconcile", "status"])
def test_custom_task_opt_out_preserves_registration(tmp_path, monkeypatch, capsys, entry):
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("gateway:\n  windows_task_reconcile: false\n", encoding="utf-8")
    task = "Hermes_Gateway_fixture"
    script = tmp_path / (task + ".cmd")
    xml = gateway_windows._build_scheduled_task_xml(task, script.with_suffix(".vbs"), r"PC\service")
    xml = xml.replace("LogonTrigger", "BootTrigger").replace("InteractiveToken", "Password")
    xml = xml.replace('version="1.4"', 'version="1.3"')
    calls = []

    def schtasks(args):
        calls.append(list(args))
        return 0, xml, ""

    monkeypatch.setattr(gateway_windows, "_exec_schtasks", schtasks)
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script)
    monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: r"PC\interactive")
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script)
    if entry == "reconcile":
        assert gateway_windows.reconcile_scheduled_task(task) is False
    else:
        gateway_windows._print_scheduled_task_drift(task)
    assert not any(c[0] in ("/Delete", "/Create") for c in calls)
    output = capsys.readouterr().out
    assert "disabled" in output.lower()
    assert "Repair:" not in output


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("setting", ["", "gateway:\n  windows_task_reconcile: true\n"])
def test_default_and_enabled_policy_keep_template_repair(tmp_path, monkeypatch, setting):
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(setting, encoding="utf-8")
    script = tmp_path / "gateway.cmd"
    xml = gateway_windows._build_scheduled_task_xml("fixture", script.with_suffix(".vbs"), "user")
    registered = {"xml": xml.replace("<Interval>PT1M</Interval>", "<Interval>PT5M</Interval>")}
    calls = []

    def schtasks(args):
        calls.append(args)
        if args[0] == "/Create":
            registered["xml"] = Path(args[args.index("/XML") + 1]).read_text(encoding="utf-16")
        return 0, registered["xml"], ""

    monkeypatch.setattr(gateway_windows, "_exec_schtasks", schtasks)
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: script)
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script)
    monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: "user")
    assert gateway_windows.reconcile_scheduled_task("fixture") is True
    assert [c[0] for c in calls if c[0] in ("/Delete", "/Create")] == ["/Delete", "/Create"]
    calls.clear()
    assert gateway_windows.reconcile_scheduled_task("fixture") is False
    assert not any(c[0] in ("/Delete", "/Create") for c in calls)


def test_policy_read_error_does_not_authorize_rewrite(monkeypatch):
    from hermes_cli import config

    def broken_config():
        raise OSError("fixture read failure")

    monkeypatch.setattr(config, "load_config", broken_config)
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: pytest.fail("must not write"))
    assert gateway_windows.reconcile_scheduled_task("fixture") is False
