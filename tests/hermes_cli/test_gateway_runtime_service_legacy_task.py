"""A Scheduled Task registered before the wscript launcher is login persistence, not a refusal."""
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

ACCOUNT, SID = "MXC-WIN11\\hermes", "S-1-5-21-3316076380-1684120700-1229173288-1000"


def _legacy_task_xml(command, *, arguments=None, user=SID, extra_principal="", extra_settings=""):
    """``schtasks /Query /TN Hermes_Gateway /XML`` on Windows 11 (build 26200) for the task the old
    installer's ``schtasks /Create /F /SC ONLOGON /RL LIMITED /TN Hermes_Gateway /TR <wrapper>`` made:
    schema 1.2, a logon trigger, no restart policy, and the ``.cmd`` wrapper as the whole action."""
    args = f"\n      <Arguments>{arguments}</Arguments>" if arguments is not None else ""
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Date>2026-11-26T02:55:05</Date>
    <Author>{ACCOUNT}</Author>
    <URI>\\Hermes_Gateway</URI>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>{extra_principal}
    </Principal>
  </Principals>
  <Settings>
    <DisallowStartIfOnBatteries>true</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>true</StopIfGoingOnBatteries>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>{extra_settings}
    <IdleSettings>
      <Duration>PT10M</Duration>
      <WaitTimeout>PT1H</WaitTimeout>
      <StopOnIdleEnd>true</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
  </Settings>
  <Triggers>
    <LogonTrigger>
      <StartBoundary>2026-11-26T02:55:00</StartBoundary>
    </LogonTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>{args}
    </Exec>
  </Actions>
</Task>
"""


def _wrapper(home: Path, task_name: str = "Hermes_Gateway") -> Path:
    return home / "gateway-service" / f"{task_name}.cmd"


@pytest.mark.parametrize("case", ["bare", "quoted", "utf16"])
def test_the_pre_wscript_task_of_this_home_is_logon_only(case, tmp_path):
    from hermes_cli.gateway_runtime_service_identity import LogonOnlyTask, verify_windows_task

    home = tmp_path / "hermes"
    command = str(_wrapper(home))
    xml = _legacy_task_xml(f'"{command}"' if case == "quoted" else command)
    with pytest.raises(LogonOnlyTask):
        verify_windows_task(xml.encode("utf-16") if case == "utf16" else xml, home, ACCOUNT, SID,
                            task_name="Hermes_Gateway")


@pytest.mark.parametrize("case,reason,check", [
    # Anything but this home's own wrapper, alone, keeps the refusal it had.
    ("other_home", "service_identity_unverified", "not wscript.exe"),
    ("other_task_name", "service_identity_unverified", "not wscript.exe"),
    ("other_script", "service_identity_unverified", "not wscript.exe"),
    ("with_arguments", "service_identity_unverified", "not wscript.exe"),
    ("relative", "service_identity_unverified", "not wscript.exe"),
    ("no_task_name", "service_identity_unverified", "not wscript.exe"),
    ("renamed_link", "service_identity_unverified", "not wscript.exe"),
    ("unmatched_quote", "service_identity_unverified", "not wscript.exe"),
    # A task Task Scheduler can start or restart on its own is not login persistence.
    ("time_trigger", "service_identity_unverified", "not wscript.exe"),
    ("second_trigger", "service_identity_unverified", "not wscript.exe"),
    ("repeating", "service_identity_unverified", "not wscript.exe"),
    ("restart_policy", "service_identity_unverified", "not wscript.exe"),
    # The principal and enabled checks come first, exactly as for a current task.
    ("wrong_user", "service_account_mismatch", None),
    ("elevated", "service_identity_unverified", "RunLevel"),
    ("disabled", "service_disabled", None),
    ("extra_action", "service_identity_unverified", "action(s)"),
])
def test_only_this_homes_own_wrapper_is_recognised(case, reason, check, tmp_path):
    from hermes_cli.gateway_runtime_service_identity import verify_windows_task

    home = tmp_path / "hermes"
    command, kwargs, task_name = str(_wrapper(home)), {}, "Hermes_Gateway"
    if case == "other_home":
        command = str(_wrapper(tmp_path / "elsewhere"))
    if case == "other_task_name":
        command = str(_wrapper(home, "Hermes_Gateway_work"))
    if case == "other_script":
        command = str(home / "gateway-service" / "anything.cmd")
    if case == "with_arguments":
        kwargs["arguments"] = "/c calc.exe"
    if case == "relative":
        command = "gateway-service\\Hermes_Gateway.cmd"
    if case == "no_task_name":
        task_name = None
    if case == "renamed_link":  # another registered name for the same file is not the installer's action
        _wrapper(home).parent.mkdir(parents=True)
        _wrapper(home).write_text("@echo off\r\n")
        (home / "gateway-service" / "retired.cmd").symlink_to(_wrapper(home))
        command = str(home / "gateway-service" / "retired.cmd")
    if case == "unmatched_quote":
        command = '"' + command
    if case == "restart_policy":
        kwargs["extra_settings"] = ("\n    <RestartOnFailure><Interval>PT1M</Interval><Count>3</Count>"
                                    "</RestartOnFailure>")
    if case == "wrong_user":
        kwargs["user"] = "S-1-5-18"
    if case == "elevated":
        kwargs["extra_principal"] = "\n      <RunLevel>HighestAvailable</RunLevel>"
    if case == "disabled":
        kwargs["extra_settings"] = "\n    <Enabled>false</Enabled>"
    xml = _legacy_task_xml(command, **kwargs)
    if case == "extra_action":
        xml = xml.replace("</Actions>", "<Exec><Command>other.exe</Command></Exec></Actions>")
    if case == "time_trigger":
        xml = xml.replace("LogonTrigger>", "TimeTrigger>")
    if case == "second_trigger":
        xml = xml.replace("</Triggers>", "<BootTrigger><Enabled>true</Enabled></BootTrigger></Triggers>")
    if case == "repeating":
        xml = xml.replace("</LogonTrigger>", "<Repetition><Interval>PT1M</Interval></Repetition></LogonTrigger>")
    with pytest.raises(ValueError, match=reason) as refused:
        verify_windows_task(xml, home, ACCOUNT, SID, task_name=task_name)
    if check:
        assert check in refused.value.check


def _schtasks(service, monkeypatch, name, xml):
    out = {"whoami.exe": f'"{ACCOUNT}","{SID}"\r\n', "/XML": xml, "/FO": f'"\\{name}","N/A","Ready"\r\n'}
    queries = []

    def run(argv, *_args, **_kwargs):
        queries.append(list(argv))
        return SimpleNamespace(returncode=0, stdout=next(text for key, text in out.items() if key in argv).encode())

    monkeypatch.setattr(service, "_run", run)
    return queries


def test_discovery_leaves_the_pre_wscript_task_for_the_next_logon(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_service as service

    home = tmp_path / "profile"
    home.mkdir()
    name = "Hermes_Gateway_" + service.service_suffix(home)
    queries = _schtasks(service, monkeypatch, name, _legacy_task_xml(str(_wrapper(home.resolve(), name))))
    assert service._windows(home.resolve(), time.monotonic() + 5) is None
    assert all(argv[0] == "whoami.exe" or argv[:2] == ["schtasks.exe", "/Query"] for argv in queries)


def test_discovery_still_refuses_a_task_running_another_script(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_service as service

    home = tmp_path / "profile"
    home.mkdir()
    name = "Hermes_Gateway_" + service.service_suffix(home)
    _schtasks(service, monkeypatch, name, _legacy_task_xml(str(home.resolve() / "gateway-service" / "other.cmd")))
    with pytest.raises(service.RuntimeStartError, match="service_identity_unverified") as refused:
        service._windows(home.resolve(), time.monotonic() + 5)
    assert "not wscript.exe" in refused.value.detail


def test_ensure_starts_a_stopped_gateway_beside_the_pre_wscript_task(tmp_path, monkeypatch):
    """The reported boot: Desktop's ensure met the old registration and refused with
    ``service_identity_unverified`` on every launch. It now takes the detached direct spawn manual
    ``hermes gateway start`` uses, and never runs or rewrites the task."""
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    name = "Hermes_Gateway_" + service.service_suffix(home)
    queries = _schtasks(service, monkeypatch, name, _legacy_task_xml(str(_wrapper(home.resolve(), name))))
    spawned = []
    monkeypatch.setattr(service.sys, "platform", "win32")
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: runtime.GatewayDiscovery("absent"))
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))
    result = runtime.ensure_gateway_runtime(home, timeout=0.3)
    assert result.reason_code != "service_identity_unverified"
    assert spawned == [home.resolve()]
    assert all(argv[0] == "whoami.exe" or argv[:2] == ["schtasks.exe", "/Query"] for argv in queries)
