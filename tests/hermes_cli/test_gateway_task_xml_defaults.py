"""Registered Task Scheduler XML defaults must preserve service admission (#139013)."""
import subprocess
import sys
import time
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("run_level,enabled,reason", [
    ("LeastPrivilege", "true", None),
    (None, "true", None),
    ("LeastPrivilege", None, None),
    (None, None, None),
    ("HighestAvailable", None, "service_identity_unverified"),
    ("", None, "service_identity_unverified"),
    (None, "false", "service_disabled"),
    (None, "", "service_disabled"),
])
def test_registered_defaults_keep_verified_task_startable(tmp_path, monkeypatch, run_level, enabled, reason):
    from hermes_cli import gateway_runtime_service as service
    from hermes_cli.gateway_windows import _build_gateway_vbs_script, _build_scheduled_task_xml

    home = tmp_path / "requested"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    script = home / "gateway.vbs"
    script.write_text(
        _build_gateway_vbs_script(sys.executable, str(home), str(home), ""),
        encoding="utf-8", newline="",
    )
    original = script.read_bytes()
    sid = "S-1-5-21-123"
    name = "Hermes_Gateway_" + service.service_suffix(home)
    task = ET.fromstring(_build_scheduled_task_xml(name, script, sid))
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    for parent_path, tag, value in [
        ("t:Principals/t:Principal", "RunLevel", run_level),
        ("t:Settings", "Enabled", enabled),
    ]:
        parent = task.find(parent_path, ns)
        element = parent.find("t:" + tag, ns)
        if value is None:
            parent.remove(element)
        else:
            element.text = value
    xml = ET.tostring(task, encoding="utf-16")
    calls = []

    def supervisor(argv, **kwargs):
        calls.append(list(argv))
        if argv[0] == "whoami.exe":
            output = f'"DOMAIN\\owner","{sid}"\n'
        elif "/XML" in argv:
            output = xml
        elif "/Query" in argv:
            output = f'"\\{name}","N/A","Ready"\n'
        else:
            assert list(argv) == ["schtasks.exe", "/Run", "/TN", name]
            output = b""
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(subprocess, "run", supervisor)
    observed = None
    try:
        found = service.discover_existing_gateway_service(home, deadline=time.monotonic() + 10)
        assert found is not None
        service.start_existing_gateway_service(found, deadline=time.monotonic() + 10)
    except service.RuntimeStartError as exc:
        observed = exc.reason
    assert observed == reason
    assert sum("/Run" in call for call in calls) == (0 if reason else 1)
    assert script.read_bytes() == original
