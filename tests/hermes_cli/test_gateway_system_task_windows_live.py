"""Real SYSTEM Task Scheduler stop/run on disposable administrator Windows runners.

No UAC is requested. Non-admin developer runs skip this case. The fixture task
and its processes are bounded and removed even when an assertion fails.
"""

import hashlib
from importlib import import_module
import subprocess
import sys
import uuid
from xml.sax.saxutils import escape

import psutil
import pytest

from tests.waiting import wait_until


@pytest.mark.platforms("windows")
def test_real_system_task_stops_and_restarts_with_its_original_principal(tmp_path):
    from hermes_cli import gateway_windows
    from hermes_cli.gateway_windows_task_control import process_is_system
    from hermes_cli.update_windows_tasks import definition_matches_home, task_operation

    name = "Hermes_Gateway_fixture_" + uuid.uuid4().hex
    service = tmp_path / "gateway-service"
    service.mkdir()
    marker = tmp_path / "pid.txt"
    child = tmp_path / "fixture.py"
    child.write_text(
        "import os, threading\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(str(os.getpid()), encoding='ascii')\n"
        "threading.Event().wait(90)\n",
        encoding="utf-8",
    )
    script = service / f"{name}.cmd"
    script.write_text(
        "@echo off\r\n"
        + subprocess.list2cmdline([sys.executable, str(child)])
        + "\r\n",
        encoding="utf-8",
    )
    xml_path = tmp_path / "task.xml"
    xml_path.write_text(
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
        '<Principals><Principal id="system"><UserId>S-1-5-18</UserId>'
        "<RunLevel>HighestAvailable</RunLevel></Principal></Principals>"
        "<Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><ExecutionTimeLimit>PT2M</ExecutionTimeLimit></Settings>"
        '<Actions Context="system"><Exec><Command>cmd.exe</Command>'
        f'<Arguments>/c "{escape(str(script))}"</Arguments></Exec></Actions></Task>',
        encoding="utf-16",
    )
    # Validate even on a non-admin host. TASK_VALIDATE_ONLY (1) registers nothing.
    scheduler = import_module("win32com.client").Dispatch("Schedule.Service")
    scheduler.Connect()
    scheduler.GetFolder("\\").RegisterTask(
        name, xml_path.read_text(encoding="utf-16"), 1, "SYSTEM", None, 5
    )
    if not gateway_windows._is_running_as_admin():
        pytest.skip(
            "XML validated; SYSTEM execution requires a disposable administrator runner"
        )
    owned = []
    created = False
    try:
        code, out, err = gateway_windows._exec_schtasks(
            ["/Create", "/TN", name, "/XML", str(xml_path), "/RU", "SYSTEM"]
        )
        assert code == 0, (out, err)
        created = True
        xml = task_operation(name, "query")
        assert xml is not None and definition_matches_home(xml, tmp_path, name)
        digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        for _round in range(2):
            marker.unlink(missing_ok=True)
            task_operation(name, "run", digest)
            wait_until(
                lambda: marker.exists() and marker.stat().st_size > 0,
                timeout=30,
                what="SYSTEM fixture PID",
            )
            pid = int(marker.read_text(encoding="ascii"))
            process = psutil.Process(pid)
            owned.append((pid, process.create_time()))
            assert process_is_system(pid)
            task_operation(name, "stop", digest)
            wait_until(
                lambda: not psutil.pid_exists(pid),
                timeout=15,
                what="original SYSTEM process exit",
            )
        assert owned[0][0] != owned[1][0]
    finally:
        if created:
            gateway_windows._exec_schtasks(["/End", "/TN", name])
        for pid, born in owned:
            try:
                process = psutil.Process(pid)
                if process.create_time() == born:
                    process.kill()
                    process.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
        if created:
            code, out, err = gateway_windows._exec_schtasks(
                ["/Delete", "/TN", name, "/F"]
            )
            assert code == 0, (out, err)
