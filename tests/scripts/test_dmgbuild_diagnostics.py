"""DMG diagnostics observe failures without changing the native result."""
import io
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.bundles import dmgbuild_diagnostics as diagnostics


def test_detach_diagnostics_run_before_cleanup_and_preserve_results():
    events = []
    attachment = {"system-entities": [
        {"dev-entry": "/dev/disk24"},
        {"dev-entry": "/dev/disk25s1", "mount-point": "/Volumes/Install Hermes Agent"},
    ]}
    failure = (16, "hdiutil: couldn't eject disk25 - Resource busy")

    def native(command, *args, **kwargs):
        events.append((command, args, kwargs))
        if command == "attach":
            return 0, attachment
        return (0, "detached") if "-force" in args or command != "detach" else failure

    def report(target, image, output):
        events.append(("diagnostic", target, image, output))

    observed = diagnostics.wrap_hdiutil(native, report=report)
    observed("attach", "-nobrowse", "/tmp/our staging.dmg")
    assert observed("detach", "/dev/disk25s1", plist=False) is failure
    observed("detach", "-force", "/dev/disk25s1", plist=False)
    assert [event[0] for event in events] == ["attach", "detach", "diagnostic", "detach"]
    assert events[2][1] == "/dev/disk25s1"
    assert events[2][2] == {"image": "/tmp/our staging.dmg", "entities": attachment["system-entities"]}
    assert events[2][3] == failure[1]

    def broken_report(*args):
        raise OSError("diagnostic command unavailable")

    errors = io.StringIO()
    broken = diagnostics.wrap_hdiutil(native, report=broken_report, stream=errors)
    assert broken("detach", "/dev/disk25s1", plist=False) is failure
    assert "diagnostic command unavailable" in errors.getvalue()
    assert broken("convert", "input", "-o", "output")[0] == 0

    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["timeout"] <= 15
        if argv[-1] == "/usr/bin/true":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "COMMAND PID PPID USER FD TYPE NAME\nmds 4321 1 root 7r DIR /Volumes/Install Hermes Agent\n", "")

    diagnostics.report_detach_failure("/dev/disk25s1", events[2][2], failure[1], run=run, stream=errors)
    text = errors.getvalue()
    assert "mds 4321 1 root 7r" in text
    assert any("+f" in argv and argv[-1] == "/Volumes/Install Hermes Agent" for argv in calls)
    file_query = next(argv for argv in calls if "-f" in argv)
    assert "/tmp/our staging.dmg" in file_query and "/dev/disk24" in file_query
    assert "/dev/disk25s1" in file_query and "/dev/rdisk25s1" in file_query
    assert all("+D" not in argv for argv in calls)

    def timeout(argv, **kwargs):
        if argv[-1] == "/usr/bin/true":
            return subprocess.CompletedProcess(argv, 1, "", "sudo: a password is required")
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"partial holder data")

    errors = io.StringIO()
    diagnostics.report_detach_failure("/dev/disk25s1", events[2][2], failure[1], run=timeout, stream=errors)
    assert "unprivileged" in errors.getvalue()
    assert "partial holder data" in errors.getvalue()
    assert "timed out" in errors.getvalue()


def test_diagnostic_entrypoint_preserves_cli_arguments_and_failure(tmp_path):
    # This package replays the supplier call contract, not a native DMG build.
    package = tmp_path / "dmgbuild"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text(
        'def hdiutil(command, *args, **kwargs):\n'
        '    print("fixture native", command, *args, flush=True)\n'
        '    if command == "attach":\n'
        '        return 0, {"system-entities": [{"dev-entry": "/dev/hermes-dmg-fixture-not-a-device"}]}\n'
        '    return (0, "cleanup") if "-force" in args else (16, "Resource busy")\n', encoding="utf-8")
    (package / "__main__.py").write_text(
        'import json, sys\nfrom . import core\n'
        'def main():\n'
        '    print(json.dumps(sys.argv[1:]), flush=True)\n'
        '    core.hdiutil("attach", "-nobrowse", "/hermes-dmg-fixture-no-image.dmg")\n'
        '    result = core.hdiutil("detach", "/dev/hermes-dmg-fixture-not-a-device", plist=False)\n'
        '    core.hdiutil("detach", "-force", "/dev/hermes-dmg-fixture-not-a-device", plist=False)\n'
        '    raise SystemExit(result[0])\n', encoding="utf-8")
    args = ["-s", "settings with spaces.json", "Install Hermes Agent", "output.dmg"]
    result = subprocess.run([sys.executable, str(Path(diagnostics.__file__)), *args],
                            cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path)},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 16
    assert json.dumps(args) in result.stdout
    assert result.stdout.index("[dmg-detach]") < result.stdout.index("fixture native detach -force")


@pytest.mark.platforms("macos")
def test_native_busy_image_reports_the_process_holding_its_file(tmp_path):
    image, mount = tmp_path.resolve() / "held.dmg", tmp_path.resolve() / "mount"
    mount.mkdir()
    subprocess.run(["/usr/bin/hdiutil", "create", "-size", "16m", "-fs", "HFS+",
                    "-volname", "Hermes diagnostic fixture", str(image)],
                   check=True, capture_output=True, timeout=60)

    def native(command, *args, plist=True):
        argv = ["/usr/bin/hdiutil", command, *args, *(["-plist"] if plist else [])]
        result = subprocess.run(argv, capture_output=True, timeout=60)
        output = plistlib.loads(result.stdout) if plist and result.returncode == 0 else (result.stdout + result.stderr).decode()
        return result.returncode, output

    log = io.StringIO()
    observed = diagnostics.wrap_hdiutil(native, report=lambda *args: diagnostics.report_detach_failure(*args, stream=log))
    device = None
    try:
        code, info = observed("attach", "-nobrowse", "-mountpoint", str(mount), str(image))
        assert code == 0, info
        device = next(e["dev-entry"] for e in info["system-entities"] if e.get("mount-point"))
        held = mount / "held-open.txt"
        with held.open("w") as handle:
            handle.write("owned test handle\n")
            handle.flush()
            code, output = observed("detach", device, plist=False)
            assert code != 0, "native fixture must reproduce a busy detach"
            report = log.getvalue()
            assert str(os.getpid()) in report
            assert str(held) in report
            assert "COMMAND" in report and "PID" in report
    finally:
        if device:
            subprocess.run(["/usr/bin/hdiutil", "detach", "-force", device], capture_output=True, timeout=60)
