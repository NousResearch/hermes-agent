"""A failed macOS packaged update keeps what tells its causes apart.

The stable rehearsal rc.1-v0.23.2 showed "Hermes Agent quit unexpectedly" after
Update now, and its artifact could not say whether OLD crashed on quit, NEW
crashed at launch, or ShipIt never swapped: none of ShipIt's log, the crash
reports or the installed bundle's state were kept.
"""

import os
from pathlib import Path
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[3]
COLLECTOR = ROOT / "tests" / "install" / "e2e-assets" / "mac-update-evidence.sh"


def _write(path: Path, text: str, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.mark.platforms("posix")
def test_collects_shipit_logs_new_crash_reports_and_installed_state(tmp_path):
    now = time.time()
    since = _write(tmp_path / "update-started", "", mtime=now - 600)
    sandbox, runner = tmp_path / "sandbox-home", tmp_path / "runner-home"
    shipit = runner / "Library/Caches/com.nousresearch.hermes-bundled.ShipIt"
    _write(shipit / "ShipIt_stderr.log", "ShipIt: install failed: code object is not signed\n")
    _write(shipit / "ShipItState.plist", "<plist/>")
    staged = _write(shipit / "update.Xy/Hermes Agent.app/Contents/Resources/payload.bin", "x" * 4096)
    reports = runner / "Library/Logs/DiagnosticReports"
    _write(reports / "Hermes Agent-before.ips", '{"app_name":"Hermes Agent","app_version":"0.21.0"}\n', mtime=now - 3600)
    _write(reports / "Hermes Agent-after.ips", '{"app_name":"Hermes Agent","app_version":"0.22.0"}\n')
    _write(reports / "Retired/ShipIt-after.ips", '{"app_name":"ShipIt"}\n')
    hermes_home = sandbox / ".hermes"
    _write(hermes_home / "logs/desktop.log", "[updater] quitAndInstall\n")
    app = tmp_path / "Applications/Hermes Agent.app"
    _write(app / "Contents/Resources/install-stamp.json", '{"tag": "v0.22.0"}')

    out = tmp_path / "evidence"
    result = subprocess.run(
        ["bash", str(COLLECTOR), "--out", str(out), "--since", str(since), "--app", str(app),
         "--home", str(sandbox), "--home", str(runner), "--hermes-home", str(hermes_home)],
        capture_output=True, text=True, timeout=60, check=False,
    )

    assert result.returncode == 0, result.stderr
    copied = out / "caches/home2/com.nousresearch.hermes-bundled.ShipIt"
    assert (copied / "ShipIt_stderr.log").read_text() == "ShipIt: install failed: code object is not signed\n"
    assert (copied / "ShipItState.plist").is_file()
    assert not list(copied.rglob("payload.bin"))
    assert "update.Xy" in (copied / "listing.txt").read_text()
    assert staged.is_file()
    # The collector also reads the host's /Library/Logs/DiagnosticReports, where
    # a CI runner's own tools can drop reports mid-test, so check only ours.
    copied_reports = {p.name for p in (out / "crash-reports").iterdir()}
    assert {"Hermes Agent-after.ips", "ShipIt-after.ips"} <= copied_reports
    assert "Hermes Agent-before.ips" not in copied_reports
    assert (out / "hermes-logs/desktop.log").read_text() == "[updater] quitAndInstall\n"
    assert '"tag": "v0.22.0"' in (out / "installed-app.txt").read_text()
    assert "code object is not signed" in result.stdout
    assert '"app_version":"0.22.0"' in result.stdout
