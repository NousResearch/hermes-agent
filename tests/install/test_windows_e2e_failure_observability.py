from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AHK = ROOT / "tests" / "install" / "e2e-assets" / "install-and-launch.ahk"
WINDOWS_E2E = ROOT / "tests" / "install" / "windows-e2e.ps1"
BOOTSTRAP_RS = (
    ROOT / "apps" / "bootstrap-installer" / "src-tauri" / "src" / "bootstrap.rs"
)
FAILURE_MARKER = "bootstrap FAILED"


def test_ahk_checks_bootstrap_failure_before_success_or_timeout():
    source = AHK.read_text(encoding="utf-8")
    poll_loop = source[source.index("while (A_TickCount < waitDeadline)") :]

    failure_check = poll_loop.index(f'BootstrapLogContains("{FAILURE_MARKER}")')
    completion_check = poll_loop.index('BootstrapLogContains("bootstrap complete")')
    timeout_error = poll_loop.index("install did not finish within 45 minutes")

    assert failure_check < completion_check < timeout_error
    assert "throw Error(Format(\"install failed:" in poll_loop


def test_windows_driver_includes_installer_failure_reason():
    source = WINDOWS_E2E.read_text(encoding="utf-8")

    assert 'Select-String "bootstrap FAILED"' in source
    assert "installer reported: $failureReason" in source
    assert 'Select-String "Unhandled error:"' in source
    assert "AutoHotkey driver reported: $ahkFailure" in source

    accept_success = source.index(
        'Assert-True ($ahk.ExitCode -eq 0) "AutoHotkey driver exited 0'
    )
    assert source.index("if ($failureReason)") < accept_success
    assert source.index("if ($ahkFailure)") < accept_success


def test_every_bootstrap_error_writes_the_e2e_failure_marker():
    source = BOOTSTRAP_RS.read_text(encoding="utf-8")
    spawn_start = source.index("tokio::spawn(async move")
    task = source[spawn_start : source.index("Ok(())", spawn_start)]

    assert 'if let Err(err) = &result {' in task
    assert '"bootstrap FAILED"' in task
