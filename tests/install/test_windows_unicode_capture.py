from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_windows_bootstrap_captures_uv_paths_as_utf8():
    """The PowerShell 5.1 native pipeline decodes uv paths with the OEM code page."""
    script = (ROOT / "scripts" / "install.ps1").read_text(encoding="utf-8-sig")
    helper_start = script.index("function Invoke-Utf8NativeCapture")
    helper_end = script.index("# Interactive runs collapse child-process output", helper_start)
    helper = script[helper_start:helper_end]
    bootstrap_start = script.index("function Get-BootstrapPython")
    bootstrap_end = script.index("function Invoke-BootstrapPm", bootstrap_start)
    bootstrap = script[bootstrap_start:bootstrap_end]

    assert "$psi.StandardOutputEncoding = $utf8" in helper
    assert "$psi.StandardErrorEncoding = $utf8" in helper
    assert "Invoke-Utf8NativeCapture $uv" in bootstrap
    assert "Invoke-Native { & $uv python find" not in bootstrap
