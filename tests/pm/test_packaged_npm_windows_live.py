"""Opt-in MSIX execution proof, exclusively on a disposable GitHub-hosted Windows runner.

The regular test suite skips this fixture. The on-demand Windows workflow owns the opt-in,
package installation and removal. No certificates, PATH edits or Hermes installation are used.
"""

import base64
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration]

_PACKAGE_NAME = "HermesNpmIntegrationProbe"
_ALIAS = "hermes-npm-integration-probe.exe"


def _run(argv, *, timeout=180):
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=False,
    )
    assert result.returncode == 0, (
        f"Fixture command failed ({result.returncode}): {argv}\n"
        f"{result.stdout}\n{result.stderr}"
    )
    return result


def _powershell(command):
    return _run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$ErrorActionPreference='Stop'; " + command,
        ]
    )


def _manifest(package, arch):
    assets = package / "Assets"
    assets.mkdir()
    (assets / "Logo.png").write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
    (package / "AppxManifest.xml").write_text(
        f'''<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10" xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10" xmlns:uap5="http://schemas.microsoft.com/appx/manifest/uap/windows10/5" xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities" IgnorableNamespaces="uap uap5 rescap">
  <Identity Name="{_PACKAGE_NAME}" Publisher="CN=HermesNpmProbe, OID.2.25.311729368913984317654407730594956997722=1" Version="1.0.0.0" ProcessorArchitecture="{arch}" />
  <Properties><DisplayName>npm integration probe</DisplayName><PublisherDisplayName>CI test</PublisherDisplayName><Logo>Assets\\Logo.png</Logo></Properties>
  <Dependencies><TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.26100.0" /></Dependencies>
  <Resources><Resource Language="en-US" /></Resources>
  <Applications><Application Id="Probe" Executable="runtime\\python.exe" EntryPoint="Windows.FullTrustApplication">
    <uap:VisualElements DisplayName="npm integration probe" Description="Temporary npm execution test" BackgroundColor="transparent" Square150x150Logo="Assets\\Logo.png" Square44x44Logo="Assets\\Logo.png" />
    <Extensions><uap5:Extension Category="windows.appExecutionAlias"><uap5:AppExecutionAlias><uap5:ExecutionAlias Alias="{_ALIAS}" /></uap5:AppExecutionAlias></uap5:Extension></Extensions>
  </Application></Applications>
  <Capabilities><rescap:Capability Name="runFullTrust" /></Capabilities>
</Package>
''',
        encoding="utf-8",
    )


def _stage_tools(package):
    import pm
    from pm.lock import Facts, Lockfile
    from pm.packages import Nodejs, Npm
    from pm.paths import lockfile_path
    from pm.store import current_target, tree_digest

    python = pm.installed_package("python")
    assert python is not None
    shutil.copytree(python.path, package / "runtime")
    tools = package / "tools"
    tools.mkdir()
    lock = Lockfile(lockfile_path())
    target = current_target()
    for tool in (Nodejs(), Npm()):
        installed = pm.installed_package(tool.name)
        assert installed is not None
        version = lock.version(tool.name)
        assert version is not None and installed.binary is not None
        entry = tools / tool.store_entry(version, target)
        if tool.name == "node":
            entry.mkdir()
            shutil.copy2(installed.binary, entry / "node.exe")
        else:
            shutil.copytree(installed.path, entry)
        Facts(tools / "facts.json").record(
            tool.name,
            version,
            entry.name,
            tool.env(entry, target),
            tools,
            target=target,
            artifacts=[
                artifact["sha256"] for artifact in lock.artifacts(tool.name, target)
            ],
            digest=tree_digest(entry),
        )
    shutil.copy2(lock.path, package / "lock.json")
    (package / "manifest.json").write_text("{}", encoding="utf-8")
    return target.rsplit("-", 1)[1]


def test_packaged_npm_launch_and_lifecycle(tmp_path):
    if os.environ.get("HERMES_MSIX_INTEGRATION") != "1":
        pytest.skip(
            "native MSIX deployment requires the explicit disposable-runner CI opt-in"
        )
    assert os.environ.get("GITHUB_ACTIONS") == "true"
    assert os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"
    assert ctypes.windll.shell32.IsUserAnAdmin(), (
        "unsigned executable MSIX requires an elevated disposable runner"
    )
    assert not _powershell(
        f"Get-AppxPackage '{_PACKAGE_NAME}' | Select-Object -ExpandProperty PackageFullName"
    ).stdout.strip()
    sdk = Path(os.environ["ProgramFiles(x86)"]) / "Windows Kits/10/bin"
    makeappx = sorted(sdk.glob("10.*/x64/makeappx.exe"))[-1]
    package = tmp_path / "package"
    package.mkdir()
    shutil.copytree(
        Path(os.environ["HERMES_MSIX_FIXTURE_PATH"]), package, dirs_exist_ok=True
    )
    archive = tmp_path / "probe.msix"
    _run([str(makeappx), "pack", "/d", str(package), "/p", str(archive), "/o"])
    source = Path(__file__).resolve().parents[2]
    receipt = tmp_path / "result.json"
    try:
        _powershell(f"Add-AppxPackage -Path '{archive}' -AllowUnsigned")
        alias = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/WindowsApps" / _ALIAS
        result = _run(
            [
                str(alias),
                "-I",
                "-S",
                "-B",
                str(source / "tests/pm/fixtures/npm_package_probe.py"),
                str(source),
                str(receipt),
            ]
        )
        assert receipt.is_file(), result.stdout + result.stderr
        evidence = json.loads(receipt.read_text(encoding="utf-8"))
        assert "WindowsApps" in evidence["python"]
        assert evidence["identity_result"] == 122
        assert evidence["rows"][1]["returncode"] != 0
        assert all(row["returncode"] == 0 for row in evidence["rows"][2:])
        print(json.dumps(evidence, indent=2))
    finally:
        _powershell(f"Get-AppxPackage '{_PACKAGE_NAME}' | Remove-AppxPackage -AllUsers")
    assert not _powershell(
        f"Get-AppxPackage '{_PACKAGE_NAME}' | Select-Object -ExpandProperty PackageFullName"
    ).stdout.strip()


if __name__ == "__main__":
    assert os.environ.get("GITHUB_ACTIONS") == "true"
    assert os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"
    assert os.environ.get("HERMES_MSIX_INTEGRATION") == "1"
    assert sys.argv[1:2] == ["--stage"] and len(sys.argv) == 3
    destination = Path(sys.argv[2])
    destination.mkdir(parents=True)
    _manifest(destination, _stage_tools(destination))
