"""Native #104212 regression: portable Node inherits its destination ACL.

The real staging function consumes synthetic download bytes. The owned home
fixture has an explicit user grant; the owned temporary directory can receive
the protected descriptor used by hardened Python releases. Live Hermes is
never modified, and the current user's access is retained throughout.
"""

import io
import json
import subprocess
import tempfile
import uuid
import zipfile

import pytest

import hermes_constants as hc


@pytest.mark.windows_only
@pytest.mark.parametrize("hardened_temp", [False, True])
def test_staged_node_inherits_destination_permissions(tmp_path, monkeypatch, hardened_temp):
    home = tmp_path / "managed-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_TEST_ACL_HOME", str(home))
    # Pytest's own temp root may already carry OWNER RIGHTS, so distinguish the
    # destination's normal user grant from the security-release temp descriptor.
    setup = """
$path = $env:HERMES_TEST_ACL_HOME
$acl = [System.IO.Directory]::GetAccessControl($path, [System.Security.AccessControl.AccessControlSections]::Access)
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
  $identity, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
$acl.AddAccessRule($rule)
[System.IO.Directory]::SetAccessControl($path, $acl)
"""
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", setup],
                   capture_output=True, text=True, check=True, timeout=30,
                   creationflags=subprocess.CREATE_NO_WINDOW)
    version = "node-v22.99.0-win-x64"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(version + "/node.exe", b"fixture")
        archive.writestr(version + "/node_modules/npm/package.json", b"{}")
    monkeypatch.setattr(hc, "_fetch_url", lambda url, timeout: (
        stream.getvalue() if url.endswith(".zip") else (version + ".zip").encode()))
    original_temp = tempfile.TemporaryDirectory

    def protected_temp(*args, **kwargs):
        # Keep the simulated Python security-release behavior reproducible even
        # when the test runner itself predates that tempfile change.
        temporary = original_temp(dir=tmp_path)
        monkeypatch.setenv("HERMES_TEST_OWNED_TEMP", temporary.name)
        script = """
$path = $env:HERMES_TEST_OWNED_TEMP
$acl = [System.IO.Directory]::GetAccessControl($path, [System.Security.AccessControl.AccessControlSections]::Access)
$acl.SetAccessRuleProtection($true, $false)
foreach ($sid in @('S-1-5-18', 'S-1-5-32-544', 'S-1-3-4')) {
  $identity = [System.Security.Principal.SecurityIdentifier]::new($sid)
  $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
    $identity, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
  $acl.AddAccessRule($rule)
}
[System.IO.Directory]::SetAccessControl($path, $acl)
"""
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                                    capture_output=True, text=True, timeout=30,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            assert result.returncode == 0, result.stderr
        except BaseException:
            temporary.cleanup()
            raise
        return temporary

    if hardened_temp:
        monkeypatch.setattr(tempfile, "TemporaryDirectory", protected_temp)
    staged = hc._stage_windows_node_zip(home, "x64")
    assert staged is not None
    assert (staged / "node.exe").read_bytes() == b"fixture"
    monkeypatch.setenv("HERMES_TEST_ACL_HOME", str(home))
    monkeypatch.setenv("HERMES_TEST_ACL_STAGE", str(staged))
    script = """
function Describe-Acl($path) {
  $acl = Get-Acl -LiteralPath $path
  @{
    protected = $acl.AreAccessRulesProtected
    trustees = @($acl.Access | ForEach-Object { $_.IdentityReference.Value } | Sort-Object -Unique)
  }
}
@{home=(Describe-Acl $env:HERMES_TEST_ACL_HOME); stage=(Describe-Acl $env:HERMES_TEST_ACL_STAGE)} | ConvertTo-Json -Depth 5
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script], capture_output=True,
        text=True, check=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
    descriptor = json.loads(result.stdout)
    assert descriptor["stage"]["protected"] is False
    assert descriptor["stage"]["trustees"] == descriptor["home"]["trustees"]


@pytest.mark.parametrize("fault", ["invalid-zip", "missing-layout", "partial-extract"])
def test_failed_staging_preserves_live_tree_and_removes_owned_scratch(tmp_path, monkeypatch, fault):
    home = tmp_path / "managed-home"
    live = home / "node"
    live.mkdir(parents=True)
    sentinel = live / "node.exe"
    sentinel.write_bytes(b"working")
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("unrecognized-layout/file", b"fixture")
    payload = b"not a zip" if fault == "invalid-zip" else archive_bytes.getvalue()
    monkeypatch.setattr(hc, "_fetch_url", lambda url, timeout: (
        payload if url.endswith(".zip") else b"node-v22.99.0-win-x64.zip"))
    if fault == "partial-extract":
        def extract(_archive, destination):
            (destination / "partial").write_bytes(b"partial")
            raise OSError("disk full")
        monkeypatch.setattr(zipfile.ZipFile, "extractall", extract)
    assert hc._stage_windows_node_zip(home, "x64") is None
    assert sentinel.read_bytes() == b"working"
    assert sorted(path.name for path in home.iterdir()) == ["node"]


@pytest.mark.parametrize("boundary", ["unpack-exists", "unpack-denied", "stage-exists"])
def test_staging_never_removes_or_nests_into_unowned_paths(tmp_path, monkeypatch, boundary):
    home = tmp_path / "managed-home"
    home.mkdir()
    token = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(uuid, "uuid4", lambda: token)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("node-v22.99.0-win-x64/node.exe", b"candidate")
    monkeypatch.setattr(hc, "_fetch_url", lambda url, timeout: (
        payload.getvalue() if url.endswith(".zip") else b"node-v22.99.0-win-x64.zip"))
    # Cover the original shortened token and the full UUID independently of
    # which generation of the production staging implementation is running.
    owned_elsewhere = []
    for identifier in (token.hex, token.hex[:8]):
        path = home / (f"node.new-{identifier}" + (".unpack" if boundary.startswith("unpack") else ""))
        path.mkdir()
        (path / "sentinel").write_bytes(b"another operation")
        owned_elsewhere.append(path)
    if boundary == "unpack-denied":
        real_mkdir = type(home).mkdir

        def mkdir(path, *args, **kwargs):
            if path in owned_elsewhere:
                raise PermissionError("cannot claim this directory")
            return real_mkdir(path, *args, **kwargs)
        monkeypatch.setattr(type(home), "mkdir", mkdir)
    assert hc._stage_windows_node_zip(home, "x64") is None
    for path in owned_elsewhere:
        assert (path / "sentinel").read_bytes() == b"another operation"
        assert list(path.iterdir()) == [path / "sentinel"]
