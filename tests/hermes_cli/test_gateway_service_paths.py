"""WSL service PATH deduplication, including the no-shell-scraping contract."""

from unittest.mock import patch

import pytest

from hermes_cli.gateway_service_paths import deduplicate_wsl_paths


@pytest.mark.parametrize("existing", [[], ["/mnt/c/windows/SYSTEM32/", "/opt/bin/"]])
def test_equivalent_windows_dirs_are_added_once_without_folding_linux_case(existing):
    system32 = "/mnt/c/WINDOWS/system32"
    powershell = "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0"
    candidates = [
        system32, "/mnt/c/Windows/System32/", powershell, powershell + "/",
        "/opt/Bin", "/opt/bin", "/opt/Bin/", "/", "",
    ]
    original_existing, original_candidates = existing[:], candidates[:]

    result = deduplicate_wsl_paths(existing, candidates)

    expected = [powershell, "/opt/Bin", "/"] if existing else [
        system32, powershell, "/opt/Bin", "/opt/bin", "/",
    ]
    assert result == expected
    assert existing == original_existing
    assert candidates == original_candidates


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("system", [False, True])
def test_generated_units_deduplicate_tool_dirs_without_scraping_shell_path(
    system, tmp_path, monkeypatch,
):
    from hermes_cli import gateway as gw

    powershell = "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0"
    resolved = {
        "powershell.exe": powershell + "/powershell.exe",
        "cmd.exe": "/mnt/c/Windows/System32/cmd.exe",
        "wsl.exe": "/mnt/c/WINDOWS/system32/wsl.exe",
    }
    monkeypatch.setattr(gw, "is_wsl", lambda: True)
    monkeypatch.setattr(gw.shutil, "which", resolved.get)
    monkeypatch.setenv("PATH", "/mnt/d/heavy-app/bin:/usr/bin")
    real_exists = gw.Path.exists
    monkeypatch.setattr(
        gw.Path, "exists", lambda path: str(path).startswith("/mnt/c/") or real_exists(path),
    )
    monkeypatch.setattr(
        gw, "_system_service_identity", lambda run_as_user: ("alice", "alice", str(tmp_path), 1000),
    )

    unit = gw.generate_systemd_unit(system=system, run_as_user="alice" if system else None)
    path_line = next(line for line in unit.splitlines() if line.startswith('Environment="PATH='))
    entries = path_line.removeprefix('Environment="PATH=').removesuffix('"').split(":")
    windows_entries = [entry for entry in entries if entry.startswith("/mnt/")]

    assert sum(entry.lower() == powershell.lower() for entry in windows_entries) == 1
    assert sum(entry.lower() == "/mnt/c/windows/system32" for entry in windows_entries) == 1
    assert all(not entry.endswith("/") for entry in windows_entries)
    assert "/mnt/d/heavy-app/bin" not in entries


def test_service_path_skips_nonexistent_node_modules(tmp_path):
    """Service PATH should not include node_modules/.bin if it doesn't exist."""
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    node_modules_bin = str(tmp_path / "node_modules" / ".bin")
    assert node_modules_bin not in dirs


def test_service_path_includes_node_modules_when_present(tmp_path):
    """Service PATH should include node_modules/.bin when it exists."""
    nm_bin = tmp_path / "node_modules" / ".bin"
    nm_bin.mkdir(parents=True)
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    assert str(nm_bin) in dirs
