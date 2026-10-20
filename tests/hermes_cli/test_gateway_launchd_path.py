"""The LaunchAgent's PATH keeps the user's shell PATH but never a package-manager environment dir."""

import os
import plistlib

import hermes_cli.gateway as gateway_cli


def test_launchd_plist_path_drops_package_manager_environment_dirs(monkeypatch, tmp_path):
    """#83699: every launcher prepends the selected PM environment's venv/bin to PATH. Baked into
    the plist it dies with the next environment rotation, and the PATH-masking staleness check
    never rewrites it. Only PM environment dirs are dropped; the user's own PATH survives."""
    from pm import environments

    installs = tmp_path / "installs"
    env_bin = installs / "0123abcd" / "environments" / "feedbeef" / "venv" / "bin"
    env_bin.mkdir(parents=True)
    gone_bin = installs / "0123abcd" / "environments" / "deadbeef" / "venv" / "bin"  # already rotated away
    user_bin = tmp_path / "custom-bin"
    user_bin.mkdir()
    monkeypatch.setattr(environments, "installs_root", lambda: installs)
    monkeypatch.setenv("PATH", os.pathsep.join([str(env_bin), str(gone_bin), str(user_bin), "/usr/bin"]))

    entries = plistlib.loads(gateway_cli.generate_launchd_plist().encode())["EnvironmentVariables"]["PATH"].split(":")

    assert str(env_bin) not in entries
    assert str(gone_bin) not in entries
    assert str(user_bin) in entries
    assert "/usr/bin" in entries
