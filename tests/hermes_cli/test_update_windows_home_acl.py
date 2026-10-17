"""Windows update ACL repair command contracts."""

from hermes_cli.update_cmd_maint import _windows_home_acl_grant_command


def test_windows_home_acl_grant_targets_the_current_user_sid_recursively(tmp_path):
    home = tmp_path / "hermes home"

    assert _windows_home_acl_grant_command(home, "S-1-5-21-100-200-300-400") == [
        "icacls",
        str(home),
        "/grant",
        "*S-1-5-21-100-200-300-400:(OI)(CI)F",
        "/T",
        "/C",
        "/Q",
    ]
