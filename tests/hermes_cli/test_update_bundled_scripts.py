import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.main import cmd_update


def _make_run_side_effect(branch="main", commit_count="1"):
    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")
        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


def test_update_syncs_bundled_helper_scripts(capsys):
    args = SimpleNamespace()

    with patch("shutil.which", side_effect={"uv": "/usr/bin/uv", "npm": "/usr/bin/npm"}.get), patch(
        "subprocess.run"
    ) as mock_run, patch(
        "tools.skills_sync.sync_skills",
        return_value={"copied": [], "updated": [], "user_modified": [], "cleaned": []},
    ), patch(
        "tools.bundled_scripts_sync.sync_bundled_scripts",
        return_value={"linked": ["telegram-healthcheck-stateful"], "updated": [], "missing": []},
    ) as mock_sync_scripts, patch(
        "hermes_cli.profiles.list_profiles",
        return_value=[],
    ), patch(
        "hermes_cli.main._update_node_dependencies"
    ), patch(
        "hermes_cli.main._build_web_ui"
    ), patch(
        "hermes_cli.main._install_python_dependencies_with_optional_fallback"
    ), patch(
        "hermes_cli.main._clear_bytecode_cache",
        return_value=0,
    ):
        mock_run.side_effect = _make_run_side_effect()
        cmd_update(args)

    mock_sync_scripts.assert_called_once_with(quiet=True)
    out = capsys.readouterr().out
    assert "Syncing bundled helper scripts" in out
    assert "telegram-healthcheck-stateful" in out
