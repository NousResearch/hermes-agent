"""The current-checkout repair path must rebuild the Desktop app (#97343).

A Windows git install runs `hermes update` from `hermes.exe`, which reexecs a
venv-Python child to finish the dependency sync. That child completes through
``_repair_node_deps_on_current_checkout`` / the hand-off repair branch, never
through the commits-pulled path that owns the Desktop rebuild — so a
successful-looking update left the packaged desktop app on the previous build.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hermes_cli import update_cmd


def test_current_checkout_repair_rebuilds_desktop_under_project_root():
    """The repair passes PROJECT_ROOT/apps/desktop and the pre-update flag."""
    completion = MagicMock(return_value=True)
    with (
        patch.object(update_cmd, "_update_node_dependencies", return_value=[]),
        patch.object(update_cmd, "_m") as m,
        patch.object(update_cmd, "_check_and_apply_config_migration"),
        patch.object(
            update_cmd, "_rebuild_desktop_after_update", return_value=True
        ) as rebuild,
    ):
        m.return_value.PROJECT_ROOT = update_cmd.Path("/fake/hermes")
        complete = update_cmd._repair_node_deps_on_current_checkout(
            completion, had_desktop_app_before_update=True
        )

    assert complete is True
    rebuild.assert_called_once()
    assert rebuild.call_args[0][0] == update_cmd.Path("/fake/hermes/apps/desktop")
    assert rebuild.call_args[1]["had_desktop_app_before_update"] is True
    completion.assert_called_once_with("✓ Already up to date!")


def test_failed_desktop_rebuild_withholds_success_completion():
    """A failed rebuild must not report success and must return False."""
    completion = MagicMock(return_value=True)
    with (
        patch.object(update_cmd, "_update_node_dependencies", return_value=[]),
        patch.object(update_cmd, "_m") as m,
        patch.object(update_cmd, "_check_and_apply_config_migration"),
        patch.object(update_cmd, "_rebuild_desktop_after_update", return_value=False),
    ):
        m.return_value.PROJECT_ROOT = update_cmd.Path("/fake/hermes")
        complete = update_cmd._repair_node_deps_on_current_checkout(completion)

    assert complete is False
    for call in completion.call_args_list:
        assert not call[0][0].startswith("✓")


def test_handoff_venv_repair_finishes_node_and_web_phase(tmp_path):
    """A successful Python repair must not bypass the remaining update work."""
    project_root = tmp_path / "hermes"
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()

    with (
        patch.object(update_cmd, "venv_python_path", return_value=venv_python),
        patch.object(update_cmd, "_pip_install_prefix", return_value=(["uv", "pip"], None)),
        patch.object(update_cmd, "_venv_core_imports_healthy", return_value=(True, "ok")),
        patch.object(update_cmd, "_write_update_incomplete_marker"),
        patch.object(update_cmd, "_update_node_dependencies", return_value=[]) as update_node,
        patch.object(update_cmd, "_check_and_apply_config_migration"),
        patch.object(update_cmd, "_rebuild_desktop_after_update", return_value=True),
        patch.object(update_cmd, "_print_verified_update_completion", return_value=True) as completion,
        patch("hermes_cli.managed_uv.ensure_uv", return_value="uv"),
        patch.object(update_cmd, "_m") as m,
    ):
        m.return_value.PROJECT_ROOT = project_root
        complete = update_cmd._repair_venv_on_current_checkout(
            assume_yes=True,
            gateway_mode=False,
            pre_update_snapshot_id=None,
            had_desktop_app_before_update=False,
            active_lazy_features=(),
            active_tool_dependencies=(),
            _windows_gateway_resume=None,
        )

    assert complete is True
    update_node.assert_called_once_with()
    m.return_value._build_web_ui.assert_called_once_with(project_root / "web")
    completion.assert_called_once_with("✓ Update complete!")
