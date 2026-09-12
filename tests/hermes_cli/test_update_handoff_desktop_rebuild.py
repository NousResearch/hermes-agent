"""The current-checkout repair path must rebuild the Desktop app (#97343).

A retry with no new commits must still prepare the Desktop product selected
before the update. Failure must stop before configuration and success reporting.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import update_cmd


def test_current_checkout_repair_rebuilds_desktop_under_project_root(tmp_path):
    """The retry preserves the pre-update desktop selection and checkout root."""
    completion = MagicMock(return_value=True)
    with (
        patch.object(update_cmd, "_prepare_updated_checkout") as prepare,
        patch.object(update_cmd, "_m") as m,
        patch.object(update_cmd, "_check_and_apply_config_migration"),
        patch.object(update_cmd, "_print_verified_update_completion", completion),
    ):
        m.return_value.PROJECT_ROOT = tmp_path
        complete = update_cmd._repair_current_checkout(
            assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
            had_desktop_app_before_update=True, upstream_checked=True,
        )

    assert complete is True
    prepare.assert_called_once_with(tmp_path, desktop=True)
    completion.assert_called_once_with("✓ Already up to date!")


def test_failed_desktop_rebuild_withholds_success_completion(tmp_path):
    """A failed build propagates before config migration or success reporting."""
    completion = MagicMock(return_value=True)
    with (
        patch.object(update_cmd, "_m") as m,
        patch.object(update_cmd, "_check_and_apply_config_migration") as migrate,
        patch.object(update_cmd, "_prepare_updated_checkout", side_effect=RuntimeError("desktop build failed")),
        patch.object(update_cmd, "_print_verified_update_completion", completion),
    ):
        m.return_value.PROJECT_ROOT = tmp_path
        with pytest.raises(RuntimeError, match="desktop build failed"):
            update_cmd._repair_current_checkout(
                assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
                had_desktop_app_before_update=True, upstream_checked=True,
            )

    migrate.assert_not_called()
    completion.assert_not_called()
