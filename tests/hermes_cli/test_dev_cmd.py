from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.dev_sync import DevSyncError
from hermes_cli.subcommands.dev import _cmd_dev_sync


def test_sync_failure_exits_nonzero(tmp_path, capsys):
    args = SimpleNamespace(watch=False, only=None, desktop=False)
    with (
        patch("hermes_cli.subcommands.dev.dev_sync_run", side_effect=DevSyncError("node install failed")),
        pytest.raises(SystemExit) as exc,
    ):
        _cmd_dev_sync(args, tmp_path)

    assert exc.value.code == 1
    assert "Sync failed: node install failed" in capsys.readouterr().err


def test_watch_flag_is_forwarded(tmp_path):
    args = SimpleNamespace(watch=True, only=["web"], desktop=False)
    with patch("hermes_cli.subcommands.dev.dev_sync_run") as sync:
        _cmd_dev_sync(args, tmp_path)

    sync.assert_called_once_with(tmp_path, watch=True, only=["web"], desktop=False)
