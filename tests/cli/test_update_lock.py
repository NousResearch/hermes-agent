from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from hermes_cli import main


def test_update_lock_blocks_second_holder(tmp_path: Path, capsys):
    with patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        first = main._acquire_update_lock()
        try:
            second = main._acquire_update_lock()
            assert second is None
            assert "Another Hermes update is already running" in capsys.readouterr().out
        finally:
            main._release_update_lock(first)


def test_update_lock_released_allows_reacquire(tmp_path: Path):
    with patch("hermes_cli.config.get_hermes_home", return_value=tmp_path):
        first = main._acquire_update_lock()
        main._release_update_lock(first)
        second = main._acquire_update_lock()
        try:
            assert second is not None
        finally:
            main._release_update_lock(second)
