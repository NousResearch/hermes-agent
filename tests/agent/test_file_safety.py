import os

from unittest.mock import patch
from agent.file_safety import get_safe_write_root


def test_get_safe_write_root_unset() -> None:
    with patch.dict(os.environ, clear=True):
        assert get_safe_write_root() is None


def test_get_safe_write_root_empty() -> None:
    with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": ""}, clear=True):
        assert get_safe_write_root() is None


def test_get_safe_write_root_valid() -> None:
    with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": "/some/path"}, clear=True):
        with patch("os.path.realpath", return_value="/resolved/path"):
            with patch("os.path.expanduser", return_value="/expanded/path"):
                assert get_safe_write_root() == "/resolved/path"


def test_get_safe_write_root_with_tilde() -> None:
    with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": "~/my_project"}, clear=True):
        with patch("os.path.expanduser", return_value="/home/user/my_project"):
            with patch("os.path.realpath", return_value="/home/user/my_project"):
                assert get_safe_write_root() == "/home/user/my_project"


def test_get_safe_write_root_exception() -> None:
    with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": "/some/path"}, clear=True):
        with patch("os.path.expanduser", side_effect=Exception("mocked error")):
            assert get_safe_write_root() is None
