r"""A ``\\?\``-prefixed HERMES_HOME names the same gateway pipe (and home key) as the plain path."""

from pathlib import Path

import pytest

import hermes_constants as hc
from gateway.control_socket import windows_pipe_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"\\?\C:\hermes-home", r"C:\hermes-home"),
        (r"\\?\UNC\server\share\hermes-home", r"\\server\share\hermes-home"),
        (r"\\?\unc\server\share", r"\\server\share"),
        (r"C:\hermes-home", r"C:\hermes-home"),
        (r"\\server\share\hermes-home", r"\\server\share\hermes-home"),
        (r"\\.\pipe\x", r"\\.\pipe\x"),
        ("/home/user/.hermes", "/home/user/.hermes"),
    ],
)
def test_strip_extended_length_prefix(raw, expected):
    assert hc.strip_extended_length_prefix(raw) == expected


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("exists", [True, False])
def test_extended_length_home_gets_the_same_pipe_and_key(tmp_path: Path, exists):
    hc.reset_hermes_home_key_cache()
    home = tmp_path / "hermes-home"
    if exists:
        home.mkdir()
    extended = Path("\\\\?\\" + str(home))
    assert windows_pipe_name(extended) == windows_pipe_name(home)
    assert hc.hermes_home_key(extended) == hc.hermes_home_key(home)
