"""Windows: every spelling of a protected Hermes path is denied, not just the plain one (#154282).

``\\\\?\\C:\\...`` and the local machine's administrative share (``\\\\localhost\\C$\\...``,
``\\\\127.0.0.1\\C$\\...``, ``\\\\<this computer>\\C$\\...``) open the same file as ``C:\\...``; the
name-based denies used to compare the resolved string, which keeps the prefix, so those spellings
read and wrote straight through. Dummy files only, in a throwaway home.
"""

import json
import socket
from unittest.mock import patch

import pytest

from hermes_agent.agent import file_safety as fs

pytestmark = pytest.mark.platforms("windows")

BS = "\\"


def _spellings(path: str) -> dict[str, str]:
    drive, rest = path[0], path[2:]
    def share(host: str) -> str:
        return BS * 2 + host + BS + drive + "$" + rest

    return {
        "extended-length": BS * 2 + "?" + BS + path,
        "extended-length, forward slashes": ("//?/" + path).replace(BS, "/"),
        "localhost share": share("localhost"),
        "localhost share, upper case": share("LOCALHOST"),
        "127.0.0.1 share": share("127.0.0.1"),
        "127.1.2.3 share": share("127.1.2.3"),
        "this computer's share": share(socket.gethostname()),
    }


@pytest.fixture
def home(tmp_path):
    home = tmp_path.resolve() / "hermes"
    for rel in ("mcp-tokens/server/token.json", "vault/vault.key", "auth.json"):
        target = home / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"secret": "DUMMY"}), encoding="utf-8")
    with patch.object(fs, "_hermes_home_path", return_value=home), \
            patch.object(fs, "_hermes_root_path", return_value=home):
        yield home


@pytest.mark.parametrize("rel", ["mcp-tokens/server/token.json", "vault/vault.key", "auth.json"])
def test_every_local_spelling_of_a_protected_file_is_read_denied(home, rel):
    plain = str(home / rel)
    assert fs.get_read_block_error(plain)
    for label, spelled in _spellings(plain).items():
        assert fs.get_read_block_error(spelled), label


@pytest.mark.parametrize("rel", ["mcp-tokens/server/token.json", "vault/vault.key"])
def test_every_local_spelling_of_a_protected_folder_is_write_denied(home, rel):
    plain = str(home / rel)
    assert fs.get_write_denied_error(plain)
    for label, spelled in _spellings(plain).items():
        assert fs.get_write_denied_error(spelled), label


def test_ordinary_files_stay_allowed_in_every_spelling(home):
    plain = str(home / "notes.txt")
    for label, spelled in {"plain": plain, **_spellings(plain)}.items():
        assert fs.get_read_block_error(spelled) is None, label
        assert fs.get_write_denied_error(spelled) is None, label


@pytest.mark.parametrize("path, expected", [
    ("\\\\?\\C:\\Users\\x\\a.txt", "C:\\Users\\x\\a.txt"),
    ("\\\\localhost\\C$", "C:\\"),
    ("\\\\localhost.\\d$\\data", "d:\\data"),
    ("\\\\fileserver.example.com\\C$\\x", "\\\\fileserver.example.com\\C$\\x"),  # a remote host stays as written
    ("\\\\localhost\\share\\x", "\\\\localhost\\share\\x"),  # not an administrative share
    ("C:\\plain\\path", "C:\\plain\\path"),
])
def test_local_drive_spelling(path, expected):
    assert fs.local_drive_spelling(path) == expected
