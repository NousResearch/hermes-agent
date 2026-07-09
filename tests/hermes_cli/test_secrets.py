"""Tests for hermes_cli.secrets — Windows Credential Manager secret access."""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from hermes_cli import secrets


# --- get() ---

def test_get_returns_keyring_value_when_present():
    with patch.object(secrets, "_keyring_available", return_value=True), \
         patch.object(secrets, "_read_env_value", return_value=None):
        with patch("keyring.get_password", return_value="from-keyring"):
            v, src = secrets.get("MY_KEY")
    assert v == "from-keyring"
    assert src == "keyring"


def test_get_falls_back_to_environ_when_keyring_empty():
    with patch.object(secrets, "_keyring_available", return_value=True), \
         patch.object(secrets, "_read_env_value", return_value=None), \
         patch.dict(os.environ, {"MY_KEY_ENV": "from-env"}, clear=False):
        # keyring returns None, so falls through to env
        with patch("keyring.get_password", return_value=None):
            v, src = secrets.get("MY_KEY_ENV")
    assert v == "from-env"
    assert src == "env"


def test_get_falls_back_to_env_file_when_env_missing():
    with patch.object(secrets, "_keyring_available", return_value=True), \
         patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MY_KEY_FILE", None)
        with patch("keyring.get_password", return_value=None):
            with patch.object(secrets, "_read_env_value", return_value="from-file"):
                v, src = secrets.get("MY_KEY_FILE")
    assert v == "from-file"
    assert src == "env_file"


def test_get_returns_none_when_not_found():
    with patch.object(secrets, "_keyring_available", return_value=True), \
         patch.object(secrets, "_read_env_value", return_value=None):
        os.environ.pop("MY_NONEXISTENT_KEY", None)
        with patch("keyring.get_password", return_value=None):
            v, src = secrets.get("MY_NONEXISTENT_KEY")
    assert v is None
    assert src == ""


def test_get_skips_keyring_when_unavailable():
    """If keyring is broken (e.g. headless CI), still fall back to env/file."""
    with patch.object(secrets, "_keyring_available", return_value=False), \
         patch.dict(os.environ, {"MY_KEY2": "from-env"}, clear=False):
        v, src = secrets.get("MY_KEY2")
    assert v == "from-env"
    assert src == "env"


def test_get_handles_keyring_exception():
    """If keyring throws, fall through to env/file (defensive)."""
    with patch.object(secrets, "_keyring_available", return_value=True), \
         patch.dict(os.environ, {"MY_KEY3": "from-env"}, clear=False):
        with patch("keyring.get_password", side_effect=OSError("credential store unavailable")):
            v, src = secrets.get("MY_KEY3")
    assert v == "from-env"
    assert src == "env"


# --- set() ---

def test_set_stores_value_when_keyring_available():
    with patch.object(secrets, "_keyring_available", return_value=True):
        with patch("keyring.set_password") as mock_set:
            ok = secrets.set("MY_KEY", "my-value")
    assert ok is True
    mock_set.assert_called_once_with("hermes", "MY_KEY", "my-value")


def test_set_returns_false_when_keyring_unavailable():
    with patch.object(secrets, "_keyring_available", return_value=False):
        ok = secrets.set("MY_KEY", "my-value")
    assert ok is False


def test_set_returns_false_on_exception():
    with patch.object(secrets, "_keyring_available", return_value=True):
        with patch("keyring.set_password", side_effect=OSError("store broken")):
            ok = secrets.set("MY_KEY", "my-value")
    assert ok is False


# --- delete() ---

def test_delete_removes_from_keyring():
    with patch.object(secrets, "_keyring_available", return_value=True):
        with patch("keyring.delete_password") as mock_del:
            ok = secrets.delete("MY_KEY")
    assert ok is True
    mock_del.assert_called_once_with("hermes", "MY_KEY")


def test_delete_returns_false_when_keyring_unavailable():
    with patch.object(secrets, "_keyring_available", return_value=False):
        ok = secrets.delete("MY_KEY")
    assert ok is False


# --- migrate_from_env() ---

def test_migrate_no_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(secrets, "ENV_FILE", tmp_path / "missing.env")
    result = secrets.migrate_from_env()
    assert result == {}


def test_migrate_keyring_unavailable(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("KEY1=value1\nKEY2=value2\n")
    monkeypatch.setattr(secrets, "ENV_FILE", env)
    with patch.object(secrets, "_keyring_available", return_value=False):
        result = secrets.migrate_from_env()
    assert result == {"*": "skipped_keyring_unavailable"}


def test_migrate_stores_all_keys(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment line\n"
        "KEY1=value1\n"
        "KEY2=value2\n"
        "\n"  # blank line
        "EMPTY=\n"
        "KEY3=value with spaces\n"
    )
    monkeypatch.setattr(secrets, "ENV_FILE", env)
    with patch.object(secrets, "_keyring_available", return_value=True):
        with patch.object(secrets, "set", return_value=True) as mock_set:
            result = secrets.migrate_from_env()
    # KEY1, KEY2, KEY3 should be stored. EMPTY should be skipped_empty.
    assert result["KEY1"] == "stored"
    assert result["KEY2"] == "stored"
    assert result["KEY3"] == "stored"
    assert result["EMPTY"] == "skipped_empty"
    assert mock_set.call_count == 3


def test_migrate_records_failed_storage(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("KEY1=value1\n")
    monkeypatch.setattr(secrets, "ENV_FILE", env)
    with patch.object(secrets, "_keyring_available", return_value=True):
        with patch.object(secrets, "set", return_value=False):
            result = secrets.migrate_from_env()
    assert result["KEY1"] == "skipped_failed"


# --- _read_env_value() ---

def test_read_env_value_returns_value(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nKEY1=value1\nKEY2=\"value2 with spaces\"\n")
    assert secrets._read_env_value("KEY1", path=env) == "value1"
    assert secrets._read_env_value("KEY2", path=env) == "value2 with spaces"


def test_read_env_value_returns_none_for_missing_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("KEY1=value1\n")
    assert secrets._read_env_value("OTHER_KEY", path=env) is None


def test_read_env_value_returns_none_for_missing_file(tmp_path):
    assert secrets._read_env_value("KEY1", path=tmp_path / "missing.env") is None


def test_read_env_value_handles_garbled_file(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"\xff\xfe\x00garbled\xff bytes here\n")
    # Should not raise, should return None or whatever it can extract
    result = secrets._read_env_value("KEY1", path=env)
    # Either None (nothing matched) or whatever garbage it parsed — we just
    # want to confirm it doesn't raise.
    assert result is None or isinstance(result, str)