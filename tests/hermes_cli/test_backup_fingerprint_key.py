"""``hermes backup`` never ships the per-install fingerprint key (agent/redact.py)."""

from pathlib import Path

from hermes_cli.backup import _SECRET_FILE_NAMES, _should_exclude


def test_secret_fingerprint_key_is_excluded_from_backups():
    """The HMAC key behind ``***#xxxxxx`` previews is per-install: a restore elsewhere mints its own."""
    assert _should_exclude(Path("secret_fingerprint.key"))
    assert "secret_fingerprint.key" not in _SECRET_FILE_NAMES
